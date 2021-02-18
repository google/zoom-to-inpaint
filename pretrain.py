# coding=utf-8
# Copyright 2021 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Training pipeline for pre-training stage.

The TrainingPipeline class defines the training pipeline for the pre-training
stage, where the coarse, refinement, and SR network can be pre-trained
separately with distributed training using MirroredStrategy.

The network to train can be chosen with --network_mode.
Configurations are defined in pretrain_flags() in config.py, and the training
setting for SR is in pretrain_sr.cfg.
"""

import os

import net
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import utils_data
import utils_loss
import utils_mask
import utils_misc
import utils_summary


class TrainingPipeline():
  """Training pipeline."""

  def __init__(self, flags):
    self.log_dir = os.path.join(flags.work_dir, flags.network_mode, 'logs')
    self.ckpt_dir = os.path.join(flags.work_dir, flags.network_mode, 'ckpt')
    self.network_mode = flags.network_mode
    self.batch_size = flags.batch_size
    self.num_epochs = flags.num_epochs
    self.lr_boundary = flags.lr_boundary
    self.lr_values = flags.lr_values
    self.patch_size = flags.patch_size
    self.lambda_l1 = flags.lambda_l1
    self.lambda_vgg = flags.lambda_vgg
    self.vgg_layer = flags.vgg_layer
    self.dataset = flags.dataset
    self.sr_scale = flags.sr_scale

    # set values dependent on dataset
    if self.dataset == 'div2k':
      # num_val: number of validation images
      self.num_val = (100 // self.batch_size) * self.batch_size
      # buffer_size: size of buffer for tfds
      if (self.network_mode == 'coarse') or (self.network_mode == 'refine'):
        self.buffer_size = 200  # due to memory
      elif self.network_mode == 'sr':
        self.buffer_size = 2000
    else:
      self.num_val = 50 * self.batch_size
      self.buffer_size = 2000
    # logging
    if not os.path.exists(os.path.join(self.log_dir, 'logs.txt')):
      utils_misc.check_folder(self.log_dir)
      with open(os.path.join(self.log_dir, 'logs.txt'), 'w') as log:
        log.write('Epoch \t val_PSNR \t total_loss \t L1 loss \t VGG loss \n')

  def build_model(self):
    """Build model with distributed training using MirroredStrategy."""
    self.strategy = tf.distribute.MirroredStrategy()
    # load dataset - only take num_val number of validation images as val_ds
    train_ds, ds_info = tfds.load(
        self.dataset,
        split='train',
        shuffle_files=True,
        as_supervised=True,
        with_info=True)
    val_ds = tfds.load(
        self.dataset,
        split='validation[:{}]'.format(self.num_val),
        as_supervised=True)
    if (self.network_mode == 'coarse') or (self.network_mode == 'refine'):
      data = utils_data.DatasetPre(
          self.batch_size, self.patch_size, self.buffer_size, mask_type='small')
    elif self.network_mode == 'sr':
      data = utils_data.DatasetFull(
          self.sr_scale, self.batch_size, self.patch_size, self.buffer_size,
          mask_type=None)
    self.train_ds = data.get_train_dataset(train_ds)
    self.val_ds = data.get_test_dataset(val_ds)
    print('Dataset - {} loaded!'.format(self.dataset))

    if self.dataset == 'places365_small':
      # 1.8K images for places365_small
      self.num_iter = int(ds_info.splits['train'].num_examples / 1000)
    else:
      self.num_iter = int(ds_info.splits['train'].num_examples)

    # for multi-gpu training
    self.train_ds = self.strategy.experimental_distribute_dataset(self.train_ds)
    self.val_ds = self.strategy.experimental_distribute_dataset(self.val_ds)

    with self.strategy.scope():
      network = net.Network(self.patch_size)
      if self.network_mode == 'coarse':
        self.net = network.coarse_network()
      elif self.network_mode == 'refine':
        self.net = network.refinement_network()
      elif self.network_mode == 'sr':
        self.net = network.sr_network(self.sr_scale)

      # define loss
      self.l1_loss_fn = tf.keras.losses.MeanAbsoluteError(
          reduction=tf.keras.losses.Reduction.NONE)
      self.vgg_model = utils_loss.vgg_model_from_layers(
          layer_names=[self.vgg_layer])
      print('VGG model loaded!')

      # learning rate scheduling
      lr_schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
          boundaries=[lr * self.num_iter for lr in self.lr_boundary],
          values=self.lr_values)

      # optimizer
      self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

      # metrics
      self.train_psnr = tf.keras.metrics.Mean('train_psnr', dtype=tf.float32)
      self.val_loss = tf.keras.metrics.Mean('val_loss', dtype=tf.float32)
      self.val_loss_l1 = tf.keras.metrics.Mean('val_loss_l1', dtype=tf.float32)
      self.val_loss_vgg = tf.keras.metrics.Mean(
          'val_loss_vgg', dtype=tf.float32)
      self.val_psnr = tf.keras.metrics.Mean('val_psnr', dtype=tf.float32)
      self.metrics = [self.train_psnr, self.val_loss, self.val_loss_l1,
                      self.val_loss_vgg, self.val_psnr]

      # set checkpoint
      ckpt = tf.train.Checkpoint(optimizer=self.optimizer, model=self.net)
      self.ckpt_manager = tf.train.CheckpointManager(ckpt, self.ckpt_dir,
                                                     max_to_keep=3)
      # restore checkpoint if possible
      latest_ckpt_path = self.ckpt_manager.latest_checkpoint
      ckpt.restore(latest_ckpt_path)
      if latest_ckpt_path:
        print('Restored from {}'.format(latest_ckpt_path))
        self.start_epoch = int(latest_ckpt_path.split('-')[1])
        print('Starting from {}-th epoch'.format(self.start_epoch))
      else:
        print('Training {} network from scratch!!!'.format(self.network_mode))
        self.start_epoch = 0

  def distributed_l1_loss(self, x, y):
    """L1 loss for distributed training."""
    per_example_loss = self.l1_loss_fn(x, y)
    per_example_loss = tf.math.reduce_mean(per_example_loss, axis=[1, 2])
    return tf.nn.compute_average_loss(per_example_loss,
                                      global_batch_size=self.batch_size)

  def distributed_vgg_loss(self, x, y):
    """VGG loss for distributed training."""
    per_example_loss = utils_loss.vgg_loss(x, y, self.vgg_model, axis=[1, 2, 3])
    return tf.nn.compute_average_loss(per_example_loss,
                                      global_batch_size=self.batch_size)

  def get_local_results(self, pred):
    """Get results after distributed training."""
    pred = self.strategy.experimental_local_results(pred)
    pred = tf.concat(tf.nest.flatten(pred), axis=0)
    return pred

  def train_step(self, inputs):
    """Train step."""
    images, labels = inputs
    with tf.GradientTape() as tape:
      predictions = self.net(images, training=True)
      l1_loss = self.distributed_l1_loss(predictions, labels)
      vgg_loss = self.distributed_vgg_loss(predictions, labels)
      loss = self.lambda_l1 * l1_loss + self.lambda_vgg * vgg_loss
    gradients = tape.gradient(loss, self.net.trainable_variables)
    self.optimizer.apply_gradients(zip(gradients, self.net.trainable_variables))
    # record
    self.train_psnr(utils_misc.measure_psnr(labels, predictions, 1.))
    return self.lambda_l1 * l1_loss, self.lambda_vgg * vgg_loss, loss

  def val_step(self, inputs):
    """Validation step."""
    if (self.network_mode == 'coarse') or (self.network_mode == 'refine'):
      masks, labels = inputs
      images = tf.concat([(1 - masks)*labels[:, :, :, :3], masks], axis=3)
    else:
      images, labels = inputs
    predictions = self.net(images, training=False)
    l1_loss = self.l1_loss_fn(predictions, labels)
    l1_loss = tf.math.reduce_mean(l1_loss, axis=[1, 2])
    vgg_loss = utils_loss.vgg_loss(
        predictions, labels, self.vgg_model, axis=[1, 2, 3])
    self.loss = self.lambda_l1 * l1_loss + self.lambda_vgg * vgg_loss
    # record
    self.val_loss(self.loss)
    self.val_loss_l1(self.lambda_l1 * l1_loss)
    self.val_loss_vgg(self.lambda_vgg * vgg_loss)
    self.val_psnr(utils_misc.measure_psnr(labels, predictions, 1.))
    return predictions

  @tf.function
  def distributed_train_step(self, data_inputs):
    """Train step with distributed training."""
    per_replica_l1, per_replica_vgg, per_replica_total = self.strategy.run(
        self.train_step, args=(data_inputs,))
    reduced_l1_loss = self.strategy.reduce(tf.distribute.ReduceOp.SUM,
                                           per_replica_l1, axis=None)
    reduced_vgg_loss = self.strategy.reduce(tf.distribute.ReduceOp.SUM,
                                            per_replica_vgg, axis=None)
    reduced_total_loss = self.strategy.reduce(tf.distribute.ReduceOp.SUM,
                                              per_replica_total, axis=None)
    return reduced_l1_loss, reduced_vgg_loss, reduced_total_loss

  @tf.function
  def distributed_val_step(self, data_inputs):
    """Validation step with distributed training."""
    distributed_pred = self.strategy.run(self.val_step, args=(data_inputs,))
    return distributed_pred

  def train(self):
    """Train function."""
    writer_train = tf.summary.create_file_writer(os.path.join(self.log_dir,
                                                              'train'))
    writer_val = tf.summary.create_file_writer(os.path.join(self.log_dir,
                                                            'val'))
    # init
    val_psnr_per_epoch = []
    train_it = iter(self.train_ds)
    val_it = iter(self.val_ds)

    # loop over epoch
    for epoch in range(self.start_epoch, self.num_epochs):
      train_loss_l1 = 0.0
      train_loss_vgg = 0.0
      train_loss = 0.0
      num_batches = 0
      for it in range(self.num_iter):
        l1, vgg, total = self.distributed_train_step(next(train_it))
        train_loss_l1 += l1
        train_loss_vgg += vgg
        train_loss += total
        num_batches += 1
        if it % 100 == 0:
          template = ('Epoch [{}/{}] - [{}/{}], Total Loss: {}, L1 Loss: {}, '
                      'VGG Loss: {}, PSNR (dB): {}')
          print(template.format(epoch + 1,
                                self.num_epochs,
                                it + 1,
                                self.num_iter,
                                train_loss / num_batches,
                                train_loss_l1 / num_batches,
                                train_loss_vgg / num_batches,
                                self.train_psnr.result()))
          # summary
          utils_summary.scalar_summary_from_dict(
              writer_train,
              {'loss (total)': train_loss / num_batches,
               'loss (L1)': train_loss_l1 / num_batches,
               'loss (VGG)': train_loss_vgg / num_batches,
               'PSNR': self.train_psnr.result()},
              step=epoch*self.num_iter+it,
              prefix='train ')

      # validation
      for i in range(int(self.num_val / self.batch_size)):
        val_images, val_labels = val_it.get_next()
        if (self.network_mode == 'coarse') or (self.network_mode == 'refine'):
          # get the same mask every time by setting seed value.
          val_mask = tf.expand_dims(
              utils_mask.random_mask(self.patch_size, 'small', seed=i), axis=0)
          # for ease of distributed training, use same mask among iteration
          curr_batch_size = self.batch_size // \
              self.strategy.num_replicas_in_sync
          val_mask = tf.tile(val_mask, [curr_batch_size, 1, 1, 1])
          def value_fn(ctx):
            return val_mask
          distributed_mask = (
              self.strategy.experimental_distribute_values_from_function(
                  value_fn))
          val_pred = self.distributed_val_step((distributed_mask, val_labels))
          val_mask = tf.tile(val_mask,
                             [self.strategy.num_replicas_in_sync, 1, 1, 1])
          val_labels = self.get_local_results(val_labels)
          val_images = (1 - val_mask) * val_labels
        else:
          val_pred = self.distributed_val_step((val_images, val_labels))
          val_images = self.get_local_results(val_images)
          val_labels = self.get_local_results(val_labels)
        val_pred = self.get_local_results(val_pred)
      val_psnr_per_epoch.append(self.val_psnr.result())
      # summary
      utils_summary.scalar_summary_from_dict(
          writer_val,
          {'loss (total)': self.val_loss.result(),
           'loss (L1)': self.val_loss_l1.result(),
           'loss (VGG)': self.val_loss_vgg.result(),
           'PSNR': self.val_psnr.result()},
          step=(epoch+1)*self.num_iter,
          prefix='val ')
      if (self.network_mode == 'coarse') or (self.network_mode == 'refine'):
        utils_summary.image_summary_from_dict(
            writer_val,
            {'Input': val_images[:, :, :, :3],
             'Mask': val_images[:, :, :, 3:],
             'Prediction': val_pred,
             'Label': val_labels},
            step=(epoch+1)*self.num_iter,
            max_num=self.batch_size)
      else:
        utils_summary.image_summary_from_dict(
            writer_val,
            {'Input': val_images[:, :, :, :3],
             'Prediction': val_pred,
             'Label': val_labels},
            step=(epoch+1)*self.num_iter,
            max_num=self.batch_size)
      # print results
      template = ('Epoch [{}/{}], Validation Total Loss: {}, Validation L1 '
                  'Loss: {}, Validation VGG Loss: {}, Validation PSNR (dB): {}')
      print(template.format(epoch + 1,
                            self.num_epochs,
                            self.val_loss.result(),
                            self.val_loss_l1.result(),
                            self.val_loss_vgg.result(),
                            self.val_psnr.result()))
      # logging
      with open(os.path.join(self.log_dir, 'logs.txt'), 'a') as log:
        log.write('{} \t {:.4f} \t {:.4f} \t {:.4f} \t {:.4f} \n'.format(
            epoch+1, self.val_psnr.result(), self.val_loss.result(),
            self.val_loss_l1.result(), self.val_loss_vgg.result()
        ))
      # reset metrics
      for metric in self.metrics:
        metric.reset_states()

      # save checkpoint
      save_path = self.ckpt_manager.save()
      print('Checkpoint saved for epoch {} ({})'.format(epoch + 1, save_path))

    template = '=== Average Validation PSNR (dB): {} ==='
    print(template.format(np.mean(val_psnr_per_epoch)))
