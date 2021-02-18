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

"""Training pipeline for the main training stage.

The TrainingPipeline class defines the training pipeline for the main training
stage, where the coarse network, SR network and the refinement network are all
trained jointly in a GAN framework with distributed training using
MirroredStrategy.

The type of mask (small or large) can be chosen with --mask_type.
Configurations are defined in train_flags() in config.py, and settings used in
the paper are provided in train_small_mask.cfg and train_large_mask.cfg.
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
    self.log_dir = os.path.join(flags.work_dir, flags.mask_type+'_mask', 'logs')
    self.ckpt_dir = os.path.join(
        flags.work_dir, flags.mask_type+'_mask', 'ckpt')
    self.ckpt_dir_keep = os.path.join(
        flags.work_dir, flags.mask_type+'_mask', 'ckpt', 'saved')
    self.mask_type = flags.mask_type
    self.batch_size = flags.batch_size
    self.num_epochs = flags.num_epochs
    self.lr_boundary = flags.lr_boundary
    self.lr_values = flags.lr_values
    self.lr_d_ratio = flags.lr_d_ratio
    self.patch_size = flags.patch_size
    self.lambda_l1 = flags.lambda_l1
    self.lambda_vgg = flags.lambda_vgg
    self.lambda_vgg_ref = flags.lambda_vgg_ref
    self.lambda_gan = flags.lambda_gan
    self.vgg_layer = flags.vgg_layer
    self.dataset = flags.dataset
    self.sr_scale = flags.sr_scale
    self.pretrain_dir_all = flags.pretrain_dir_all
    self.pretrain_dir_coarse = flags.pretrain_dir_coarse
    self.pretrain_dir_refine = flags.pretrain_dir_refine
    self.pretrain_dir_sr = flags.pretrain_dir_sr

    # num_val: number of validation images,
    # 100 validation images (in div2k) made divisible by batch_size
    self.num_val = (100 // self.batch_size) * self.batch_size
    self.buffer_size = 200
    # logging
    if not os.path.exists(os.path.join(self.log_dir, 'logs.txt')):
      utils_misc.check_folder(self.log_dir)
      with open(os.path.join(self.log_dir, 'logs.txt'), 'w') as log:
        log.write('Epoch \t val_PSNR \t total_loss \t loss (coarse) \t loss '
                  '(sr) \t loss (refine) \n')

  def build_model(self):
    """Build model with distributed training using MirroredStrategy."""
    self.strategy = tf.distribute.MirroredStrategy()
    # load dataset - only take num_val number of test images as val
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
    data = utils_data.DatasetFull(self.sr_scale, self.batch_size,
                                  self.patch_size, buffer_size=self.buffer_size,
                                  mask_type=self.mask_type)
    self.train_ds = data.get_train_dataset(train_ds)
    self.val_ds = data.get_test_dataset(val_ds)
    print('Dataset - {} loaded!'.format(self.dataset))
    self.num_iter = int(ds_info.splits['train'].num_examples)

    # for multi-gpu training
    self.train_ds = self.strategy.experimental_distribute_dataset(self.train_ds)
    self.val_ds = self.strategy.experimental_distribute_dataset(self.val_ds)

    with self.strategy.scope():
      network = net.Network(self.patch_size)
      network_big = net.Network([self.patch_size[0]*self.sr_scale,
                                 self.patch_size[1]*self.sr_scale,
                                 self.patch_size[2]])
      self.net = network.coarse_network()
      self.net_sr = network.sr_network(self.sr_scale)
      self.net_refine = network_big.refinement_network()
      self.discriminator = network_big.discriminator()

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
      lr_schedule_d = tf.keras.optimizers.schedules.PiecewiseConstantDecay(
          boundaries=[lr * self.num_iter for lr in self.lr_boundary],
          values=(np.asarray(self.lr_values) * self.lr_d_ratio).tolist())

      # optimizer
      self.optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
      self.optimizer_d = tf.keras.optimizers.Adam(learning_rate=lr_schedule_d)

      # metrics
      self.train_psnr = tf.keras.metrics.Mean('train_psnr', dtype=tf.float32)
      self.val_loss = tf.keras.metrics.Mean('val_loss', dtype=tf.float32)
      self.val_loss_coarse = tf.keras.metrics.Mean('val_loss_coarse',
                                                   dtype=tf.float32)
      self.val_loss_sr = tf.keras.metrics.Mean('val_loss_sr',
                                               dtype=tf.float32)
      self.val_loss_refine = tf.keras.metrics.Mean('val_loss_refine',
                                                   dtype=tf.float32)
      self.val_psnr = tf.keras.metrics.Mean('val_psnr', dtype=tf.float32)
      self.metrics = [self.train_psnr, self.val_loss, self.val_loss_coarse,
                      self.val_loss_sr, self.val_loss_refine, self.val_psnr]

      # set checkpoint
      ckpt = tf.train.Checkpoint(optimizer=self.optimizer,
                                 optimizer_d=self.optimizer_d,
                                 model=self.net,
                                 model_sr=self.net_sr,
                                 model_refine=self.net_refine,
                                 model_d=self.discriminator)
      self.ckpt_manager = tf.train.CheckpointManager(
          ckpt, self.ckpt_dir, max_to_keep=3)
      self.ckpt_manager_save = tf.train.CheckpointManager(
          ckpt, self.ckpt_dir_keep, max_to_keep=None)
      # restore checkpoint if possible
      latest_ckpt_path = self.ckpt_manager.latest_checkpoint
      ckpt.restore(latest_ckpt_path)
      if latest_ckpt_path:
        print('Restored from {}'.format(latest_ckpt_path))
        self.start_epoch = int(latest_ckpt_path.split('-')[1])
        print('Starting from {}-th epoch'.format(self.start_epoch))
      else:
        # restore pre-trained checkpoint for entire framework
        if self.pretrain_dir_all:
          self.ckpt_manager_pretrain = tf.train.CheckpointManager(
              ckpt, self.pretrain_dir_all, max_to_keep=3)
          latest_ckpt_path = self.ckpt_manager_pretrain.latest_checkpoint
          ckpt.restore(latest_ckpt_path)
          print('Pre-trained model restored!!!')
        else:
          # restore pre-trained checkpoint for coarse network
          if self.pretrain_dir_coarse:
            ckpt_pretrain = tf.train.Checkpoint(optimizer=self.optimizer,
                                                model=self.net)
            self.ckpt_manager_pretrain = tf.train.CheckpointManager(
                ckpt_pretrain, self.pretrain_dir_coarse, max_to_keep=3)
            latest_ckpt_path = self.ckpt_manager_pretrain.latest_checkpoint
            ckpt_pretrain.restore(latest_ckpt_path)
            print('Pre-trained model (coarse) restored!!!')
          # restore pre-trained checkpoint for refinement network
          if self.pretrain_dir_refine:
            ckpt_pretrain = tf.train.Checkpoint(optimizer=self.optimizer,
                                                model=self.net_refine)
            self.ckpt_manager_pretrain = tf.train.CheckpointManager(
                ckpt_pretrain, self.pretrain_dir_refine, max_to_keep=3)
            latest_ckpt_path = self.ckpt_manager_pretrain.latest_checkpoint
            ckpt_pretrain.restore(latest_ckpt_path)
            print('Pre-trained model (refine) restored!!!')
          # restore pre-trained checkpoint for sr network
          if self.pretrain_dir_sr:
            ckpt_pretrain = tf.train.Checkpoint(
                optimizer=self.optimizer, model=self.net_sr)
            self.ckpt_manager_pretrain = tf.train.CheckpointManager(
                ckpt_pretrain, self.pretrain_dir_sr, max_to_keep=3)
            latest_ckpt_path = self.ckpt_manager_pretrain.latest_checkpoint
            ckpt_pretrain.restore(latest_ckpt_path)
            print('Pre-trained model (SR) restored!!!')
          print('Training joint framework ({} mask) from scratch!!!'.format(
              self.mask_type))
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

  def distributed_d_hinge_loss(self, real, fake):
    """Hinge loss for D for distributed training."""
    per_example_loss = utils_loss.discriminator_hinge_loss(real, fake, axis=1)
    return tf.nn.compute_average_loss(per_example_loss,
                                      global_batch_size=self.batch_size)

  def distributed_g_hinge_loss(self, fake):
    """Hinge loss for G for distributed training."""
    per_example_loss = utils_loss.generator_hinge_loss(fake, axis=1)
    return tf.nn.compute_average_loss(per_example_loss,
                                      global_batch_size=self.batch_size)

  def distributed_gradient_loss(self, x, y):
    """Gradient loss for distributed training."""
    per_example_loss = utils_loss.image_gradient_loss(x, y, axis=[1, 2, 3])
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
    with tf.GradientTape() as tape, tf.GradientTape() as tape_d:
      input_inpaint = images[:, :, :, :4]  # masked lr image, mask
      labels_inpaint = images[:, :, :, 4:]  # full lr image
      mask = images[:, :, :, 3:4]
      mask_hr = tf.image.resize(
          mask,
          [self.patch_size[0]*self.sr_scale, self.patch_size[1]*self.sr_scale],
          'nearest')
      # outputs
      pred_coarse = self.net(input_inpaint, training=True)
      pred_sr = self.net_sr(pred_coarse, training=True)
      pred_hr = self.net_refine(tf.concat([pred_sr, mask_hr], axis=3),
                                training=True)
      mask_pred_hr = pred_hr*mask_hr + labels*(1.-mask_hr)
      d_real = self.discriminator(tf.concat([labels, mask_hr], axis=3),
                                  training=True)
      d_fake = self.discriminator(tf.concat([mask_pred_hr, mask_hr], axis=3),
                                  training=True)
      # loss
      l1_loss_coarse = self.distributed_l1_loss(
          pred_coarse, labels_inpaint)
      vgg_loss_coarse = self.distributed_vgg_loss(
          pred_coarse, labels_inpaint)
      l1_loss_sr = self.distributed_l1_loss(pred_sr, labels)
      l1_loss_ref = self.distributed_l1_loss(mask_pred_hr, labels)
      grad_loss_ref = self.distributed_gradient_loss(mask_pred_hr, labels)
      vgg_loss_ref = self.distributed_vgg_loss(mask_pred_hr, labels)
      gan_loss_g = self.distributed_g_hinge_loss(d_fake)
      gan_loss_d = self.distributed_d_hinge_loss(d_real, d_fake)
      # total loss for generator
      loss_coarse = self.lambda_l1 * l1_loss_coarse + \
      self.lambda_vgg * vgg_loss_coarse
      loss_sr = self.lambda_l1 * l1_loss_sr
      loss_refine = self.lambda_l1 * (l1_loss_ref + grad_loss_ref) + \
      self.lambda_vgg_ref * vgg_loss_ref + \
      self.lambda_gan * gan_loss_g
      loss = loss_coarse + loss_sr + loss_refine

    # get gradients
    gradients = tape.gradient(loss, self.net.trainable_variables +
                              self.net_sr.trainable_variables +
                              self.net_refine.trainable_variables)
    gradients_d = tape_d.gradient(gan_loss_d,
                                  self.discriminator.trainable_variables)
    # apply gradients
    self.optimizer.apply_gradients(zip(gradients,
                                       self.net.trainable_variables +
                                       self.net_sr.trainable_variables +
                                       self.net_refine.trainable_variables))
    self.optimizer_d.apply_gradients(zip(
        gradients_d, self.discriminator.trainable_variables))
    # record
    self.train_psnr(utils_misc.measure_psnr(labels, mask_pred_hr, 1.))
    return self.lambda_l1 * l1_loss_ref, self.lambda_vgg_ref * vgg_loss_ref, \
        self.lambda_gan * gan_loss_g, loss_coarse, loss_sr, loss_refine, loss

  def val_step(self, inputs):
    """Validation step."""
    mask, labels = inputs
    images = tf.image.resize(labels, [self.patch_size[0], self.patch_size[1]],
                             'bicubic', antialias=True)
    input_inpaint = tf.concat([(1-mask)*images, mask], axis=3)
    labels_inpaint = images
    mask_hr = tf.image.resize(
        mask,
        [self.patch_size[0]*self.sr_scale, self.patch_size[1]*self.sr_scale],
        'nearest')
    # outputs
    pred_coarse = self.net(input_inpaint, training=False)
    pred_sr = self.net_sr(pred_coarse, training=False)
    pred_hr = self.net_refine(tf.concat([pred_sr, mask_hr], axis=3),
                              training=False)
    mask_pred_hr = pred_hr*mask_hr + labels*(1.-mask_hr)
    pred_lr = tf.image.resize(pred_hr, [self.patch_size[0], self.patch_size[1]],
                              'bicubic', antialias=True)
    mask_pred_lr = pred_lr*mask + input_inpaint[:, :, :, :3]
    d_fake = self.discriminator(tf.concat([mask_pred_hr, mask_hr], axis=3),
                                training=False)
    # loss
    l1_loss_coarse = self.l1_loss_fn(pred_coarse, labels_inpaint)
    l1_loss_coarse = tf.math.reduce_mean(l1_loss_coarse, axis=[1, 2])
    vgg_loss_coarse = utils_loss.vgg_loss(
        pred_coarse, labels_inpaint, self.vgg_model, axis=[1, 2, 3])
    l1_loss_sr = self.l1_loss_fn(pred_sr, labels)
    l1_loss_sr = tf.math.reduce_mean(l1_loss_sr, axis=[1, 2])
    l1_loss_ref = self.l1_loss_fn(mask_pred_hr, labels)
    l1_loss_ref = tf.math.reduce_mean(l1_loss_ref, axis=[1, 2])
    grad_loss_ref = utils_loss.image_gradient_loss(
        mask_pred_hr, labels, axis=[1, 2, 3])
    vgg_loss_ref = utils_loss.vgg_loss(
        mask_pred_hr, labels, self.vgg_model, axis=[1, 2, 3])
    gan_loss_g = utils_loss.generator_hinge_loss(d_fake, axis=1)
    # total loss for generator
    loss_coarse = self.lambda_l1 * l1_loss_coarse + \
    self.lambda_vgg * vgg_loss_coarse
    loss_sr = self.lambda_l1 * l1_loss_sr
    loss_refine = self.lambda_l1 * (l1_loss_ref + grad_loss_ref) + \
    self.lambda_vgg_ref * vgg_loss_ref + \
        self.lambda_gan * gan_loss_g
    loss = loss_coarse + loss_sr + loss_refine
    # record
    self.val_loss(loss)
    self.val_loss_coarse(loss_coarse)
    self.val_loss_sr(loss_sr)
    self.val_loss_refine(loss_refine)
    self.val_psnr(utils_misc.measure_psnr(labels, mask_pred_hr, 1.))
    return pred_coarse, pred_sr, mask_pred_hr, mask_pred_lr

  @tf.function
  def distributed_train_step(self, data_inputs):
    """Train step with distributed training."""
    per_replica_l1, per_replica_vgg, per_replica_gan, per_replica_coarse,\
        per_replica_sr, per_replica_refine, per_replica_total = \
        self.strategy.run(self.train_step, args=(data_inputs,))
    reduced_l1_loss = self.strategy.reduce(tf.distribute.ReduceOp.SUM,
                                           per_replica_l1, axis=None)
    reduced_vgg_loss = self.strategy.reduce(tf.distribute.ReduceOp.SUM,
                                            per_replica_vgg, axis=None)
    reduced_gan_loss = self.strategy.reduce(tf.distribute.ReduceOp.SUM,
                                            per_replica_gan, axis=None)
    reduced_coarse_loss = self.strategy.reduce(tf.distribute.ReduceOp.SUM,
                                               per_replica_coarse, axis=None)
    reduced_sr_loss = self.strategy.reduce(tf.distribute.ReduceOp.SUM,
                                           per_replica_sr, axis=None)
    reduced_refine_loss = self.strategy.reduce(tf.distribute.ReduceOp.SUM,
                                               per_replica_refine, axis=None)
    reduced_total_loss = self.strategy.reduce(tf.distribute.ReduceOp.SUM,
                                              per_replica_total, axis=None)
    return reduced_l1_loss, reduced_vgg_loss, reduced_gan_loss, \
        reduced_coarse_loss, reduced_sr_loss, reduced_refine_loss, \
        reduced_total_loss

  @tf.function
  def distributed_val_step(self, data_inputs):
    """Validation step with distributed training."""
    dist_pred_coarse, dist_pred_sr, dist_mask_pred_hr, dist_mask_pred_lr = \
        self.strategy.run(self.val_step, args=(data_inputs,))
    return dist_pred_coarse, dist_pred_sr, dist_mask_pred_hr, dist_mask_pred_lr

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
      train_loss_gan = 0.0
      train_loss_coarse = 0.0
      train_loss_sr = 0.0
      train_loss_refine = 0.0
      train_loss = 0.0
      num_batches = 0
      for it in range(self.num_iter):
        l1, vgg, gan, coarse, sr, refine, total = \
            self.distributed_train_step(next(train_it))
        train_loss_l1 += l1
        train_loss_vgg += vgg
        train_loss_gan += gan
        train_loss_coarse += coarse
        train_loss_sr += sr
        train_loss_refine += refine
        train_loss += total
        num_batches += 1
        if it % 100 == 0:
          template = ('Epoch [{}/{}] - [{}/{}], Total Loss: {}, '
                      'Loss - coarse: {}, Loss - sr: {}, Loss - refine: {}, '
                      'For refine - l1: {} vgg: {} gan: {}, PSNR (dB): {}')
          print(template.format(epoch + 1,
                                self.num_epochs,
                                it + 1,
                                self.num_iter,
                                train_loss / num_batches,
                                train_loss_coarse / num_batches,
                                train_loss_sr / num_batches,
                                train_loss_refine / num_batches,
                                train_loss_l1 / num_batches,
                                train_loss_vgg / num_batches,
                                train_loss_gan / num_batches,
                                self.train_psnr.result()))
          # summary
          utils_summary.scalar_summary_from_dict(
              writer_train,
              {'loss (total)': train_loss / num_batches,
               'loss (coarse)': train_loss_coarse / num_batches,
               'loss (sr)': train_loss_sr / num_batches,
               'loss (refine)': train_loss_refine / num_batches,
               'loss (refine - L1)': train_loss_l1 / num_batches,
               'loss (refine - VGG)': train_loss_vgg / num_batches,
               'loss (refine - GAN)': train_loss_gan / num_batches,
               'PSNR': self.train_psnr.result()},
              step=epoch*self.num_iter+it,
              prefix='train ')

      # validation
      for i in range(int(self.num_val / self.batch_size)):
        _, val_labels = val_it.get_next()
        # get the same mask every time by setting seed value
        val_mask = tf.expand_dims(utils_mask.random_mask(
            self.patch_size, self.mask_type, seed=i), axis=0)
        # for ease of distributed learning, same mask used among iteration
        curr_batch_size = self.batch_size // self.strategy.num_replicas_in_sync
        val_mask = tf.tile(val_mask, [curr_batch_size, 1, 1, 1])
        def value_fn(ctx):
          return val_mask
        distributed_mask = (
            self.strategy.experimental_distribute_values_from_function(
                value_fn))
        # get prediction
        val_pred_coarse, val_pred_sr, val_pred_hr, val_pred_lr = \
            self.distributed_val_step((distributed_mask, val_labels))
        val_pred_coarse = self.get_local_results(val_pred_coarse)
        val_pred_sr = self.get_local_results(val_pred_sr)
        val_pred_hr = self.get_local_results(val_pred_hr)
        val_pred_lr = self.get_local_results(val_pred_lr)
        val_labels = self.get_local_results(val_labels)
        val_mask = tf.tile(
            val_mask, [self.strategy.num_replicas_in_sync, 1, 1, 1])
        val_images = tf.image.resize(val_labels,
                                     [self.patch_size[0], self.patch_size[1]],
                                     'bicubic', antialias=True)
        val_images = tf.concat([(1-val_mask)*val_images, val_mask], axis=3)
      val_psnr_per_epoch.append(self.val_psnr.result())
      # summary
      utils_summary.scalar_summary_from_dict(
          writer_val,
          {'loss (total)': self.val_loss.result(),
           'loss (coarse)': self.val_loss_coarse.result(),
           'loss (sr)': self.val_loss_sr.result(),
           'loss (refine)': self.val_loss_refine.result(),
           'PSNR': self.val_psnr.result()},
          step=(epoch+1)*self.num_iter,
          prefix='val ')
      utils_summary.image_summary_from_dict(
          writer_val,
          {'Input': val_images[:, :, :, :3],
           'Mask': val_images[:, :, :, 3:4],
           'Prediction (coarse)': val_pred_coarse,
           'Prediction (SR)': val_pred_sr,
           'Prediction (HR)': val_pred_hr,
           'Prediction (LR)': val_pred_lr,
           'Label': val_labels},
          step=(epoch+1)*self.num_iter,
          max_num=self.batch_size)
      # print results
      template = ('Epoch [{}/{}], Validation Total Loss: {}, '
                  'Validation Loss - coarse: {}, Validation Loss - sr: {}, '
                  'Validation Loss - refine: {}, '
                  'Validation PSNR (dB): {}')
      print(template.format(epoch + 1,
                            self.num_epochs,
                            self.val_loss.result(),
                            self.val_loss_coarse.result(),
                            self.val_loss_sr.result(),
                            self.val_loss_refine.result(),
                            self.val_psnr.result()))
      # logging
      with open(os.path.join(self.log_dir, 'logs.txt'), 'a') as log:
        log.write(('{} \t {:.4f} \t {:.4f} \t {:.4f} \t {:.4f} \t {:.4f} \n'.
                   format(
                       epoch+1, self.val_psnr.result(), self.val_loss.result(),
                       self.val_loss_coarse.result(), self.val_loss_sr.result(),
                       self.val_loss_refine.result())))
      # reset metrics
      for metric in self.metrics:
        metric.reset_states()

      # save checkpoint
      save_path = self.ckpt_manager.save()
      print('Checkpoint saved for epoch {} ({})'.format(epoch + 1, save_path))
      if (epoch + 1) % 100 == 0:
        save_path = self.ckpt_manager_save.save(checkpoint_number=epoch+1)
        print('Checkpoint saved for epoch {} ({})'.format(epoch + 1, save_path))

    template = '=== Average Validation PSNR (dB): {} ==='
    print(template.format(np.mean(val_psnr_per_epoch)))
