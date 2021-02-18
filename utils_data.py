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

"""Utility functions for data processing."""

import numpy as np
import tensorflow as tf
import utils_mask


def random_crop(img, label, patch_size):
  """Random crop image."""
  img = tf.image.random_crop(img, patch_size)
  return img, label


def center_crop(img, label, patch_size):
  """Center crop image."""
  img_shape = tf.shape(img)[:2]
  offset_h = (img_shape[0] - patch_size[0]) // 2
  offset_w = (img_shape[1] - patch_size[1]) // 2
  img = tf.image.crop_to_bounding_box(img, offset_h, offset_w, patch_size[0],
                                      patch_size[1])
  return img, label


def random_crop_sr(lr, hr, hr_patch_size, sr_scale):
  """Random crop LR, HR patch."""
  lr_patch_size = np.array(hr_patch_size) // sr_scale
  lr_shape = tf.shape(lr)[:2]

  lr_h = tf.random.uniform(shape=(), maxval=lr_shape[0] - lr_patch_size[0] + 1,
                           dtype=tf.int32)
  lr_w = tf.random.uniform(shape=(), maxval=lr_shape[1] - lr_patch_size[1] + 1,
                           dtype=tf.int32)

  hr_w = lr_w * sr_scale
  hr_h = lr_h * sr_scale

  lr_cropped = lr[lr_h:lr_h + lr_patch_size[0], lr_w:lr_w + lr_patch_size[1]]
  hr_cropped = hr[hr_h:hr_h + hr_patch_size[0], hr_w:hr_w + hr_patch_size[1]]

  return lr_cropped, hr_cropped


def center_crop_sr(lr, hr, hr_patch_size, sr_scale):
  """Center crop LR, HR patch."""
  lr_patch_size = np.array(hr_patch_size) // sr_scale
  lr_shape = tf.shape(lr)[:2]

  lr_h = (lr_shape[0] - lr_patch_size[0]) // 2
  lr_w = (lr_shape[1] - lr_patch_size[1]) // 2

  hr_w = lr_w * sr_scale
  hr_h = lr_h * sr_scale

  lr_cropped = lr[lr_h:lr_h + lr_patch_size[0], lr_w:lr_w + lr_patch_size[1]]
  hr_cropped = hr[hr_h:hr_h + hr_patch_size[0], hr_w:hr_w + hr_patch_size[1]]

  return lr_cropped, hr_cropped


class DatasetPre():
  """Dataset class used during pre-training."""

  def __init__(self, batch_size, patch_size, buffer_size, mask_type='small'):
    self.batch_size = batch_size
    self.patch_size = patch_size
    self.buffer_size = buffer_size
    self.mask_type = mask_type

  def prepare_data(self, img, label):
    """Prepare data for training.

    Normalize images 'uint8' -> 'float32' and generate input-label pair.

    Args:
      img: A Tensor. (H x W x 3)
      label: Unused.

    Returns:
      x: A Tensor of size (H x W x 4). Consists of masked image (H x W x 3) and
      mask (H x W x 1) stacked along the channel dimension. Used as input to
      an inpainting network.
      img: A Tensor of size (H x W x 3). Full image to be used as ground truth.
    """
    img = tf.cast(img, tf.float32) / 255.
    # generate input
    mask = utils_mask.random_mask(self.patch_size, self.mask_type)
    masked_img = img * (1. - mask)
    x = tf.concat([masked_img, mask], axis=2)
    return x, img

  def prepare_data_test(self, img, label):
    """Prepare data for testing."""
    img = tf.cast(img, tf.float32) / 255.
    return img, img

  def get_train_dataset(self, train_ds):
    """Train dataset processing."""
    train_ds = train_ds.map(
        lambda img, label: random_crop(img, label, self.patch_size),
        num_parallel_calls=tf.data.experimental.AUTOTUNE)
    train_ds = train_ds.map(self.prepare_data)
    train_ds = train_ds.shuffle(self.buffer_size).repeat().batch(
        self.batch_size)
    train_ds = train_ds.prefetch(tf.data.experimental.AUTOTUNE)
    return train_ds

  def get_test_dataset(self, test_ds):
    """Test dataset processing."""
    test_ds = test_ds.map(
        lambda img, label: center_crop(img, label, self.patch_size),
        num_parallel_calls=tf.data.experimental.AUTOTUNE)
    test_ds = test_ds.map(self.prepare_data_test)
    test_ds = test_ds.repeat().batch(self.batch_size)
    test_ds = test_ds.prefetch(tf.data.experimental.AUTOTUNE)
    return test_ds


class DatasetFull():
  """Dataset class for the full inpainting framework including SR."""

  def __init__(self, sr_scale, batch_size, patch_size, buffer_size,
               mask_type='small'):
    self.sr_scale = sr_scale
    self.batch_size = batch_size
    self.patch_size = patch_size
    self.hr_patch_size = [sr_scale * patch_size[0], sr_scale * patch_size[1],
                          patch_size[2]]
    self.buffer_size = buffer_size
    self.mask_type = mask_type  # set to None for SR without inpainting

  def prepare_data(self, img, label):
    """Prepare data for training.

    Normalize images 'uint8' -> 'float32', resize by bicubic down-scaling
    and generate input-label pair.

    Args:
      img: An image. (unused)
      label: A Tensor of size (sH x sW x 3). High resolution label to be used
      as ground truth.

    Returns:
      img: Returns a Tensor of size (H x W x 3) if mask_type=None.
      Else, returns a Tensor of size (H x W x 7) that consists of a masked
      image (H x W x 3), a mask (H x W x 1) and a low resolution image
      (H x W x 3) stacked in the channel dimension.
      The LR image is used as the LR ground truth for inpainting.
      label: A Tensor of size (sH x sW x 3). High resolution label to be used
      as ground truth.
    """
    label = tf.cast(label, tf.float32) / 255.
    img = tf.image.resize(label, [self.patch_size[0], self.patch_size[1]],
                          'bicubic', antialias=True)
    if self.mask_type is not None:
      mask = utils_mask.random_mask(self.patch_size, self.mask_type)
      masked_img = img * (1. - mask)
      img = tf.concat([masked_img, mask, img], axis=2)
    return img, label

  def prepare_data_test(self, img, label):
    """Prepare data for testing."""
    label = tf.cast(label, tf.float32) / 255.
    img = tf.image.resize(label, [self.patch_size[0], self.patch_size[1]],
                          'bicubic', antialias=True)
    return img, label

  def get_train_dataset(self, train_ds):
    """Train dataset processing."""
    train_ds = train_ds.map(
        lambda lr, hr: random_crop_sr(
            lr, hr, self.hr_patch_size, self.sr_scale),
        num_parallel_calls=tf.data.experimental.AUTOTUNE)
    train_ds = train_ds.map(self.prepare_data)
    train_ds = train_ds.shuffle(self.buffer_size).repeat().batch(
        self.batch_size)
    train_ds = train_ds.prefetch(tf.data.experimental.AUTOTUNE)
    return train_ds

  def get_test_dataset(self, test_ds):
    """Test dataset processing."""
    test_ds = test_ds.map(
        lambda lr, hr: center_crop_sr(
            lr, hr, self.hr_patch_size, self.sr_scale),
        num_parallel_calls=tf.data.experimental.AUTOTUNE)
    test_ds = test_ds.map(self.prepare_data_test)
    test_ds = test_ds.repeat().batch(self.batch_size)
    test_ds = test_ds.prefetch(tf.data.experimental.AUTOTUNE)
    return test_ds
