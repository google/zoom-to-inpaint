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

"""Miscellaneous utility functions.

File I/O and image quality metrics.
"""

import os

import numpy as np
from PIL import Image
import tensorflow as tf


@tf.function
def measure_psnr(x, y, peak):
  """Peak signal to noise ratio."""
  return tf.image.psnr(x, y, peak)


@tf.function
def measure_ssim(x, y, peak):
  """Measure tructural similarity."""
  return tf.image.ssim(x, y, peak)


@tf.function
def measure_msssim(x, y, peak,
                   power_factors=(0.0448, 0.2856, 0.3001, 0.2363, 0.1333),
                   filter_size=11):
  """Measure mlti-scale structural similarity."""
  return tf.image.ssim_multiscale(x, y, peak,
                                  power_factors=power_factors,
                                  filter_size=filter_size)


@tf.function
def measure_l1_loss(x, y):
  """Measure L1 loss."""
  return tf.math.reduce_mean(tf.math.abs(x - y))


@tf.function
def resize_to_fixed_size(img, size):
  """Resize 4D image Tensor to given size (H, W)."""
  _, h, w, _ = img.shape
  if (h != size[0]) or (w != size[1]):
    img = tf.image.resize(img, size, 'bicubic', antialias=True)
  return img


def check_folder(directory):
  """Create directory if it does not already exist."""
  if not os.path.exists(directory):
    os.makedirs(directory)
  return directory


def read_img_and_normalize(path):
  """Read image from file path."""
  img = np.array(Image.open(path), 'float32')
  img = np.expand_dims(img, axis=0)
  img = np.clip(img / 255., 0, 1)
  return img


def denormalize_and_save_img(img, path):
  """Save image to file path."""
  img = np.clip(img, 0, 1) * 255
  img = np.squeeze(img)
  img = Image.fromarray(img.astype(np.uint8))
  img.save(path)
