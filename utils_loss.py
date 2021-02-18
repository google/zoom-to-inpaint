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

"""Utility functions for loss functions."""

import tensorflow as tf


def vgg_model_from_layers(layer_names):
  """Creates a vgg model that returns a list of intermediate outputs."""
  vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
  vgg.trainable = False
  outputs = [vgg.get_layer(name).output for name in layer_names]
  model = tf.keras.Model([vgg.input], outputs)
  return model


def vgg_loss(x, y, vgg_model, axis=None):
  """Perceptual loss using VGG19.

  Args:
    x: An input Tensor to compute VGG loss.
    y: An input Tensor to compute VGG loss.
    vgg_model: A tf.keras.Model.
    axis: None to reduce all. Omit batch axis for distributed training.

  Returns:
    A Tensor.
  """
  def preprocess(z):
    """Pre-process input to match conditions VGG19 was trained with."""
    z = tf.keras.applications.vgg19.preprocess_input(z * 255)
    z = tf.image.resize(z, (224, 224), antialias=True)
    return z
  x_vgg = vgg_model(preprocess(x))
  y_vgg = vgg_model(preprocess(y))

  return tf.math.reduce_mean(tf.math.abs(x_vgg - y_vgg), axis=axis)


def discriminator_hinge_loss(real, fake, axis=None):
  """Hinge GAN loss for discriminator."""
  real_loss = tf.reduce_mean(tf.nn.relu(1.0 - real), axis=axis)
  fake_loss = tf.reduce_mean(tf.nn.relu(1.0 + fake), axis=axis)
  loss = real_loss + fake_loss
  return loss


def generator_hinge_loss(fake, axis=None):
  """Hinge GAN loss for generator."""
  return -tf.reduce_mean(fake, axis=axis)


def image_gradient_loss(x, y, axis=None):
  """Image gradient loss."""
  diff = x - y
  dx, dy = tf.image.image_gradients(diff)
  loss = tf.reduce_mean(dx * dx + dy * dy, axis=axis)
  return loss
