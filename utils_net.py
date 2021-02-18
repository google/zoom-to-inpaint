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

"""Utility functions for network architecture."""

import math

import tensorflow as tf
from tensorflow.keras import layers


def gated_conv(x, channels, k_sz, stride=1, pad='same', rate=1,
               activation=tf.keras.activations.elu):
  """Gated convolution.

  Paper: https://arxiv.org/abs/1806.03589

  Performs 2D convolution with 'channels' number of output channels,
  then performs element-wise gating after splitting by 2.

  Args:
    x: A Tensor with shape [N, H, W, channels]
    channels: An int. Number of output channels.
    k_sz: An int. Convolution kernel size.
    stride: An int. Convolution stride.
    pad: A string. Convolution padding.
    rate: An int. Dilation rate.
    activation: tf.keras.activations.

  Returns:
    out: A Tensor with shape [N, H, W, channels/2]
  """
  x = layers.Conv2D(channels, k_sz, strides=stride, padding=pad,
                    dilation_rate=rate)(x)
  if channels == 3:
    return x
  x, y = tf.split(x, 2, 3)
  if activation is not None:
    x = activation(x)
  y = tf.keras.activations.sigmoid(y)
  out = x * y
  return out


class SpectralNorm(tf.keras.layers.Wrapper):
  """Spectral Normalization for stable training of GANs.

  Paper: https://arxiv.org/abs/1802.05957
  """

  def __init__(self, layer, steps=1, **kwargs):
    super().__init__(layer)
    self.steps = steps

  def build(self, input_shape):
    if not self.layer.built:
      self.layer.build(input_shape)
      self.kernel = self.layer.kernel
      self.kernel_sz = self.kernel.shape.as_list()

      self.u = self.add_weight(shape=(1, self.kernel_sz[-1]),
                               initializer=tf.initializers.TruncatedNormal(
                                   stddev=0.02),
                               trainable=False,
                               name='spectral_norm_u',
                               dtype=tf.float32)
      self.v = self.add_weight(
          shape=(1, self.kernel_sz[0] * self.kernel_sz[1] * self.kernel_sz[2]),
          initializer=tf.initializers.TruncatedNormal(stddev=0.02),
          trainable=False,
          name='spectral_norm_v',
          dtype=tf.float32)
    super().build()

  def call(self, inputs):
    kernel = tf.reshape(self.kernel, [-1, self.kernel_sz[-1]])
    u_init = self.u
    v_init = None
    for _ in range(self.steps):
      v_init = tf.math.l2_normalize(tf.matmul(u_init, tf.transpose(kernel)))
      u_init = tf.math.l2_normalize(tf.matmul(v_init, kernel))
    self.u.assign(u_init)
    self.v.assign(v_init)

    sigma = tf.matmul(tf.matmul(v_init, kernel), tf.transpose(u_init))
    kernel_norm = self.kernel / sigma
    self.layer.kernel.assign(kernel_norm)

    output = self.layer(inputs)
    self.layer.kernel.assign(self.kernel)
    return output


###########################################################################
# Contextual attention
###########################################################################


def extract_image_patches(net, patch_size, strides):
  """Extract image patches.

  Args:
    net: A Tensor. (B x H x W x C)
    patch_size: An int. The size of the patch.
    strides: A list of ints. How far the centers of two consecutive patches are
      in the input Tensor. (1 x stride_rows x stride_cols x 1)

  Returns:
    Output extract patches. (B x H' x W' x P*P*C)
      P : patch_size
      H': H // stride_rows
      W': W // stride_cols
  """
  channels = net.get_shape().as_list()[-1]
  kernel_size = patch_size * patch_size * channels
  kernel = tf.reshape(
      tf.eye(kernel_size, dtype=net.dtype),
      [patch_size, patch_size, channels, -1])
  patches = tf.nn.conv2d(net, kernel, strides=strides, padding='SAME')
  return patches


def get_attention_map(net, mask, propagation_size=3,
                      softmax_scale=10, patch_size=3):
  """Computes cosine similarity between patches inside and outside of the mask.

  This is based on DeepFill: https://arxiv.org/abs/1801.07892
  The original implementation uses convolution op, however this method
  and apply_attention_map use only tf.matmul to remove the for-loop
  iterations over mini-batch samples.

  The computed attention map is used later by apply_attention_map method to
  fill the mask area.

  example:
    attention_map = utils.get_attention_map(deeper_net, mask)
        ...
    applied = utils.apply_attention_map(net, attention_map)

  Args:
    net: Input tensor.  (B x H x W x C)
    mask: Input mask.  (B x H x W x 1)
    propagation_size: The size of propagation to encourage coherency of
      attention. The convolution of the computed attention map with
      identity matrix as kernel is used as described in the DeepFill paper.
    softmax_scale: When this is larger than 0, multiplies the computed attention
      map by this value and applies softmax spatially.
    patch_size: An int. The size of the patch.

  Returns:
    Computed attention map.  (B x H*W x H*W)
    The attention map is used in apply_attention_map method.

  Raises:
    ValueError : When `net` and `mask` are not in the same resolution.
  """
  b, h, w, c = tf.shape(net)

  if mask.shape.as_list()[1] != net.shape.as_list()[1] or \
      mask.shape.as_list()[2] != net.shape.as_list()[2]:
    raise ValueError('The resolutions of net and mask must be the same.')

  patches = extract_image_patches(net, patch_size, [1, 1, 1, 1])

  # Normalizing patches.  (B x H x W x P*P*C)
  patches_normalized = tf.math.divide(
      patches, tf.maximum(tf.norm(patches, axis=-1, keepdims=True), 1e-9))

  # Transpose inverted mask.  (B x P*P*C x H*W)
  patches_transposed = tf.transpose(
      tf.reshape(patches, [b, h * w, patch_size * patch_size * c]),
      [0, 2, 1])

  # (B x H*W x P*P*C)
  patches_normalized_reshaped = tf.reshape(
      patches_normalized, [b, h * w, patch_size * patch_size * c])

  # (B x H*W x H*W)
  attention_map = tf.matmul(
      patches_normalized_reshaped, patches_transposed)

  if propagation_size > 0:
    # Attention propagation.
    prop_weight = tf.reshape(
        tf.eye(propagation_size),
        [propagation_size, propagation_size, 1, 1])
    proped_horizontally = tf.nn.conv2d(
        tf.expand_dims(attention_map, axis=-1), prop_weight,
        strides=[1, 1, 1, 1],
        padding='SAME')

    transposed = tf.reshape(
        tf.transpose(tf.reshape(
            proped_horizontally,
            [b, h, w, h, w]), [0, 2, 1, 4, 3]),
        [b, w * h, w * h, 1])
    proped_vertically = tf.nn.conv2d(
        transposed, prop_weight,
        strides=[1, 1, 1, 1],
        padding='SAME')

    attention_map = tf.reshape(
        tf.transpose(tf.reshape(
            proped_vertically,
            [b, w, h, w, h]), [0, 2, 1, 4, 3]),
        [b, h * w, h * w])

  # In order to make the similarity map stable on the edges of the mask, use
  # 3x3 max_pool2d to clear values.
  mask_filter = tf.reshape(
      tf.nn.max_pool2d(
          mask, [3, 3], [1, 1], padding='SAME'),
      [tf.shape(attention_map)[0], 1, -1])
  if softmax_scale > 0.0:
    attention_map_scaled = attention_map * softmax_scale
    attention_map_normalized = attention_map_scaled - tf.reduce_max(
        attention_map_scaled, axis=-1, keepdims=True)
    attention_map_exp = tf.exp(attention_map_normalized) * (
        1.0 + tf.math.multiply(mask_filter, -1.0))
    attention_map = attention_map_exp / tf.maximum(tf.reduce_sum(
        attention_map_exp, axis=-1, keepdims=True), 1e-9)
  else:
    attention_map *= mask_filter
  return attention_map


def apply_attention_map(net, attention, mask):
  """Copies patches of outside mask to inside based on the attention map.

  The input attention map is the cosine similarity between inside and outside of
  the mask. This method fills the mask area by computing a spacial dot
  product of the input tensor and the attention map per pixel.

  When the resolution of the input tensor is higher than the attention map,
  patches of the input tensor are used for the computation.

  Args:
    net: Input tensor.  (B x H x W x C)
    attention: Input attention map computed by get_attention_map.
      (B x H*W x H*W)
    mask: A grayscale image, where a positive value indicates a region in image
      where attention map should be applied.  (B x H x W x 1)

  Returns:
    Result applied with the input attention map in the same shape as the input.
    (B x H x W x C)

  Raises:
    ValueError: When the shapes of the input net and attention map are not
      correctly defined. Please see the code for the details.
  """
  _, h, w, _ = net.shape.as_list()
  attention_size = attention.shape.as_list()[1]
  square_block_size = h * w // attention_size
  if square_block_size * attention_size != h * w:
    raise ValueError(
        'Invalid shape. The multiplication of the input height(%d) and '
        'width(%d) must be a multiple of the second dimension size(%d) of '
        'the attention map.' % (
            h, w, attention_size))
  block_size = int(math.sqrt(square_block_size))
  if block_size * block_size != square_block_size:
    raise ValueError(
        'Invalid shape. The multiplication of the input height and width '
        'divided by the number of the second dimension of the attention map'
        '(%d) must be a square of an integer.' % square_block_size)

  b, h, _, c = tf.shape(net)
  if block_size > 1:
    depth = tf.nn.space_to_depth(net, block_size)
  else:
    depth = net
  h, w = depth.shape.as_list()[1:3]
  right = tf.reshape(depth, [b, -1, c * block_size * block_size])
  mult = tf.reshape(tf.matmul(attention, right), [b, h, w, -1])
  if block_size > 1:
    mult = tf.nn.depth_to_space(mult, block_size)
  # Filter out the applied attention with mask because the attention map
  # is generated in lower resolution and the result may include pixels
  # outside of the mask area.
  return net * (1.0 - mask) + mult * mask


def contextual_attention_module(x, mask, attention_rate=2,
                                propagation_size=3, softmax_scale=10):
  """Contextual attention module with gated convolution.

  Computes the attention map on features down-scaled by attention_rate with
  function get_attention_map(), and applies it using apply_attention_map().
  Has a gated convolution + BatchNorm at the front and back.

  Args:
    x: Input Tensor. (B x H x W x C)
    mask: A binary mask. (B x H x W x 1)
    attention_rate: Down-scaling rate for computing the attention map to reduce
      computation/memory.
    propagation_size: In get_attention_map()
    softmax_scale: In get_attention_map()

  Returns:
    Result after applying contextual attention. (B x H x W x C)
  """
  _, h, w, c = x.shape
  mask_lr = tf.image.resize(mask, [h, w], 'nearest')
  mask_lr_small = tf.image.resize(
      mask_lr, [h//attention_rate, w//attention_rate], 'nearest')

  x_ca = gated_conv(x, c*2, 3)
  x_ca = layers.experimental.SyncBatchNormalization()(x_ca)
  x_ca_small = tf.image.resize(
      x_ca, [h//attention_rate, w//attention_rate], 'nearest')

  x_att = get_attention_map(x_ca_small, mask_lr_small,
                            propagation_size=propagation_size,
                            softmax_scale=softmax_scale)
  x_att.set_shape(
      [None, (h**2)//(attention_rate**2), (w**2)//(attention_rate**2)])
  x_ca = apply_attention_map(x_ca, x_att, mask_lr)

  x_ca = gated_conv(x_ca, c*2, 3)
  x_ca = layers.experimental.SyncBatchNormalization()(x_ca)
  return x_ca
