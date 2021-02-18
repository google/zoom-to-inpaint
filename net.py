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

"""Network architectures.

Defines the CNN architecture of coarse network, refinement network, SR network
and the discirminator in the Zoom-to-Inpaint framework.
"""

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.initializers import GlorotNormal

import utils_net


class Network:
  """Network class with network architectures of individual components."""

  def __init__(self, patch_size):
    self.patch_size = patch_size

  def coarse_network(self, ch=64):
    """Coarse network.

    An auto-encoder with ResBlocks +
    GatedConv (https://arxiv.org/abs/1806.03589) + SyncBatchNorm.

    Args:
      ch: Number of output channels.

    Returns:
      model: A tf.Keras.Model.
    """
    inputs = layers.Input((self.patch_size[0], self.patch_size[1],
                           self.patch_size[2]+1))
    img = inputs[:, :, :, :3]
    mask = inputs[:, :, :, 3:]
    # Encoder level 1
    x = utils_net.gated_conv(inputs, ch, 3)
    x = layers.experimental.SyncBatchNormalization()(x)
    res = x
    x = utils_net.gated_conv(x, ch, 3)
    x = layers.experimental.SyncBatchNormalization()(x)
    x = utils_net.gated_conv(x, ch, 3)
    x = layers.experimental.SyncBatchNormalization()(x)
    x = res + x
    # Encoder level 2
    x = layers.MaxPool2D(pool_size=(2, 2))(x)
    x = utils_net.gated_conv(x, ch*2, 3)
    x = layers.experimental.SyncBatchNormalization()(x)
    res = x
    x = utils_net.gated_conv(x, ch*2, 3)
    x = layers.experimental.SyncBatchNormalization()(x)
    x = utils_net.gated_conv(x, ch*2, 3)
    x = layers.experimental.SyncBatchNormalization()(x)
    x = res + x
    # Bottleneck
    x = layers.MaxPool2D(pool_size=(2, 2))(x)
    x = utils_net.gated_conv(x, ch*4, 3, rate=2)
    x = layers.experimental.SyncBatchNormalization()(x)
    res = x
    x = utils_net.gated_conv(x, ch*4, 3, rate=4)
    x = layers.experimental.SyncBatchNormalization()(x)
    x = utils_net.gated_conv(x, ch*4, 3, rate=8)
    x = layers.experimental.SyncBatchNormalization()(x)
    x = res + x
    # Decoder level 2
    x = layers.UpSampling2D()(x)
    x = utils_net.gated_conv(x, ch*2, 3)
    x = layers.experimental.SyncBatchNormalization()(x)
    res = x
    x = utils_net.gated_conv(x, ch*2, 3)
    x = layers.experimental.SyncBatchNormalization()(x)
    x = utils_net.gated_conv(x, ch*2, 3)
    x = layers.experimental.SyncBatchNormalization()(x)
    x = utils_net.gated_conv(x, ch*2, 3)
    x = layers.experimental.SyncBatchNormalization()(x)
    x = res + x
    # Decoder level 1
    x = layers.UpSampling2D()(x)
    x = utils_net.gated_conv(x, ch, 3)
    x = layers.experimental.SyncBatchNormalization()(x)
    res = x
    x = utils_net.gated_conv(x, ch, 3)
    x = layers.experimental.SyncBatchNormalization()(x)
    x = utils_net.gated_conv(x, ch, 3)
    x = layers.experimental.SyncBatchNormalization()(x)
    x = utils_net.gated_conv(x, ch, 3)
    x = layers.experimental.SyncBatchNormalization()(x)
    x = res + x
    # output
    x = utils_net.gated_conv(x, 3, 3)
    x = x*mask + img*(1.-mask)
    # model
    model = tf.keras.Model(inputs, x, name='coarse_network')
    model.summary()

    return model

  def refinement_network(self, ch=64, rate=2, attention_rate=2):
    """Refinement network.

    An auto-encoder with ResBlocks +
    GatedConv (https://arxiv.org/abs/1806.03589) + SyncBatchNorm
    + Contextual Attention (https://arxiv.org/abs/1801.07892).

    Args:
      ch: Number of output channels.
      rate: Down/Up-scaling rate (scaling factor) at each encoder/decoder level.
      attention_rate: Down-scaling rate for applying contextual attention.

    Returns:
      model: A tf.Keras.Model.
    """
    inputs = layers.Input((self.patch_size[0], self.patch_size[1],
                           self.patch_size[2]+1))
    img = inputs[:, :, :, :3]
    mask = inputs[:, :, :, 3:]

    # Encoder level 1
    x = utils_net.gated_conv(inputs, ch, 3)
    x = layers.experimental.SyncBatchNormalization()(x)
    res = x
    x = utils_net.gated_conv(x, ch, 3)
    x = layers.experimental.SyncBatchNormalization()(x)
    x = utils_net.gated_conv(x, ch, 3)
    x = layers.experimental.SyncBatchNormalization()(x)
    x = res + x
    # Encoder level 2
    x = layers.MaxPool2D(pool_size=(rate, rate))(x)
    x = utils_net.gated_conv(x, ch*2, 3)
    x = layers.experimental.SyncBatchNormalization()(x)
    res = x
    x = utils_net.gated_conv(x, ch*2, 3)
    x = layers.experimental.SyncBatchNormalization()(x)
    x = utils_net.gated_conv(x, ch*2, 3)
    x = layers.experimental.SyncBatchNormalization()(x)
    x = res + x
    # Bottleneck
    x = layers.MaxPool2D(pool_size=(rate, rate))(x)
    x = utils_net.gated_conv(x, ch*4, 3, rate=2)
    x = layers.experimental.SyncBatchNormalization()(x)
    res = x
    x = utils_net.gated_conv(x, ch*4, 3, rate=4)
    x = layers.experimental.SyncBatchNormalization()(x)
    x = utils_net.gated_conv(x, ch*4, 3, rate=8)
    x = layers.experimental.SyncBatchNormalization()(x)
    x = res + x
    # contextual attention
    x_ca = utils_net.contextual_attention_module(
        x, mask, attention_rate=attention_rate)
    x = tf.concat([x, x_ca], axis=3)
    x = utils_net.gated_conv(x, ch*4, 3)
    x = layers.experimental.SyncBatchNormalization()(x)

    # Decoder level 2
    x = layers.UpSampling2D(size=(rate, rate))(x)
    x = utils_net.gated_conv(x, ch*2, 3)
    x = layers.experimental.SyncBatchNormalization()(x)
    res = x
    x = utils_net.gated_conv(x, ch*2, 3)
    x = layers.experimental.SyncBatchNormalization()(x)
    x = utils_net.gated_conv(x, ch*2, 3)
    x = layers.experimental.SyncBatchNormalization()(x)
    x = utils_net.gated_conv(x, ch*2, 3)
    x = layers.experimental.SyncBatchNormalization()(x)
    x = res + x
    # Decoder level 1
    x = layers.UpSampling2D(size=(rate, rate))(x)
    x = utils_net.gated_conv(x, ch, 3)
    x = layers.experimental.SyncBatchNormalization()(x)
    res = x
    x = utils_net.gated_conv(x, ch, 3)
    x = layers.experimental.SyncBatchNormalization()(x)
    x = utils_net.gated_conv(x, ch, 3)
    x = layers.experimental.SyncBatchNormalization()(x)
    x = utils_net.gated_conv(x, ch, 3)
    x = layers.experimental.SyncBatchNormalization()(x)
    x = res + x
    # output
    x = utils_net.gated_conv(x, 3, 3)
    x = x + img
    # model
    model = tf.keras.Model(inputs, x,
                           name='refinement_network')
    model.summary()

    return model

  def sr_network(self, sr_scale, ch=64):
    """Super-resolution network.

    An SR network with cascaded residual blocks.

    Args:
      sr_scale: Scaling factor for SR.
      ch: Number of output channels.

    Returns:
      model: A tf.Keras.Model.
    """
    def conv(x, channels, k_sz, stride=1, actvn='relu'):
      """Conv2D layer."""
      return  layers.Conv2D(channels, k_sz, strides=stride, activation=actvn,
                            padding='same', kernel_initializer=GlorotNormal)(x)

    def resblock(x, ch, k_sz):
      """Residual block."""
      skip = x
      x = conv(x, ch, k_sz)
      x = conv(x, ch, k_sz)
      return x+skip

    # input
    inputs = layers.Input(self.patch_size)
    x = conv(inputs, ch, 3)
    # Residual blocks
    x = resblock(x, ch, 3)
    x = resblock(x, ch, 3)
    x = resblock(x, ch, 3)
    x = resblock(x, ch, 3)
    # Reconstruction
    x = conv(x, ch*sr_scale*sr_scale, 3)
    x = tf.nn.depth_to_space(x, sr_scale)
    x = conv(x, 3, 3, actvn=None)
    bic = tf.image.resize(
        inputs,
        [self.patch_size[0]*sr_scale, self.patch_size[1]*sr_scale],
        'bicubic')
    x = bic + x
    # model
    model = tf.keras.Model(inputs, x, name='SR_network')
    model.summary()

    return model

  def discriminator(self, ch=64):
    """Discriminator.

    A Patch GAN with spectral normalization.
    Patch GAN: https://arxiv.org/abs/1611.07004
    Spectral normalization: https://arxiv.org/abs/1802.05957

    Args:
      ch: Number of output channels.

    Returns:
      model: A tf.Keras.Model.
    """
    def conv_spectral_norm(x, channels, k_sz, stride=2,
                           activation=layers.LeakyReLU(alpha=0.2)):
      """Conv2D layer with spectral normalization."""
      x = utils_net.SpectralNorm(
          layers.Conv2D(channels, k_sz, strides=stride, activation=None,
                        padding='same'))(x)
      return activation(x)

    inputs = layers.Input((self.patch_size[0], self.patch_size[1],
                           self.patch_size[2]+1))
    x = conv_spectral_norm(inputs, ch, 5)
    x = conv_spectral_norm(x, ch*2, 5)
    x = conv_spectral_norm(x, ch*4, 5)
    x = conv_spectral_norm(x, ch*4, 5)
    x = conv_spectral_norm(x, ch*4, 5)
    x = layers.Flatten()(x)
    # model
    model = tf.keras.Model(inputs, x, name='discriminator')
    model.summary()

    return model
