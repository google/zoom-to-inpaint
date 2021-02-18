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

"""Configurations for training and testing.

Defines flags for training and testing.
"""

from absl import flags


def pretrain_flags():
  """Flags for pretraining."""
  # Training parameters
  flags.DEFINE_enum('network_mode', 'coarse', ['coarse', 'refine', 'sr'],
                    'Network mode: coarse for pretraining the coarse network, '
                    'refine for pretraining the refinement network, and sr for '
                    'pretraining the SR network.')
  flags.DEFINE_string('work_dir', './pretrain',
                      'Desired working directory to save checkpoints')
  flags.DEFINE_integer('batch_size', 16, 'Mini-batch size')
  flags.DEFINE_integer('num_epochs', 100, 'Total number of epochs')
  flags.DEFINE_multi_integer('lr_boundary', [80, 90],
                             'Learning rate boundary for stair decay')
  flags.DEFINE_multi_float('lr_values', [0.001, 0.0001, 0.00001],
                           'Learning rate for each interval')
  flags.DEFINE_multi_integer('patch_size', [256, 256, 3], 'Training patch size')

  # Loss coefficients
  flags.DEFINE_float('lambda_l1', 1, 'Reconstruction (L1) loss coefficient')
  flags.DEFINE_float('lambda_vgg', 0.01, 'VGG loss coefficient')
  flags.DEFINE_enum('vgg_layer', 'block4_conv1',
                    ['block1_conv1', 'block2_conv1', 'block3_conv1',
                     'block4_conv1'],
                    'VGG19 layer to use for perceptual loss '
                    '(name of layers in tf.keras.applications.VGG19 model)')

  # Dataset parameters
  flags.DEFINE_enum('dataset', 'places365_small', ['places365_small', 'div2k'],
                    'Name of dataset')
  flags.DEFINE_integer('sr_scale', None, 'SR scale factor')

  return flags.FLAGS


def train_flags():
  """Flags for joint training."""
  # Training parameters
  flags.DEFINE_string('work_dir', './train',
                      'Desired working directory to save checkpoints')
  flags.DEFINE_string('mask_type', 'small', 'Mask type: small or large')
  flags.DEFINE_integer('batch_size', 16, 'Mini-batch size')
  flags.DEFINE_integer(
      'num_epochs', 100,
      'Total number of epochs: 100 with small masks, 1500 with large masks')
  flags.DEFINE_multi_integer(
      'lr_boundary', [80, 90],
      'Learning rate boundary: [80, 90] with small masks, '
      '[1200, 1350] with large masks')
  flags.DEFINE_multi_float('lr_values', [0.001, 0.0001, 0.00001],
                           'Learning rate for each interval: '
                           '[0.001, 0.0001, 0.00001] with small masks, '
                           '[0.00001, 0.000001, 0.0000001] with large masks')
  flags.DEFINE_float('lr_d_ratio', 1, 'Discriminator learning rate ratio')
  flags.DEFINE_multi_integer('patch_size', [256, 256, 3], 'Training patch size')

  # Loss coefficients
  flags.DEFINE_float('lambda_l1', 1, 'Reconstruction loss coefficient')
  flags.DEFINE_float('lambda_vgg', 0.01, 'VGG loss coefficient')
  flags.DEFINE_float('lambda_vgg_ref', 0.00001,
                     'VGG loss coefficient - refinement net')
  flags.DEFINE_float('lambda_gan', 0.5, 'GAN loss coefficient')
  flags.DEFINE_enum('vgg_layer', 'block4_conv1',
                    ['block1_conv1', 'block2_conv1', 'block3_conv1',
                     'block4_conv1'],
                    'VGG19 layer to use for perceptual loss '
                    '(name of layers in tf.keras.applications.VGG19 model)')

  # Pre-trained weights. (Checkpoints in 'pretrain_dir_coarse',
  # 'pretrain_dir_refine' and 'pretrain_dir_sr' will only be loaded if
  # 'pretrain_dir_all' is set to None.)
  flags.DEFINE_string('pretrain_dir_all', None,
                      'Directory with pretrained weights for all components '
                      'in the framework. If None is given, training proceeds '
                      'without pretrained weights.')
  flags.DEFINE_string('pretrain_dir_coarse', None,
                      'Directory with pretrained weights of the coarse '
                      'network')
  flags.DEFINE_string(
      'pretrain_dir_refine', None,
      'Directory with pretrained weights of the refinement network')
  flags.DEFINE_string(
      'pretrain_dir_sr', None,
      'Directory with pretrained weights of the super-resolution network')

  # Dataset parameters
  flags.DEFINE_enum('dataset', 'div2k', ['div2k'], 'Name of dataset')
  flags.DEFINE_integer('sr_scale', 2, 'SR scale factor')

  return flags.FLAGS


def test_flags():
  """Flags for testing."""
  flags.DEFINE_bool('eval', False, 'Whether to evaluate with quality metrics. '
                    '--eval flag should only given if full images are provided '
                    'to --img_dir.')
  flags.DEFINE_string('img_dir', './data/div2k/image',
                      'Directory containing images to inpaint. '
                      'Masked images or full images can be given.')
  flags.DEFINE_string('mask_dir', './data/div2k/mask/small',
                      'Directory containing inpainting masks. Masks should '
                      'have a binary value, with 1 for hole regions.')
  flags.DEFINE_string('ckpt_dir', './ckpt/ckpt-1500',
                      'Directory containing checkpoints.')
  flags.DEFINE_string('result_dir', './results',
                      'Desired directory for saving inpainted results.')
  flags.DEFINE_integer('sr_scale', 2, 'SR scale factor')

  return flags.FLAGS
