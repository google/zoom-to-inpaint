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

"""Testing pipeline.

The TestingPipeline class includes model building and test code for testing.

Useful flags:
--img_dir=[path]: Directory containing images to inpaint.
--mask_dir=[path]: Directory containing the corresponding inpainting masks.
--result_dir=[path]: Desired directory for saving inpainted results.
--eval: Add this flag if you wish to compute and print metric values.

Configurations are defined in test_flags() in config.py.
"""

import glob
import os

import net
import numpy as np
import tensorflow as tf
import utils_misc


class TestingPipeline():
  """Testing pipeline."""

  def __init__(self, flags):
    self.eval = flags.eval
    self.img_dir = flags.img_dir
    self.mask_dir = flags.mask_dir
    self.ckpt_dir = flags.ckpt_dir
    self.result_dir = flags.result_dir
    self.sr_scale = flags.sr_scale

    self.img_size = [256, 256, 3]
    self.img_size_hr = [self.img_size[0] * self.sr_scale,
                        self.img_size[1] * self.sr_scale,
                        self.img_size[2]]

  def build_model(self):
    """Build model for testing."""
    network = net.Network(self.img_size)
    network_big = net.Network(self.img_size_hr)
    self.net = network.coarse_network()
    self.net_sr = network.sr_network(self.sr_scale)
    self.net_refine = network_big.refinement_network()

    # set checkpoint
    ckpt = tf.train.Checkpoint(model=self.net,
                               model_sr=self.net_sr,
                               model_refine=self.net_refine)
    # restore checkpoint
    ckpt.restore(self.ckpt_dir).expect_partial()
    print('Restored from {}'.format(self.ckpt_dir))

  def test(self):
    """Test function."""
    test_psnr = []
    test_ssim = []
    test_msssim = []
    test_l1_loss = []

    img_path = sorted(glob.glob(os.path.join(self.img_dir, '*.png')))
    mask_path = sorted(glob.glob(os.path.join(self.mask_dir, '*.png')))
    utils_misc.check_folder(self.result_dir)
    print('Image directory: {}'.format(self.img_dir))
    print('Results will be saved in: {}'.format(self.result_dir))

    for idx in range(len(img_path)):
      # read image and mask
      test_img = utils_misc.read_img_and_normalize(img_path[idx])
      test_mask = tf.expand_dims(utils_misc.read_img_and_normalize(
          mask_path[idx]), axis=3)
      # always resize to self.img_size
      test_img = utils_misc.resize_to_fixed_size(test_img, self.img_size[:-1])
      test_mask = utils_misc.resize_to_fixed_size(test_mask, self.img_size[:-1])
      # prepare inputs
      test_mask_hr = tf.image.resize(
          test_mask, self.img_size_hr[:-1], 'nearest')
      test_input = tf.concat([(1 - test_mask) * test_img, test_mask], axis=3)

      # run model
      output_coarse = self.net.predict(test_input)
      output_sr = self.net_sr.predict(output_coarse)
      output_refine = self.net_refine.predict(
          tf.concat([output_sr, test_mask_hr], axis=3))
      output_refine_lr = tf.image.resize(
          output_refine, self.img_size[:-1], 'bicubic', antialias=True)
      output_refine_lr = output_refine_lr * test_mask + \
          test_input[:, :, :, :self.img_size[2]] * (1. - test_mask)

      # process images
      output_refine_lr = tf.clip_by_value(output_refine_lr, 0, 1).numpy()
      utils_misc.denormalize_and_save_img(output_refine_lr, os.path.join(
          self.result_dir, os.path.basename(img_path[idx])))
      print('[{}/{}]-th image saved!'.format(idx+1, len(img_path)))
      if self.eval:
        psnr_val = utils_misc.measure_psnr(output_refine_lr, test_img, 1.)[0]
        ssim_val = utils_misc.measure_ssim(output_refine_lr, test_img, 1.)[0]
        msssim_val = utils_misc.measure_msssim(
            output_refine_lr, test_img, 1.)[0]
        l1_loss_val = utils_misc.measure_l1_loss(output_refine_lr, test_img)
        test_psnr.append(psnr_val)
        test_ssim.append(ssim_val)
        test_msssim.append(msssim_val)
        test_l1_loss.append(l1_loss_val)
        template = ('PSNR (dB): {}, SSIM: {}, MS-SSIM: {}, L1 Loss: {}')
        print(template.format(psnr_val, ssim_val, msssim_val, l1_loss_val))
    if self.eval:
      template = ('=== Average Test PSNR (dB): {}, SSIM: {}, MS-SSIM: {}, '
                  'L1 Loss: {} ===')
      print(template.format(np.mean(test_psnr), np.mean(test_ssim),
                            np.mean(test_msssim), np.mean(test_l1_loss)))
