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

"""Utility functions for mask generation."""

import cv2
import numpy as np


def random_mask(patch_size, mask_type, seed=None):
  """Random mask.

  This function generates random masks for inpainting. Two different types of
  masks can be generated, either small or large, used in our inpainting model
  (https://arxiv.org/abs/2012.09401). The seed value can be set to fix the mask.

  The large mask is the same as the brush stroke mask used in DeepFillv2:
  https://arxiv.org/abs/1806.03589

  Args:
    patch_size: Size of image patch. (H x W x C)
    mask_type: Type of mask. 'small' or 'large'
    seed: None or an int. If None, no seed is set and the mask is random (eg.
          for training). If a seed value (an int) is given, the same mask is
          generated for the same seed value. It can be used to fix the mask
          (eg. during validation).

  Returns:
    mask: Random mask. (H x W x 1)
  """
  if mask_type == 'small':
    min_num_stroke = 1  # minimum number of strokes
    max_num_stroke = 8  # maximum number of strokes
    min_num_vertex = 1  # minimum number of vertices
    max_num_vertex = 12  # maximum number of vertices
    min_thickness = 5  # minimum thickness of stroke
    max_thickness = 30  # maximum thickness of stroke
  if mask_type == 'large':
    min_num_stroke = 1  # minimum number of strokes
    max_num_stroke = 4  # maximum number of strokes
    min_num_vertex = 4  # minimum number of vertices
    max_num_vertex = 12  # maximum number of vertices
    min_thickness = 12  # minimum thickness of stroke
    max_thickness = 40  # maximum thickness of stroke

  mask = np.zeros((patch_size[0], patch_size[1], 1))
  d = np.sqrt(patch_size[0]**2+patch_size[1]**2)  # length of diagonal
  if seed is not None:
    # set base seed so that unique seeds can be created
    base_seed = max_num_stroke*max_num_vertex
    np.random.seed(5*seed*base_seed)
  # iterate over strokes
  for s in range(np.random.randint(min_num_stroke, max_num_stroke)):
    if seed is not None:
      np.random.seed(5*seed*base_seed+5*s+1)
    start_h = np.random.randint(patch_size[0])
    if seed is not None:
      np.random.seed(5*seed*base_seed+5*s+2)
    start_w = np.random.randint(patch_size[1])
    if seed is not None:
      np.random.seed(5*seed*base_seed+5*s+3)
    thickness = np.random.randint(min_thickness, max_thickness)
    # iterate over vertices
    if seed is not None:
      np.random.seed(5*seed*base_seed+5*s+4)
    for v in range(np.random.randint(min_num_vertex, max_num_vertex)):
      if mask_type == 'small':
        if seed is not None:
          np.random.seed((5*seed+1)*base_seed+s*max_num_vertex+v)
        if v == 0:
          angle = np.random.uniform(0, 2*np.pi)
        else:
          angle = np.pi + angle + np.random.uniform(-np.pi/8, np.pi/8)
        if seed is not None:
          np.random.seed((5*seed+2)*base_seed+s*max_num_vertex+v)
        length = np.random.uniform(1, d/12)
      if mask_type == 'large':
        if seed is not None:
          np.random.seed((5*seed+1)*base_seed+s*max_num_vertex+v)
        min_angle = 2*np.pi/5 - np.random.uniform(high=2*np.pi/15)
        if seed is not None:
          np.random.seed((5*seed+2)*base_seed+s*max_num_vertex+v)
        max_angle = 2*np.pi/5 + np.random.uniform(high=2*np.pi/15)
        if seed is not None:
          np.random.seed((5*seed+3)*base_seed+s*max_num_vertex+v)
        if v % 2 == 0:
          angle = 2*np.pi - np.random.uniform(min_angle, max_angle)
        else:
          angle = np.random.uniform(min_angle, max_angle)
        if seed is not None:
          np.random.seed((5*seed+4)*base_seed+s*max_num_vertex+v)
        length = np.random.normal(d/8, (d/8)//2)
        length = np.clip(length, 0, d/4)

      next_h = start_h + length*np.sin(angle)
      next_w = start_w + length*np.cos(angle)
      next_h = np.clip(next_h, 0, patch_size[0]).astype(np.int32)
      next_w = np.clip(next_w, 0, patch_size[1]).astype(np.int32)

      cv2.circle(mask, (start_h, start_w), thickness//2, 1., -1)
      cv2.line(mask, (start_h, start_w), (next_h, next_w), 1., thickness)
      start_h, start_w = next_h, next_w
    cv2.circle(mask, (start_h, start_w), thickness//2, 1., -1)

  return mask.astype(np.float32)
