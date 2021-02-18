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

"""Utility functions related to tf.summary."""

import tensorflow as tf


def scalar_summary_from_dict(writer, scalar_data, step, prefix=''):
  with writer.as_default():
    for name, value in scalar_data.items():
      tf.summary.scalar(prefix+name, value, step=step)


def image_summary_from_dict(writer, image_data, step, max_num):
  with writer.as_default():
    for name, value in image_data.items():
      tf.summary.image(name, value, max_outputs=max_num, step=step)
