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

"""Main function.

Main function for pre-training, training and testing the Zoom-to-Inpaint
framework. Required libraries can be downloaded with:
pip install -r requirements.txt

Training Zoom-to-Inpaint:
1. Pre-training by:
python main.py pretrain --network_mode=coarse
python main.py pretrain --network_mode=refine
python main.py pretrain --flagfile=pretrain_sr.cfg
2. Joint training with small masks:
python main.py train --flagfile=train_small_mask.cfg
3. Joint training with large masks:
python main.py train --flagfile=train_large_mask.cfg

Testing Zoom-to-Inpaint:
* To test on provided test set:
python main.py test --img_dir='./data/[dataset]/image'
--mask_dir='./data/[dataset]/mask/[mask_type]' --result_dir='./results'
(Available [dataset]: div2k, places_val, places_test,
available [mask_type]: small, large)
* To additionally print metric values, add --eval flag:
python main.py test --eval

Please refer to README.md for more details.
"""

import argparse
import sys
import test

import config
import pretrain
import train


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('mode', type=str, choices=['pretrain', 'train', 'test'])
  args, flags_from_args = parser.parse_known_args()

  if args.mode == 'pretrain':
    flags = config.pretrain_flags()
    # overwrite with command line specified flags
    flags(sys.argv[:1] + flags_from_args)
    pipeline = pretrain.TrainingPipeline(flags)
    pipeline.build_model()
    pipeline.train()
    print('Pre-training finished!!!')

  if args.mode == 'train':
    flags = config.train_flags()
    # overwrite with command line specified flags
    flags(sys.argv[:1] + flags_from_args)
    pipeline = train.TrainingPipeline(flags)
    pipeline.build_model()
    pipeline.train()
    print('Joint training finished!!!')

  if args.mode == 'test':
    flags = config.test_flags()
    # overwrite with command line specified flags
    flags(sys.argv[:1] + flags_from_args)
    pipeline = test.TestingPipeline(flags)
    pipeline.build_model()
    pipeline.test()
    print('Testing finished!!!')


if __name__ == '__main__':
  main()
