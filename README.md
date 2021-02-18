# Zoom-to-Inpaint: Image Inpainting with High Frequency Details

Reference code for the paper [Zoom-to-Inpaint: Image Inpainting with High Frequency Details](https://arxiv.org/).

Soo Ye Kim, Kfir Aberman, Nori Kanazawa, Rahul Garg, Neal Wadhwa, Huiwen Chang, Nikhil Karnad, Munchurl Kim, Orly Liba, arXiv, 2020. If you use this code or our dataset, please cite our paper:

```
@misc{kim2020zoomtoinpaint,
      title={Zoom-to-Inpaint: Image Inpainting with High Frequency Details}, 
      author={Soo Ye Kim and Kfir Aberman and Nori Kanazawa and Rahul Garg and Neal Wadhwa and Huiwen Chang and Nikhil Karnad and Munchurl Kim and Orly Liba},
      year={2020},
      eprint={2012.09401},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

### Requirements
This code was implemented using Tensorflow 2 with Python 3.6 under a Linux environment.
The required libraries can be viewed in requirements.txt, and can be downloaded using the following command:

`pip install -r requirements.txt`

The training pipelines are implemented with tf.distribute.MirroredStrategy() for distributed learning with up to 8 GPUs on a single worker. Note that the training codes also work on a single GPU or CPUs without any required modification.

## Training
Go through these steps to follow the training scheme in our paper:

1. Pre-training steps:
  * Pre-train the coarse network with:
    `python main.py pretrain --network_mode=coarse`
  * Pre-train the refinement network with:
    `python main.py pretrain --network_mode=refine`
  * Pre-train the super-resolution network.
    `python main.py pretrain --flagfile=pretrain_sr.cfg`

2. Train all components jointly in a GAN framework with small masks:
  `python main.py train --flagfile=train_small_mask.cfg`
3. Train all components jointly in a GAN framework with large masks.
  `python main.py train --flagfile=train_large_mask.cfg`

### Notes
* **Pre-training (stage 1)**
  * Weights will be saved in: `./pretrain/[network_mode]/ckpt`
  * Logs for Tensorboard and a text log file will be saved in: `./pretrain/[network_mode]/logs`
  * `[network_mode]: coarse, refine, sr`
* **Main training (stage 2 & 3)**
  * Weights will be saved in: `./train/[mask_type]/ckpt`
  * Logs for Tensorboard and a text log file will be saved in: `./train/[mask_type]/logs`
  * `[mask_type]: small_mask, large_mask`
* If you've followed all the training steps (same training scheme as our paper), the final weights would be the ones in: `./train/large_mask/ckpt`

## Testing

### Directory structure
* Add the provided data under a directory named 'data'
* Add the provided checkpoints under a directory named 'ckpt'

```
Zoom-to-Inpaint
├── ckpt
│    ├── checkpoint
│    ├── ckpt-1500.data
│    └── ckpt-1500.index
└── data
     ├── div2k
     │    ├── image
     │    │    ├── 0001.png
     │    │    ├── ...
     │    │    └── 0100.png
     │    ├── mask
     │    │    ├── large
     │    │    │      └── ...
     │    │    └── small
     │    │          └── ...
     │    └── masked
     │        ├── large
     │        │      └── ...
     │        └── small
     │              └── ...
     ├── places_test
     │    └── ...
     └── places_val
         └── ...
```

### Quick Start
* Run: `python main.py test`
  * Result images will be saved in `./results`.
* To print metric values: `python main.py test --eval`

### Flags
* `--img_dir=[path]`: Directory containing images (PNG) to inpaint.
* `--mask_dir=[path]`: Directory containing the corresponding inpainting masks (PNG).
* `--result_dir=[path]`: Desired directory for saving inpainted results.
* `--eval`: Add this flag if you wish to compute and print metric values.

#### Testing on provided data
```
python main.py test --img_dir='./data/[dataset]/image' --mask_dir='./data/[dataset]/mask/[mask_type]' --result_dir='./results'
```
  * `[dataset]: div2k, places_val, places_test`
  * `[mask_type]: small, large`
  * Result images will be saved in `./results`.
* Add `--eval` flag to evaluate on quality metrics.

#### Testing on your own data
* Run `python main.py test` with appropriate flag values set to `--img_dir`, `--mask_dir`, and `--result_dir`
  * Files in `--mask_dir` directory should have the same file name as their corresponding images in `--img_dir`.
  * `--eval` flag for evaluating on performance metrics should only be used if full images (without holes) are given to `--img_dir`.

#### Testing with self-trained weights
* Provide your checkpoint directory to `--ckpt_dir` (eg. `--ckpt_dir=./train/large_mask/ckpt/ckpt-1500`)

## Additional notes
* You can also set other hyperparameters in config.py by passing them as a flag or directly modifying the default values in the file.
  * Note that the location of the working directory can be changed with the `--work_dir` flag.

## Disclaimer
This is not an officially supported Google product.
