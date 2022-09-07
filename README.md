# Make Your Own Sprites: Aliasing-Aware and Cell-Controllable Pixelization


## Description
This is the official implementation of the SIGGRAPH Asia 2022 paper "Make Your Own Sprites: Aliasing-Aware and Cell-Controllable Pixelization".
**(Now only simple version is available, full version and more information will be updated soon.)**

## Prerequisites
- Linux
- Python 3
- NVIDIA GPU + CUDA CuDNN
- pytorch >= 1.7.1 and torchvision >= 0.8.2

## Dataset
The dataset is available at https://drive.google.com/file/d/1YAjcz6lScm-Gd2C5gj3iwZOhG5092fRo/view?usp=sharing.

## Pretrained Models
| Path | Description
| :--- | :----------
|[Structure Extractor](https://drive.google.com/file/d/1VRYKQOsNlE1w1LXje3yTRU5THN2MGdMM/view?usp=sharing) | A VGG-19 model pretrained on Multi-cell dataset.
|[AliasNet](https://drive.google.com/file/d/17f2rKnZOpnO9ATwRXgqLz5u5AZsyDvq_/view?usp=sharing) | An encoder-decoder network pretrained on Aliasing dataset.

## Train
`python train.py --gpu_ids 0 --batch_size 2 --preprocess none --dataroot DATASET_PATH --name YOUR_MODEL_NAME`

## Test
`python test.py --gpu_ids 0 --batch_size 1 --preprocess none --num_test 4 --epoch WHICH_EPOCH --dataroot DATASET_PATH --name YOUR_MODEL_NAME`

## Acknowledgements
- The code adapted from [pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) and [SCGAN](https://github.com/makeuptransfer/SCGAN).
- The dataset is collected from the Internet. Most of the pixel arts are from [eBoyArts](https://www.eboy.com/pool/everything/1).