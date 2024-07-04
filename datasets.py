#!/usr/bin/python3

import numpy as np
import tensorflow_datasets as tfds
import albumentations

def ImageNetSR(split = 'train', **kwargs):
  assert split in {'train', 'validation'}
  size = kwargs.get('size', 256)
  downscale_f = kwargs.get('downscale_f', 4)
  degradation = kwargs.get('degradation', 'bsrgan') # 'bsrgan' or 'bsrgan_light' or <interpolation_fn>
  min_crop_f = kwargs.get('min_crop_f', 0.5)
  max_crop_f = kwargs.get('max_crop_f', 1.)
  crop_method = kwargs.get('crop_method', 'random crop') # 'random crop' or 'center crop'

  def parse_function(example):
    image, label = example['image'], example['label']
    if not image.mode == 'RGB':
      image = image.convert('RGB')
    image = np.array(image).astype(np.uint8)
    min_side_len = min(image.shape[:2])
    crop_side_len = min_side_len * np.random.uniform(min_crop_f, max_crop_f, size=None)
    crop_side_len = int(crop_side_len)
    cropper = albumentations.CenterCrop(height=crop_side_len, width=crop_side_len) if crop_method == 'center crop' else \
              albumentations.RandomCrop(height=crop_side_len, width=crop_side_len)
    image = cropper(image = image)["image"]
    image_rescaler = albumentations.SmallestMaxSize(max_size=size, interpolation=cv2.INTER_AREA)
    image = image_rescaler(image = image)["image"]
    degradation_process = None

  ds = tfds.load('imagenet2012', split = split, shuffle_files = True)


