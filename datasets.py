#!/usr/bin/python3

from functools import partial
import numpy as np
import cv2
import PIL
import tensorflow_datasets as tfds
import albumentations

def ImageNetSR(split = 'train', **kwargs):
  assert split in {'train', 'validation'}
  size = kwargs.get('size', 256)
  downscale_f = kwargs.get('downscale_f', 4)
  degradation = kwargs.get('degradation', 'pil_nearest') # <interpolation_fn>
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
    interpolation_fn = {
      "cv_nearest": cv2.INTER_NEAREST,
      "cv_bilinear": cv2.INTER_LINEAR,
      "cv_bicubic": cv2.INTER_CUBIC,
      "cv_area": cv2.INTER_AREA,
      "cv_lanczos": cv2.INTER_LANCZOS4,
      "pil_nearest": PIL.Image.NEAREST,
      "pil_bilinear": PIL.Image.BILINEAR,
      "pil_bicubic": PIL.Image.BICUBIC,
      "pil_box": PIL.Image.BOX,
      "pil_hamming": PIL.Image.HAMMING,
      "pil_lanczos": PIL.Image.LANCZOS,
    }[degradation]
    degradation_process = partial(PIL.Image.resize, size = int(size / downscale_f), resample = interpolation_fn) if degradation.startswith("pil_") else \
                          albumentations.SmallestMaxSize(max_size = int(size / downscale_f), interpolation = interpolation_fn)
    if degradation.startswith("pil_"):
      image_pil = PIL.Image.fromarray(image)
      LR_image = self.degradation_process(image_pil)
      LR_image = np.array(LR_image).astype(np.uint8)
    else:
      LR_image = self.degradation_process(image=image)["image"]
    return {'image': (image/127.5 - 1.0).astype(np.float32), 'LR_image': (LR_image/127.5 - 1.0).astype(np.float32)}

  ds = tfds.load('imagenet2012', split = split, shuffle_files = True).map(parse_function)

if __name__ == "__main__":
  valset = ImageNetSR(split = 'validation')
  for sample in valset:
    print(sample)
