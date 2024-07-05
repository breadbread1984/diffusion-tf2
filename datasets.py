#!/usr/bin/python3

import tensorflow as tf
import tensorflow_datasets as tfds

def ImageNetSR(split = 'train', **kwargs):
  assert split in {'train', 'validation'}
  size = kwargs.get('size', 256)
  downscale_f = kwargs.get('downscale_f', 4)
  degradation = kwargs.get('degradation', 'nearest') # <interpolation_fn>
  min_crop_f = kwargs.get('min_crop_f', 0.5)
  max_crop_f = kwargs.get('max_crop_f', 1.)
  crop_method = kwargs.get('crop_method', 'random crop') # 'random crop' or 'center crop'

  def parse_function(example):
    image, label = example['image'], example['label']

    # 3) random or center crop patch of size sample in [min_crop_f * min_side_length, max_crop_f * min_side_length]
    min_side_len = tf.math.reduce_min(tf.shape(image)[:2])
    crop_side_len = tf.cast(tf.cast(min_side_len, dtype = tf.float32) * tf.random.uniform(minval = min_crop_f, maxval = max_crop_f, shape = ()), dtype = tf.int32)
    if crop_method == 'center crop':
      h_start_pos = (image.shape[0] - crop_side_len) // 2
      w_start_pos = (image.shape[1] - crop_side_len) // 2
      image = image[h_start_pos:h_start_pos + crop_side_len, w_start_pos:w_start_pos + crop_side_len]
    elif crop_method == 'random crop':
      image = tf.image.random_crop(image, size = (crop_side_len, crop_side_len, 3))
    # 4) scale image to make min_side_length equals to given size
    image = tf.image.resize(image, (size, size), method = tf.image.ResizeMethod.AREA)
    # 5) downscale image to make dimension to (h / downscale_f, w / downscale_f)
    LR_image = tf.image.resize(image, (int(size / downscale_f), int(size / downscale_f)), {
      "nearest": tf.image.ResizeMethod.NEAREST_NEIGHBOR,
      "bilinear": tf.image.ResizeMethod.BILINEAR,
      "bicubic": tf.image.ResizeMethod.BICUBIC,
      "area": tf.image.ResizeMethod.AREA,
      "lanczos": tf.image.ResizeMethod.LANCZOS5}[degradation])
    # 6) scale value of tensor to [-1,1]
    return {'image': tf.cast(image/127.5 - 1.0, dtype = tf.float32), 'LR_image': tf.cast(LR_image/127.5 - 1.0, dtype = tf.float32)}

  ds = tfds.load('imagenet2012', split = split, shuffle_files = True).map(parse_function)
  return ds

if __name__ == "__main__":
  valset = ImageNetSR(split = 'validation')
  for sample in valset:
    print(sample)
