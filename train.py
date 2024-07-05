#!/usr/bin/python3

from absl import app, flags
import tensorflow as tf
from datasets import ImageNetSR
from ddpm import DDPMTrainer

FLAGS = flags.FLAGS

def add_options():
  flags.DEFINE_string('ckpt', default = 'ckpt', help = 'path to checkpoint')
  flags.DEFINE_integer('batch', default = 12, help = 'batch size')
  flags.DEFINE_integer('save_freq', default = 1000, help = 'save checkpoint frequency')
  # dataset options
  flags.DEFINE_integer('size', default = 256, help = 'dataset size')
  flags.DEFINE_float('downscale_f', default = 4., help = 'downscale rate')
  flags.DEFINE_enum('degradation', default = 'nearest', enum_values = {'nearest', 'bilinear', 'bicubic', 'area', 'lanczos'}, help = 'method to degradation')
  flags.DEFINE_float('min_crop_f', default = 0.5, help = 'min crop ratio')
  flags.DEFINE_float('max_crop_f', default = 1., help = 'max crop ratio')
  flags.DEFINE_enum('crop', default = 'center', enum_values = {'center', 'random'}, help = 'crop method')
  # model options


def main(unused_argv):
  crop_method = {'center': 'center crop', 'random': 'random crop'}[FLAGS.crop]
  trainset = ImageNetSR(split = 'train',
                        size = FLAGS.size,
                        downscale_f = FLAGS.downscale_f,
                        degradation = FLAGS.degradation,
                        min_crop_f = FLAGS.min_crop_f,
                        max_crop_f = FLAGS.max_crop_f,
                        crop_method = crop_method).shuffle(FLAGS.batch).prefetch(FLAGS.batch).batch(FLAGS.batch)
  valset = ImageNetSR(split = 'validation',
                      size = FLAGS.size,
                      downscale_f = FLAGS.downscale_f,
                      degradation = FLAGS.degradation,
                      min_crop_f = FLAGS.min_crop_f,
                      max_crop_f = FLAGS.max_crop_f,
                      crop_method = crop_method).shuffle(FLAGS.batch).prefetch(FLAGS.batch).batch(FLAGS.batch)
  model = DDPMTrainer()
  callbacks = [
    tf.keras.callbacks.TensorBoard(log_dir = FLAGS.ckpt),
    tf.keras.callbacks.ModelCheckpoint(filepath = FLAGS.ckpt, save_freq = FLAGS.save_freq)
  ]
