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
  flags.DEFINE_integer('model_channels', default = 256, help = 'model base channel')
  flags.DEFINE_integer('out_channels', default = 3, help = 'output channel')
  flags.DEFINE_integer('num_res_blocks', default = 2, help = 'how many blocks (1 resblock + n transformer blocks) sharing a same channel number in a stage')
  flags.DEFINE_integer('num_transformer_blocks', default = 3, help = 'how many transformer blocks after one resblock in a stage')
  flags.DEFINE_float('dropout', default = 0.1, help = 'dropout rate')
  flags.DEFINE_list('channel_mult', default = ['1','2','4'], help = 'multiplier to base channel of different stages')
  flags.DEFINE_boolean('conv_resample', default = True, help = 'whether use conv during upsampling or downsampling')
  flags.DEFINE_integer('num_classes', default = None, help = 'if use class context, how many categories')
  flags.DEFINE_integer('num_heads', default = -1, help = 'number of head in transformer')
  flags.DEFINE_integer('num_head_channels', default = 32, help = 'head dimension in transformer')
  flags.DEFINE_float('max_period', default = 10000., help = 'period used in timesteps embedding')
  flags.DEFINE_boolean('use_scale_shift_norm', default = False, help = 'whether the context embedding is plus or scale plus to hidden tensor')
  flags.DEFINE_integer('transformer_depth', default = 1, help = 'how many transformer layers within a transformer block')
  flags.DEFINE_integer('context_dim', default = None, help = 'context embedding sequence channels')
  flags.DEFINE_boolean('use_spatial_transformer', default = True, help = 'whether to use spatial transformer')
  flags.DEFINE_boolean('resblock_updown', default = False, help = 'whether use convolution in upsampling and downsampling')
  flags.DEFINE_integer('n_embed', default = None, help = 'number of code if the model outputs code prediction')
  # trainer options


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
  unet_config_keys = {'model_channels','out_channels','num_res_blocks','num_transformer_blocks','dropout','channel_mult','conv_resample',
                      'num_classes','num_heads','num_head_channels','max_period','use_scale_shift_norm','transformer_depth','context_dim',
                      'use_spatial_transformer','resblock_updown','n_embed'}
  unet_config = {k: (FLAGS[k].value if k != 'channel_mult' else [int(v) for v in FLAGS[k].value]) for k in unet_config_keys}
  model = DDPMTrainer(input_shape = [FLAGS.size, FLAGS.size, 3], unet_config = unet_config)
  callbacks = [
    tf.keras.callbacks.TensorBoard(log_dir = FLAGS.ckpt),
    tf.keras.callbacks.ModelCheckpoint(filepath = FLAGS.ckpt, save_freq = FLAGS.save_freq)
  ]

if __name__ == "__main__":
  add_options()
  app.run(main)
