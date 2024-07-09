#!/usr/bin/python3

from os.path import exists, join
from absl import flags, app
import cv2
from ddpm import DDPMInfer

FLAGS = flags.FLAGS

def add_options():
  flags.DEFINE_string('ckpt', default = 'ckpt', help = 'path to ckpt')
  flags.DEFINE_string('output', default = 'output.png', help = 'output_path')
  flags.DEFINE_integer('size', default = 64, help = 'dataset size')
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

def main(unused_argv):
  unet_config_keys = {'model_channels','out_channels','num_res_blocks','num_transformer_blocks','dropout','channel_mult','conv_resample',
                      'num_classes','num_heads','num_head_channels','max_period','use_scale_shift_norm','transformer_depth','context_dim',
                      'use_spatial_transformer','resblock_updown','n_embed'}
  unet_config = {k: (FLAGS[k].value if k != 'channel_mult' else [int(v) for v in FLAGS[k].value]) for k in unet_config_keys}
  infer = DDPMInfer(input_shape = (FLAGS.size,FLAGS.size, 3), unet_config = unet_config, ckpt = FLAGS.ckpt)
  if not exists(FLAGS.ckpt): raise Exception('not existing checkpoint!')
  infer.load_weights(join(FLAGS.ckpt, 'variables', 'variables'))
  x_t = tf.random.uniform(shape = self.input_shape_, dtype = tf.float32)
  img = infer(x_t).numpy()
  cv2.imwrite('generated.png', img)
  cv2.imshow('generated image', img)
  cv2.waitKey()

if __name__ == "__main__":
  add_options()
  app.run(main)
