#!/usr/bin/python3

import tensorflow as tf

def ResBlock(input_shape, out_channels, emb_channels, dropout, use_scale_shift_norm = False, resample = None):
  assert resample in {'up','down',False}
  x = tf.keras.Input(input_shape)
  emb = tf.keras.Input((emb_channels,)) # emb.shape = (batch, emb_channels)
  results = tf.keras.layers.GroupNormalization()(x)
  results = tf.keras.layers.Lambda(lambda x: tf.keras.ops.silu(x))(x)
  if resample == 'up':
    if len(input_shape - 1) == 1:
      results = tf.keras.layers.UpSampling1D(size = (2,), interpolation = 'nearest')(results)
      results = tf.keras.layers.UpSampling1D(size = (2,), interpolation = 'nearest')(results)
    elif len(input_shape - 1) == 2:
      results = tf.keras.layers.UpSampling2D(size = (2,2), interpolation = 'nearest')(results)
      results = tf.keras.layers.UpSampling2D(size = (2,2), interpolation = 'nearest')(results)
    elif len(input_shape - 1) == 3:
      results = tf.keras.layers.UpSampling3D(size = (1,2,2), interpolation = 'nearest')(results)
      results = tf.keras.layers.UpSampling3D(size = (1,2,2), interpolation = 'nearest')(results)
    else:
      raise Exception('unknown input dimension!')
  elif resample == 'down':
    if len(input_shape - 1) == 1:
      results = tf.keras.layers.AveragePooling1D(pool_size = (2,), strides = (2,), padding = 'same')(results)
      results = tf.keras.layers.AveragePooling1D(pool_size = (2,), strides = (2,), padding = 'same')(results)
    elif len(input_shape - 1) == 2:
      results = tf.keras.layers.AveragePooling2D(pool_size = (2,2), strides = (2,2), padding = 'same')(results)
      results = tf.keras.layers.AveragePooling2D(pool_size = (2,2), strides = (2,2), padding = 'same')(results)
    elif len(input_shape - 1) == 3:
      results = tf.keras.layers.AveragePooling3D(pool_size = (1,2,2), strides = (1,2,2), padding = 'same')(results)
      results = tf.keras.layers.AveragePooling3D(pool_size = (1,2,2), strides = (1,2,2), padding = 'same')(results)
  if len(input_shape - 1) == 1:
    results = tf.keras.layers.Conv1D(out_channels, kernel_size = (3,), padding = 'same')(results) # results.shape = (batch, w, out_channels)
  elif len(input_shape - 1) == 2:
    results = tf.keras.layers.Conv2D(out_channels, kernel_size = (3,3,), padding = 'same')(results) # results.shape = (batch, h, w, out_channels)
  elif len(input_shape - 1) == 3:
    results = tf.keras.layers.Conv3D(out_channels, kernel_size = (3,3,3,), padding = 'same')(results) # results.shape = (batch, t, h, w, out_channels)
  emb_results = tf.keras.layers.Lambda(lambda x: tf.keras.ops.silu(x))(emb)
  emb_results = tf.keras.layers.Dense(out_channels * 2 if use_scale_shift_norm else out_channels)(emb_results) # emb_results.shape = (batch, out_channels)
  if len(input_shape - 1) == 1:
    emb_results = tf.keras.layers.Lambda(lambda x: tf.reshape(tf.shape(x)[0],1,tf.shape(x)[-1]))(emb_results) # emb_results.shape = (batch, 1, out_channels)
  elif len(input_shape - 1) == 2:
    emb_results = tf.keras.layers.Lambda(lambda x: tf.reshape(tf.shape(x)[0],1,1,tf.shape(x)[-1]))(emb_results) # emb_results.shape = (batch, 1, 1, out_channels)
  elif len(input_shape - 1) == 3:
    emb_results = tf.keras.layers.Lambda(lambda x: tf.reshape(tf.shape(x)[0],1,1,1,tf.shape(x)[-1]))(emb_results) # emb_results.shape = (batch, 1, 1, 1, out_channels)
  if use_scale_shift_norm:
    results = tf.keras.layers.GroupNormalization()(results) # results.shape = (batch, h, w, out_channels)
    scale, shift = tf.keras.layers.Lambda(lambda x: tf.split(x, 2, axis = -1))(emb_results) # scale.shape = shift.shape = (batch, 1, 1, out_channels)
    results = tf.keras.layers.Lambda(lambda x: x[0] * (1 + x[1]) + x[2])([results, scale, shift]) # results.shape = (batch, h, w, out_channels)
  else:
    results = tf.keras.layers.Lambda(lambda x: x[0] + x[1])([results, emb_results]) # results.shape = (batch, h, w, out_channels)
    results = tf.keras.layers.GroupNormalization()(results) # results.shape = (batch, h, w, out_channels)
  results = tf.keras.layers.Lambda(lambda x: tf.keras.ops.silu(x))(results)
  results = tf.keras.layers.Dropout(rate = dropout)(results)
  if len(input_shape - 1) == 1:
    results = tf.keras.layers.Conv1D(out_channels, kernel_size = (3,), padding = 'same', kernel_initializer = tf.keras.initializers.Zeros(), bias_initializer = tf.keras.initializers.Zeros())(results)
  elif len(input_shape - 1) == 2:
    results = tf.keras.layers.Conv2D(out_channels, kernel_size = (3,3,), padding = 'same', kernel_initializer = tf.keras.initializer.Zeros(), bias_initializer = tf.keras.initializers.Zeros())(results)
  elif len(input_shape - 1) == 3:
    results = tf.keras.layers.Conv3D(out_channels, kernel_size = (3,3,3,), padding = 'same', kernel_initializer = tf.keras.initializer.Zeros(), bias_initializer = tf.keras.initializers.Zeros())(results)
  results = tf.keras.layers.Add()([x, results])
  return tf.keras.Model(inputs = (x, emb), outputs = results)

def AttentionBlock(input_shape, num_heads = -1, num_head_channels = -1):
  x = tf.keras.Input(input_shape)
  results = tf.keras.layers.Reshape((-1,input_shape[-1]))(x) # x.shape = (batch, h * w, c)
  results = tf.keras.layers.GroupNormalization()(results)
  results = tf.keras.layers.Dense(input_shape[-1], 3 * input_shape[-1])(results) # results.shape = (batch, h * w, 3 * c)
  q,k,v = tf.keras.layers.Lambda(lambda x: tf.split(x, 3, axis = -1))(results) # q.shape = k.shape = v.shape = (batch, h * w, c)
  

def UNet(input_shape, use_context = False, **kwargs):
  image_size = kwargs.get('image_size', 32)
  in_channels = kwargs.get('in_channels', 4)
  model_channels = kwargs.get('model_channels', 256)
  out_channels = kwargs.get('out_channels', 4)
  num_res_blocks = kwargs.get('num_res_blocks', 2)
  attention_resolutions = kwargs.get('attention_resolutions', [4,2,1])
  dropout = kwargs.get('dropout', 0)
  channel_mult = kwargs.get('channel_mult', [1,2,4])
  conv_resample = kwargs.get('conv_resample', True)
  dims = kwargs.get('dims', 2)
  num_classes = kwargs.get('num_classes', None)
  num_heads = kwargs.get('num_heads', -1)
  num_head_channels = kwargs.get('num_head_channels', 32)
  max_period = kwargs.get('max_period', 10000)
  use_scale_shift_norm = kwargs.get('use_scale_shift_norm', False)
  
  h = tf.keras.Input(input_shape) # h.shape = (batch, h, w, c)
  if use_context:
    context = tf.keras.Input(input_shape) # context.shape = (batch, h, w, c)
  timesteps = tf.keras.Input(()) # timesteps.shape = (batch,)
  if num_classes is not None:
    y = tf.keras.Input(()) # y.shape = (batch,)
  # 1) timestep_embedding
  freqs = tf.keras.layers.Lambda(lambda x, p, h: tf.math.exp(-tf.math.log(p) * tf.range(h, dtype = tf.float32) / h), arguments = {'p': max_period, 'h': model_channels // 2})(timesteps) # freqs.shape = (model_channels // 2)
  args = tf.keras.layers.Lambda(lambda x: tf.expand_dims(x[0], axis = -1) * tf.expand_dims(x[1], axis = 0))([timesteps, freqs]) # args.shape = (batch, model_channels // 2)
  embedding = tf.keras.layers.Lambda(lambda x: tf.concat([tf.math.cos(x), tf.math.sin(x)], axis = -1))(args) # embedding.shape = (batch, model_channels // 2 * 2)
  if model_channels % 2:
    embedding = tf.keras.layers.Lambda(lambda x: tf.concat([x, tf.zeros_like(x[:,:1])], axis = -1))(embedding) # embedding.shape = (batch, model_channels)
  emb = tf.keras.layers.Dense(model_channels * 4, activation = tf.keras.activations.silu)(embedding)
  emb = tf.keras.layers.Dense(model_channels * 4)(emb) # emb.shape = (batch, 4 * model_channels)
  if num_classes is not None:
    class_emb = tf.keras.layers.Embedding(num_classes, model_channels * 4)(y) # class_emb.shape = (batch, model_channels * 4)
    emb = tf.keras.layers.Add()([emb, class_emb]) # emb.shape = (batch, model_channels * 4)
  # block 1
  if len(input_shape) - 1 == 1:
    results = tf.keras.layers.Conv1D(model_channels, kernel_size = (3,), padding = 'same')(x) # results.shape = input_shape[:-1] + [model_channels]
  elif len(input_shape) - 1 == 2:
    results = tf.keras.layers.Conv2D(model_channels, kernel_size = (3,3), padding = 'same')(x) # results.shape = input_shape[:-1] + [model_channels]
  elif len(input_shape) - 1 == 3:
    results = tf.keras.layers.Conv3D(modul_channels, kernel_size = (3,3,3), padding = 'same')(x) # results.shape = input_shape[:-1] + [model_channels]
  else:
    raise Exception('unsupported input shape!')
  # block 2...
  ch = model_channels
  for level, mult in enumerate(channel_mult):
    for _ in range(num_res_blocks):
      results = ResBlock(input_shape = input_shape[:-1] + [ch,], out_channels = mult * model_channels, emb_channels = 4 * model_channels, dropout = dropout, use_scale_shift_norm = use_scale_shift_norm, resample = False)([results, emb])
      ch = mult * model_channels
      for ds in attention_resolution:
        dim_head, num_heads = (ch // num_heads, num_heads) if num_head_channels == -1 else (num_head_channels, ch // num_head_channels)

        
