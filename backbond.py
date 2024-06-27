#!/usr/bin/python3

import numpy as np
import tensorflow as tf

def ResBlock(input_shape, out_channels, emb_channels, dropout, use_scale_shift_norm = False, resample = None):
  assert resample in {'up','down',False}
  x = tf.keras.Input(input_shape)
  emb = tf.keras.Input((emb_channels,)) # emb.shape = (batch, emb_channels)
  results = tf.keras.layers.GroupNormalization()(x)
  results = tf.keras.layers.Lambda(lambda x: tf.keras.ops.silu(x))(x)
  if resample == 'up':
    tensor_dim = len(input_shape) - 1
    size = 2 if tensor_dim in {1,2} else (1,2,2)
    Op = {1: tf.keras.layers.UpSampling1D,
          2: tf.keras.layers.UpSampling2D,
          3: tf.keras.layers.UpSampling3D}[tensor_dim]
    results = Op(size = size, interpolation = 'nearest')(results)
    results = Op(size = size, interpolation = 'nearest')(results)
  elif resample == 'down':
    tensor_dim = len(input_shape) - 1
    pool_size = 2 if tensor_dim in {1,2} else (1,2,2)
    strides = 2 if tensor_dim in {1,2} else (1,2,2)
    Op = {1: tf.keras.layers.AveragePooling1D,
          2: tf.keras.layers.AveragePooling2D,
          3: tf.keras.layers.AveragePooling3D}[tensor_dim]
    results = Op(pool_size = pool_size, strides = strides, padding = 'same')(results)
    results = Op(pool_size = pool_size, strides = strides, padding = 'same')(results)
  tensor_dim = len(input_shape) - 1
  results = {1: tf.keras.layers.Conv1D,
             2: tf.keras.layers.Conv2D,
             3: tf.keras.layers.Conv3D}[tensor_dim](out_channels, kernel_size = 3, padding = 'same')(results) # results.shape = input_shape[:-1] + [out_channels]
  emb_results = tf.keras.layers.Lambda(lambda x: tf.keras.ops.silu(x))(emb)
  emb_results = tf.keras.layers.Dense(out_channels * 2 if use_scale_shift_norm else out_channels)(emb_results) # emb_results.shape = (batch, out_channels)
  emb_results = tf.keras.layers.Reshape([1,] * (len(input_shape) - 1) + [emb_results.shape[-1],])(emb_results) # emb_results.shape = (batch, 1,..,1, out_channels)
  if use_scale_shift_norm:
    results = tf.keras.layers.GroupNormalization()(results) # results.shape = (batch, h, w, out_channels)
    scale, shift = tf.keras.layers.Lambda(lambda x: tf.split(x, 2, axis = -1))(emb_results) # scale.shape = shift.shape = (batch, 1, 1, out_channels)
    results = tf.keras.layers.Lambda(lambda x: x[0] * (1 + x[1]) + x[2])([results, scale, shift]) # results.shape = (batch, h, w, out_channels)
  else:
    results = tf.keras.layers.Lambda(lambda x: x[0] + x[1])([results, emb_results]) # results.shape = (batch, h, w, out_channels)
    results = tf.keras.layers.GroupNormalization()(results) # results.shape = (batch, h, w, out_channels)
  results = tf.keras.layers.Lambda(lambda x: tf.keras.ops.silu(x))(results)
  results = tf.keras.layers.Dropout(rate = dropout)(results)
  tensor_dim = len(input_shape) - 1
  results = {1: tf.keras.layers.Conv1D,
             2: tf.keras.layers.Conv2D,
             3: tf.keras.layers.Conv3D}[tensor_dim](out_channels, kernel_size = 3, padding = 'same', kernel_initializer = tf.keras.initializers.Zeros(), bias_initializer = tf.keras.initializers.Zeros())(results) # results.shape = input_shape[:-1] + [out_channels]
  results = tf.keras.layers.Add()([x, results])
  return tf.keras.Model(inputs = (x, emb), outputs = results)

def AttentionBlock(input_shape, num_heads):
  x = tf.keras.Input(input_shape)
  skip = tf.keras.layers.Reshape((-1,input_shape[-1]))(x) # x.shape = (batch, length, c)
  results = tf.keras.layers.GroupNormalization()(skip)
  results = tf.keras.layers.Dense(input_shape[-1], 3 * input_shape[-1])(results) # results.shape = (batch, length, 3 * c)
  q,k,v = tf.keras.layers.Lambda(lambda x: tf.split(x, 3, axis = -1))(results) # q.shape = k.shape = v.shape = (batch, length, c)
  q = tf.keras.layers.Reshape((-1, num_heads, input_shape[-1] // num_heads))(q) # q.shape = (batch, length, h, c // h)
  q = tf.keras.layers.Lambda(lambda x: tf.transpose(x, (0,2,1,3)))(q) # q.shape = (batch, h, length, c // h)
  k = tf.keras.layers.Reshape((-1, num_heads, input_shape[-1] // num_heads))(k) # k.shape = (batch, length, h, c // h)
  k = tf.keras.layers.Lambda(lambda x: tf.transpose(x, (0,2,1,3)))(k) # k.shape = (batch, h, length, c // h)
  v = tf.keras.layers.Reshape((-1, num_heads, input_shape[-1] // num_heads))(v) # v.shape = (batch, length, h, c // h)
  v = tf.keras.layers.Lambda(lambda x: tf.transpose(x, (0,2,1,3)))(v) # v.shape = (batch, h, length, c // h)
  weight = tf.keras.layers.Lambda(lambda x, s: tf.linalg.matmul(x[0], x[1], transpose_b = True) * s, arguments = {'s': 1 / np.sqrt(input_shape[-1] // num_heads)})([q,k]) # weight.shape = (batch, h, length, length)
  weight = tf.keras.layers.Softmax(axis = -1)(weight) # weight.shape = (batch, h, length, length)
  results = tf.keras.layers.Lambda(lambda x: tf.linalg.matmul(x[0],x[1]))([weight, v]) # a.shape = (batch,h,length,c//h)
  results = tf.keras.layers.Lambda(lambda x: tf.transpose(x, (0,2,1,3)))(results) # a.shape = (batch, length,h,c//h)
  results = tf.keras.layers.Reshape((-1, input_shape[-1]))(results) # a.shape = (batch, length, c)
  results = tf.keras.layers.Dense(input_shape[-1], kernel_initializer = tf.keras.initializers.Zeros(), bias_initializer = tf.keras.initializers.Zeros())(results) # results.shape = (batch, length, c)
  results = tf.keras.layers.Add()([skip, results]) # results.shape = (batch, length, c)
  results = tf.keras.Reshape(input_shape)(results) # results.shape = (batch, h, w, c)
  return tf.keras.Model(inputs = x, outputs = results)

def CrossAttention(query_dim, num_heads, dim_head, dropout, context_dim = None):
  x = tf.keras.Input((None, query_dim)) # x.shape = (batch, query_len, query_dim)
  if context_dim is not None:
    context = tf.keras.Input((None, context_dim)) # context.shape = (batch, context_len, context_dim)
  q = tf.keras.layers.Dense(num_head * dim_head, use_bias = False)(x) # q.shape = (batch, query_len, hn * hd)
  context_ = context if use_context else x
  k = tf.keras.layers.Dense(num_head * dim_head, use_bias = False)(context_) # k.shape = (batch, context_len, hn * hd)
  v = tf.keras.layers.Dense(num_head * dim_head, use_bias = False)(context_) # v.shape = (batch, context_len, hn * hd)
  q = tf.keras.layers.Reshape((-1, num_heads, dim_head))(q) # q.shape = (batch, query_len, hn, hd)
  k = tf.keras.layers.Reshape((-1, num_heads, dim_head))(k) # k.shape = (batch, context_len, hn, hd)
  v = tf.keras.layers.Reshape((-1, num_heads, dim_head))(v) # v.shape = (batch, context_len, hn, hd)
  q = tf.keras.layers.Lambda(lambda x: tf.transpose(x, (0,2,1,3)))(q) # q.shape = (batch, hn, query_len, hd)
  k = tf.keras.layers.Lambda(lambda x: tf.transpose(x, (0,2,1,3)))(k) # k.shape = (batch, hn, context_len, hd)
  v = tf.keras.layers.Lambda(lambda x: tf.transpose(x, (0,2,1,3)))(v) # v.shape = (batch, hn, context_len, hd)
  qk = tf.keras.layers.Lambda(lambda x, s: s * tf.linalg.matmul(x[0],x[1],transpose_b = True), arguments = {'s': dim_head ** -0.5})([q,k]) # qk.shape = (batch, head, query_len, context_len)
  attn = tf.keras.layers.Softmax(axis = -1)(qk)
  out = tf.keras.layers.Lambda(lambda x: tf.linalg.matmul(x[0],x[1]))([attn, v]) # out.shape = (batch, hn, query_len, hd)
  out = tf.keras.layers.Lambda(lambda x: tf.transpose(x, (0,2,1,3)))(out) # out.shape = (batch, query_len, hn, hd)
  out = tf.keras.layers.Reshape((-1, num_heads * dim_head))(out) # out.shape = (batch, query_len, hn * hd)
  out = tf.keras.layers.Dense(query_dim)(out) # out.shape = (batch, query_len, query_dim)
  out = tf.keras.layers.Dropout(rate = dropout)(out)
  return tf.keras.Model(inputs = (x, context) if context_dim is not None else x, outputs = out)

def BasicTransformerBlock(dim, num_heads, dim_head, dropout, context_dim = None):
  x = tf.keras.Input((None, dim)) # x.shape = (batch, query_len, dim)
  if context_dim is not None:
    context = tf.keras.Input((None, context_dim)) # context.shape = (batch, context_len, context_dim)
  skip = x
  results = tf.keras.layers.LayerNormalization()(skip)
  results = CrossAttention(dim, num_heads, dim_head, dropout)(results)
  results = tf.keras.layers.Add()([skip, results])
  skip = results
  results = tf.keras.layers.LayerNormalization()(skip)
  results = CrossAttention(dim, num_heads, dim_head, dropout, context_dim = context_dim)([results, context] if context_dim is not None else [results,])
  results = tf.keras.layers.Add()([skip, results])
  skip = results
  results = tf.keras.layers.LayerNormalization()(skip)
  results = tf.keras.layers.Dense(4 * dim, activation = tf.keras.activations.gelu)(results)
  results = tf.keras.layers.Dropout(rate = dropout)(results)
  results = tf.keras.l;ayers.Dense(dim)(results)
  results = tf.keras.layers.Add()([skip, results])
  return tf.keras.Model(inputs = (x, context) if context_dim is not None else x, outputs = results)

def SpatialTransformer(input_shape, num_heads, dim_head, depth, dropout, context_dim = None):
  x = tf.keras.Input(input_shape) # x.shape = (batch, h, w, c)
  if context_dim is not None:
    context = tf.keras.Input((None, context_dim)) # context.shape = (batch, h, w, c)
  results = tf.keras.layers.GroupNormalization()(x)
  results = tf.keras.layers.Dense(num_heads * dim_head)(results) # results.shape = (batch, h, w, d)
  results = tf.keras.layers.Reshape((-1, num_heads * dim_head))(results) # results.shape = (batch, h*w, d)
  for d in range(depth):
    results = BasicTransformerBlock(num_heads * dim_head, num_heads, dim_head, dropout, context_dim)([results, context] if context_dim is not None else [results,])
  results = tf.keras.layers.Reshape(input_shape)(results) # results.shape = (batch, h, w, d)
  results = tf.keras.layers.Dense(input_shape[-1], kernel_initializer = tf.keras.initializers.Zeros(), bias_initializer = tf.keras.initializers.Zeros())(results) # resutls.shape = (batch, h, w, c)
  results = tf.keras.layers.Add()([results, x]) # results.shape = (batch, h, w, c)
  return tf.keras.Model(inputs = (x, context) if context_dim is not None else x, outputs = results)

def UNet(input_shape, **kwargs):
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
  transformer_depth = kwargs.get('transformer_depth', 1)
  context_dim = kwargs.get('context_dim', None)
  use_spatial_transformer = kwargs.get('use_spatial_transformer', True)
  resblock_updown = kwargs.get('resblock_updown', False)
  n_embed = kwargs.get('n_embed', None)
  input_shape = [image_size, image_size, in_channel]

  x = tf.keras.Input(input_shape) # x.shape = (batch, h, w, c)
  if context_dim is not None:
    context = tf.keras.Input((None, context_dim)) # context.shape = (batch, context_len, c)
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
  # 2.1) input blocks
  hiddens = list()
  input_block_chans = [model_channels]
  # input block 1
  tensor_dim = len(input_shape) - 1
  results = {1: tf.keras.layers.Conv1D,
             2: tf.keras.layers.Conv2D,
             3: tf.keras.layers.Conv3D}[tensor_dim](model_channels, kernel_size = 3, padding = 'same')(x) # results.shape = input_shape[:-1] + [model_channels]
  hiddens.append(results)
  # input block 2...
  ch = model_channels
  for level, mult in enumerate(channel_mult):
    for _ in range(num_res_blocks):
      results = ResBlock(input_shape[:-1] + [ch,], out_channels = mult * model_channels, emb_channels = 4 * model_channels, dropout = dropout, use_scale_shift_norm = use_scale_shift_norm, resample = False)([results, emb]) # results.shape = input_shape[:-1] + [mult * model_channels]
      ch = mult * model_channels
      for ds in attention_resolution:
        dim_head, num_heads = (ch // num_heads, num_heads) if num_head_channels == -1 else (num_head_channels, ch // num_head_channels)
        if use_spatial_transformer:
          results = SpatialTransformer(input_shape[:-1] + [ch,], num_heads, dim_head, transformer_depth, dropout, context_dim)([results, context] if context_dim is not None else [results]) # results.shape = input_shape[:-1] + [mult * model_channels]
        else:
          results = AttentionBlock(input_shape[:-1] + [ch,], num_heads)(results) # results.shape = input_shape[:-1] + [mult * model_channels]
      hiddens.append(results)
      input_block_chans.append(ch)
    if level != len(channel_mult) - 1:
      if resblock_updown:
        results = ResBlock(input_shape[:-1] + [ch,], out_channels = ch, emb_channels = 4 * model_channels, dropout = dropout, use_scale_shift_norm = use_scale_shift_norm, resample = 'down')([results, emb]) # results.shape = input_shape[:-1] + [mult * model_channels]
      else:
        tensor_dim = len(input_shape) - 1
        strides = 2 if tensor_dim in {1,2} else (1,2,2)
        results = {1: tf.keras.layers.Conv1D,
                   2: tf.keras.layers.Conv2D,
                   3: tf.keras.layers.Conv3D}[tensor_dim](ch, kernel_size = 3, strides = strides, padding = 'same')(results) # results.shape = input_shape[:-1] + [mutl * model_channels]
      hiddens.append(results)
      input_block_chans.append(ch)
  # 2.2) middle block
  results = ResBlock(input_shape[:-1] + [ch,], out_channels = ch, emb_channels = 4 * model_channels, dropout = dropout, use_scale_shift_norm = use_scale_shift_norm, resample = False)([results, emb]) # results.shape = input_shape[:-1] + [ch,]
  if use_spatial_transformer:
    results = SpatialTransformer(input_shape[:-1] + [ch,], num_heads, dim_head, transformer_depth, dropout, context_dim)([results, context] if context_dim is not None else [results]) # results.shape = input_shape[:-1] + [ch,]
  else:
    results = AttentionBlock(input_shape[:-1] + [ch,], num_heads)(results) # results.shape = input_shape[:-1] + [ch,]
  results = ResBlock(input_shape[:-1] + [ch,], out_channels = ch, emb_channels = 4 * model_channels, dropout = dropout, use_scale_shift_norm = use_scale_shift_norm, resample = False)([results, emb]) # results.shape = input_shape[:-1] + [ch,]
  # 2.3) output blocks
  for level, mult in list(enumerate(channel_mult))[::-1]:
    for i in range(num_res_blocks + 1):
      ich = input_block_chans.pop()
      h = tf.keras.layers.Concatenate(axis = -1)([results, hiddens.pop()]) # h.shape = input_shape[:-1] + [ch + ich]
      results = ResBlock(input_shape[:-1] + [ch + ich,], out_channels = ch + ich, emb_channels = 4 * model_channels, dropout = dropout, use_scale_shift_norm = use_scale_shift_norm, resample = False)([h, emb]) # results.shape = input_shape[:-1] + [ch + ich]
      ch = model_channels * mult
      for ds in attention_resolutions:
        dim_head, num_heads = (ch // num_heads, num_heads) if num_head_channels == -1 else (num_head_channels, ch // num_head_channels)
        if use_spatial_transformer:
          results = SpatialTransformer(input_shape[:-1] + [ch,], num_heads, dim_head, transformer_depth, dropout, context_dim)([results, context] if context_dim is not None else [results])
        else:
          results = AttentionBlock(input_shape[:-1] + [ch,], num_heads)(results)
      if level and i == num_res_blocks:
        if resblock_updown:
          results = ResBlock(input_shape[:-1] + [ch,], ch, emb_channels = 4 * model_channels, dropout = dropout, use_scale_shift_norm = use_scale_shift_norm, resample = 'up')([results, emb])
        else:
          tensor_dim = len(input_shape) - 1
          size = 2 if tensor_dim in {1,2} else (1,2,2)
          results = {1: tf.keras.layers.UpSampling1D,
                     2: tf.keras.layers.UpSampling2D,
                     3: tf.keras.layers.UpSampling3D}[tensor_dim](size = size, interpolation = 'nearest')(results)
          resutls = {1: tf.keras.layers.Conv1D,
                     2: tf.keras.layers.Conv2D,
                     3: tf.keras.layers.Conv3D}[tensor_dim](ch, kernel_size = 3, padding = 'same')(results)
  if n_embed is not None:
    results = tf.keras.layers.GroupNormalization()(results)
    results = tf.keras.layers.Dense(n_embed)(results)
  else:
    results = tf.keras.layers.GroupNormalization()(results)
    results = tf.keras.layers.Lambda(lambda x: tf.keras.ops.silu(x))(results)
    tensor_dim = len(input_shape) - 1
    results = {1: tf.keras.layers.Conv1D,
               2: tf.keras.layers.Conv2D,
               3: tf.keras.layers.Conv3D}[tensor_dim](out_channel, kernel_size = 3, padding = 'same', kernel_initializer = tf.keras.initializers.Zeros(), bias_initializer = tf.keras.initializers.Zeros())(results)
  inputs = [x] + ([context] if context_dim is not None else []) + [timesteps] + ([y] if num_classes is not None else [])
  return tf.keras.Model(inputs = inputs if context_dim is not None else (x, timesteps, y), outputs = results)

if __name__ == "__main__":
  unet = UNet(context_dim = 128)
  x = np.random.normal(size = (1,32,32,4))
  context = np.random.normal(size = (1,32,128))
  timesteps = np.random.randint(low = 0, high = 10, size = (1,))
  results = unet([x,context,timesteps])
