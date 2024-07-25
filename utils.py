#!/usr/bin/python3

import numpy as np
import tensorflow as tf

def make_beta_schedule(schedule, n_timestep, linear_start = 1e-4, linear_end = 2e-2, cosine_s = 8e-3):
  if schedule == "linear":
    betas = tf.cast(tf.linspace(linear_start ** 0.5, linear_end ** 0.5, n_timestep), dtype = tf.float32) ** 2
  elif schedule == "cosine":
    timesteps = tf.range(n_timestep + 1, dtype = tf.float32) / n_timestep + consine_s
    alphas = timesteps / (1 + cosine_s) * np.pi / 2
    alphas = tf.math.cos(alphas) ** 2
    alphas = alphas / alphas[0]
    betas = 1 - alphas[1:] / alphas[:-1]
    betas = tf.clip_by_value(betas, 0, 0.999)
  elif schedule == "sqrt_linear":
    betas = tf.cast(tf.linspace(linear_start, linear_end, n_timestep), dtype = tf.float32)
  elif schedule == "sqrt":
    betas = tf.cast(tf.linspace(linear_start, linear_end, n_timestep), dtype = tf.float32) ** 0.5
  else:
    raise NotImplementedError("unknown schedule!")
  return betas

def extract_into_tensor(a, t, x):
  # a.shape = (timesteps,)
  # t.shape = (batch,)
  # x.shape = (batch, h, w, c)
  out = tf.gather(a, t) # out.shape = (batch,)
  out = tf.reshape(out, [tf.shape(x)[0]] + [1,] * len(x.shape[1:])) # out.shape = (batch, 1,1,1)
  return out
