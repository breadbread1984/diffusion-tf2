#!/usr/bin/python3

import tensorflow as tf
from backbone import DiffusionWrapper
from utils import make_beta_schedule, extract_into_tensor

class DDPMTrainer(tf.keras.Model):
  def __init__(self, unet_config, condition_key = None, **kwargs):
    timesteps = kwargs.get('timesteps', 1000)
    beta_schedule = kwargs.get('beta_schedule', "linear")
    linear_start = kwargs.get('linear_start', 1e-4)
    linear_end = kwargs.get('linear_end', 2e-2)
    cosine_s = kwargs.get('cosine_s', 8e-3)
    parameterization = kwargs.get('parameterization', 'eps') # eps or x0
    loss_type = kwargs.get('loss_type', 'l2') # l1 or l2
    v_posterior = kwargs.get('v_posterior', 0.)
    loss_weight = kwargs.get('loss_weight', 1.)
    elbo_weight = kwargs.get('elbo_weight', 0.)
    super(DDPM, self).__init__()
    self.timesteps = timesteps
    self.parameterization = parameterization
    self.loss_type = loss_type
    self.v_posterior = v_posterior
    self.loss_weight = loss_weight
    self.elbo_weight = elbo_weight
    self.model = DiffusionWrapper(unet_config, condition_key)
    # scheduler
    self.betas = make_beta_schedule(beta_schedule, timesteps, linear_start = linear_start, linear_end = linear_end, cosine_s = cosine_s) # betas.shape = (timesteps)
    self.alphas = 1. - self.betas # alpha_t
    self.alphas_cumprod = tf.math.cumprod(self.alphas, axis = 0) # bar{alpha}_t
    self.alphas_cumprod_prev = tf.concat([tf.ones([1,]), self.alphas_cumprod[:-1]], axis = 0)
    self.sqrt_alphas_cumprod = tf.math.sqrt(self.alphas_cumprod) # sqrt(bar{alpha}_t)
    self.sqrt_one_minus_alphas_cumprod = tf.math.sqrt(1. - self.alphas_cumprod) # sqrt(1 - bar{alpha})
    self.log_one_minus_alphas_cumprod = tf.math.log(1. - self.alphas_cumprod)
    self.sqrt_recip_alphas_cumprod = tf.math.sqrt(1. / self.alphas_cumprod)
    self.sqrt_recipm1_alphas_cumprod = tf.math.sqrt(1. / self.alphas_cumprod - 1)
    self.posterior_variance = (1 - self.v_posterior) * self.betas * (1. - alphas_cumprod_prev) / (1. - self.alphas_cumpred) + self.v_posterior * betas # delta^2
    self.lvlb_weights = self.betas ** 2 / (2 * self.posterior_variance * self.alphas) * (1 - self.alphas_cumprod) if self.parameterization == "eps" else \
                        0.5 * tf.math.sqrt(self.alphas_cumprod) / (2. * 1 - self.alphas_cumprod)
    # eps mode: KL(p(x_{t-1} | x_t, x_0) || p_theta(x_{t-1} | x_t)) = lvlb_weights * (eps - model(x0)) + C, where eps ~ U(0,1)
  def q_sample(self, x, t):
    # forward process
    # p(x_t | x_0) = N(x_t; mu = sqrt(bar{alpha}_t) * x_0, sigma = 1 - bar{alpha}_t * I)
    noise = tf.random.uniform(shape = inputs.shape)
    return extract_into_tensor(self.sqrt_alphas_cumprod, t, x) * x + extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x) * noise
  def get_loss(self, pred, target, mean = True):
    # pred.shape = (batch, h, w, c)
    # target.shape = (batch, h, w, c)
    if self.loss_type == 'l1':
      loss = tf.math.reduce_mean(tf.math.abs(target - pred)) if mean == True else tf.math.abs(target - pred)
    elif self.loss_type == 'l2':
      loss = tf.math.reduce_mean((target - pred) ** 2) if mean == True else (target - pred) ** 2
    else:
      raise NotImplementedError("unknown loss type")
    return loss
  def call(self, inputs):
    t = tf.random.uniform(minval = 0, maxval = self.timesteps, shape = (inputs.shape[0]), dtype = tf.int32)
    x_noisy = self.q_sample(inputs, t)
    model_out = self.model(x_noisy, t)
    target = noise if self.parameterization == 'eps' else x
    loss = tf.math.reduce_mean(self.get_loss(model_out, target, mean = False), axis = (1,2,3)) # loss.shape = (batch,)
    loss = tf.math.reduce_mean(loss) * self.loss_weight + tf.math.reduce_mean(loss * tf.gather(self.lvlb_weights, t) * loss) * self.elbo_weight
    return loss
