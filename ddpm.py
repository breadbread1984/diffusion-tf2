#!/usr/bin/python3

import tensorflow as tf
from backbone import DiffusionWrapper
from utils import make_beta_schedule, extract_into_tensor

class DDPMTrainer(tf.keras.Model):
  def __init__(self, input_shape, unet_config, condition_key = None, **kwargs):
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
    super(DDPMTrainer, self).__init__()
    self.timesteps = timesteps
    self.parameterization = parameterization
    self.loss_type = loss_type
    self.v_posterior = v_posterior
    self.loss_weight = loss_weight
    self.elbo_weight = elbo_weight
    self.model = DiffusionWrapper(input_shape, unet_config, condition_key)
    # scheduler
    self.betas = make_beta_schedule(beta_schedule, timesteps, linear_start = linear_start, linear_end = linear_end, cosine_s = cosine_s) # betas.shape = (timesteps)
    self.alphas = 1. - self.betas # alpha_t
    self.alphas_cumprod = tf.math.cumprod(self.alphas, axis = 0) # bar{alpha}_t
    self.alphas_cumprod_prev = tf.concat([tf.ones([1,], dtype = tf.float32), self.alphas_cumprod[:-1]], axis = 0)
    self.sqrt_alphas_cumprod = tf.math.sqrt(self.alphas_cumprod) # sqrt(bar{alpha}_t)
    self.sqrt_one_minus_alphas_cumprod = tf.math.sqrt(1. - self.alphas_cumprod) # sqrt(1 - bar{alpha})
    self.log_one_minus_alphas_cumprod = tf.math.log(1. - self.alphas_cumprod)
    self.sqrt_recip_alphas_cumprod = tf.math.sqrt(1. / self.alphas_cumprod)
    self.sqrt_recipm1_alphas_cumprod = tf.math.sqrt(1. / self.alphas_cumprod - 1)
    self.posterior_variance = (1 - self.v_posterior) * self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod) + self.v_posterior * self.betas # delta^2
    self.lvlb_weights = self.betas ** 2 / (2 * self.posterior_variance * self.alphas) * (1 - self.alphas_cumprod) if self.parameterization == "eps" else \
                        0.5 * tf.math.sqrt(self.alphas_cumprod) / (2. * 1 - self.alphas_cumprod)
    self.lvlb_weights = tf.where(tf.math.is_inf(self.lvlb_weights),
                                 tf.fill(self.lvlb_weights.shape,tf.math.reduce_min(self.lvlb_weights)),
                                 self.lvlb_weights) # lvlb_weights[0] = lvlb_weights[1]
    assert not tf.math.reduce_all(tf.math.is_nan(self.lvlb_weights))
    # eps mode: KL(p(x_{t-1} | x_t, x_0) || p_theta(x_{t-1} | x_t)) = lvlb_weights * (eps - model(x0)) + C, where eps ~ U(0,1)
  def q_sample(self, x, t):
    # forward process
    # p(x_t | x_0) = N(x_t; mu = sqrt(bar{alpha}_t) * x_0, sigma = 1 - bar{alpha}_t * I)
    noise = tf.random.uniform(shape = tf.shape(x), dtype = tf.float32)
    return extract_into_tensor(self.sqrt_alphas_cumprod, t, x) * x + extract_into_tensor(self.sqrt_one_minus_alphas_cumprod, t, x) * noise, noise
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
    t = tf.random.uniform(minval = 0, maxval = self.timesteps, shape = (tf.shape(inputs)[0],), dtype = tf.int32)
    x_noisy, noise = self.q_sample(inputs, t)
    model_out = self.model(x_noisy, t)
    target = noise if self.parameterization == 'eps' else x
    loss = tf.math.reduce_mean(self.get_loss(model_out, target, mean = False), axis = (1,2,3)) # loss.shape = (batch,)
    simple_loss = tf.math.reduce_mean(loss)
    vlb_loss = tf.math.reduce_mean(loss * tf.gather(self.lvlb_weights, t) * loss)
    total_loss = simple_loss * self.loss_weight + vlb_loss * self.elbo_weight
    return {'simple_loss': simple_loss, 'vlb_loss': vlb_loss, 'total_loss': total_loss}

class DDPMInfer(tf.keras.Model):
  def __init__(self, input_shape, unet_config, condition_key = None, **kwargs):
    timesteps = kwargs.get('timesteps', 1000)
    beta_schedule = kwargs.get('beta_schedule', "linear")
    linear_start = kwargs.get('linear_start', 1e-4)
    linear_end = kwargs.get('linear_end', 2e-2)
    cosine_s = kwargs.get('cosine_s', 8e-3)
    parameterization = kwargs.get('parameterization', 'eps') # eps or x0
    v_posterior = kwargs.get('v_posterior', 0.)
    super(DDPMInfer, self).__init__()
    self.input_shape = input_shape
    self.timesteps = timesteps
    self.parameterization = parameterization
    self.v_posterior = v_posterior
    self.model = DiffusionWrapper(input_shape, unet_config, condition_key)
    # scheduler
    self.betas = make_beta_schedule(beta_schedule, timesteps, linear_start = linear_start, linear_end = linear_end, cosine_s = cosine_s) # betas.shape = (timesteps)
    self.alphas = 1. - self.betas # alpha_t
    self.alphas_cumprod = tf.math.cumprod(self.alphas, axis = 0) # bar{alpha}_t
    self.alphas_cumprod_prev = tf.concat([tf.ones([1,], dtype = tf.float32), self.alphas_cumprod[:-1]], axis = 0) # bar{alpha}_{t-1}
    self.sqrt_alphas_cumprod = tf.math.sqrt(self.alphas_cumprod) # sqrt(bar{alpha}_t)
    self.sqrt_one_minus_alphas_cumprod = tf.math.sqrt(1. - self.alphas_cumprod) # sqrt(1 - bar{alpha}_t)
    self.log_one_minus_alphas_cumprod = tf.math.log(1. - self.alphas_cumprod) # log(1 - bar{alpha}_t)
    self.sqrt_recip_alphas_cumprod = tf.math.sqrt(1. / self.alphas_cumprod) # sqrt(1 / bar{alpha}_t)
    self.sqrt_recipm1_alphas_cumprod = tf.math.sqrt(1. / self.alphas_cumprod - 1) # sqrt(1 / bar{alpha} - 1)
    self.posterior_variance = (1 - self.v_posterior) * self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod) + self.v_posterior * self.betas
    self.posterior_log_variance_clipped = tf.math.log(tf.maximum(self.posterior_variance, 1e-20))
    self.posterior_mean_coef1 = self.betas * tf.math.sqrt(self.alphas_cumprod_prev) / (1. - self.alphas_cumprod) # beta_t * sqrt(bar{alpha}_{t-1}) / (1 - bar{alpha}_t)
    self.posterior_mean_coef2 = (1. - self.alphas_cumprod_prev) * tf.math.sqrt(self.alphas) / (1. - self.alphas_cumprod) # (1 - bar{alpha}_{t-1}) * sqrt(alpha_t) / (1 - bar{alpha}_t)
  def p_sample(self, x, t, clip_denoised = True, repeat_noise = False):
    # backward process
    model_out = self.model(x, t)
    if self.parameterization == 'eps':
      noise = model_out
      # x_recon = 1/sqrt(bar{alpha}) * eps - sqrt(1 - bar{alpha})/sqrt(bar{alpha}) * noise
      x_recon = extract_into_tensor(self.sqrt_recip_alphas_cumprod, t, x) * x - extract_into_tensor(self.sqrt_recipm1_alphas_cumprod, t, x) * noise
    elif self.parameterization == 'x0':
      x_recon = model_out
    x_recon = tf.clip_by_value(x_recon, -1., 1.) if clip_denoised else x_recon
    posterior_mean = extract_into_tensor(self.posterior_mean_coef1, t, x) * x_recon + \
                     extract_into_tensor(self.posterior_mean_coef2, t, x) * x
    posterior_log_variance = extract_into_tensor(self.posterior_log_variance_clipped, t, x)
    noise = tf.stack([tf.random.uniform(shape = self.input_shape, dtype = tf.float32)] * tf.shape(x)[0], axis = 0)
    nonzero_mask = tf.cond(tf.equal(t,0), true_fn = lambda: tf.zeros_like(x) , false_fn = lambda:tf.ones_like(x)) # zero for t == 0 ones for t != 0
    return posterior_mean + nonzero_mask * tf.math.exp(0.5 * posterior_log_variance) * noise
  def call(self, inputs):
    x_t = tf.random.uniform(shape = self.input_shape, dtype = tf.float32)
    for t in range(self.timesteps)[::-1]:
      x_tm1 = self.p_sample(x_t, t)
      x_t = x_tm1
    return x_t

if __name__ == "__main__":
  #tf.keras.backend.set_floatx('float64')
  trainer = DDPMTrainer(input_shape = [32,32,3], unet_config = {'out_channels': 3})
  loss_dict = trainer(tf.random.normal(shape = (4,32,32,3), dtype = tf.float32))
  print(loss_dict)
