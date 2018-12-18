import tensorflow as tf
import tensorflow.distributions as tfd
import numpy as np
import functools
import data

class ForwardP(object):
  
  def __init__(self,
               state_size,
               sigma_min=1e-5,
               dtype=tf.float32,
               random_seed=None):
    self.dtype=dtype
    self.state_size=state_size
    with tf.variable_scope("forward_model", reuse=tf.AUTO_REUSE):
      self.drift = tf.get_variable(name="drift",
                                   shape=[state_size],
                                   dtype=dtype)
      self.W_f = tf.get_variable(name="W_f",
                                 shape=[state_size,1],
                                 dtype=dtype)
      self.b_f = tf.get_variable(name="b_f",
                                 shape=[1],
                                 dtype=dtype)
    self.z_scale = 0.1
    self.x_scale = 0.1
    self.bern_temp = 10
   
  def log_prob(self, zs, xs, fs, lens):
    # Compute means of z locations by adding drift to each z
    z_locs = zs[:,:-1,:] + self.drift[tf.newaxis, tf.newaxis,:]
    z_locs = tf.pad(z_locs, [[0, 0], [1, 0], [0, 0]], mode="CONSTANT")
    # Compute z log probs.
    log_p_z = tfd.Normal(loc=z_locs, scale=self.z_scale).log_prob(zs)
    # Compute x log probs as normals centered at each z.
    log_p_x_given_z = tfd.Normal(loc=zs, scale=self.x_scale).log_prob(xs)
    # Compute probability of failure log probs.
    # zs are [batch, time, state_size], weight matrix is [state_size, 1]
    # After multiplication should be [batch, time, 1]
    bern_logits = tf.einsum("ijk,kl->ijl", zs, self.W_f) + self.b_f[tf.newaxis, tf.newaxis,:]
    bern_logits = tf.reshape(bern_logits, [tf.shape(bern_logits)[0],
                                           tf.shape(bern_logits)[1]])
    bern_logits *= self.bern_temp
    log_p_f_given_z = tfd.Bernoulli(logits=bern_logits).log_prob(fs)
    # Sum over state dimension.
    log_p = tf.reduce_sum(log_p_z + log_p_x_given_z + log_p_f_given_z[:,:, tf.newaxis],
                          axis=-1)

    # Mask out timesteps past the end.
    log_p *= tf.sequence_mask(lens, dtype=log_p.dtype)
    return log_p
  
  def sample(self, batch_size, z0=None, max_length=50):
    zs_ta = tf.TensorArray(dtype=self.dtype, 
                        size=5,
                        dynamic_size=True,
                        name="sample_zs")
    fs_ta = tf.TensorArray(dtype=tf.int32, 
                        size=5,
                        dynamic_size=True,
                        name="sample_fs")
    
    t0 = tf.constant(0)
    failed = tf.zeros([batch_size], dtype=tf.bool)
    lens = tf.ones([batch_size], dtype=tf.int32)
    if z0 is None:
      z0 = tf.zeros([batch_size, self.state_size], dtype=self.dtype) - self.drift[tf.newaxis,:]
    
    def while_predicate(t, failed, *unused_args):
      return tf.math.logical_and(
        tf.math.reduce_any(tf.math.logical_not(failed)),
        t < 50)
    
    def while_step(t, failed, lens, prev_z, zs_ta, fs_ta):
      # z_loc is [batch_size, state_size]
      z_loc = prev_z + self.drift[tf.newaxis, :]
      # new_zs is [batch_size, state_size]
      new_zs = tfd.Normal(loc=z_loc, scale=self.z_scale).sample()
      # multiply [batch_size, state_size] new_zs by [state_size, 1] W_f
      # then add [:, 1] b_f.
      bern_logits = tf.matmul(new_zs, self.W_f) + self.b_f[tf.newaxis,:]
      bern_logits = tf.reshape(bern_logits, [batch_size])
      bern_logits *= self.bern_temp
      # Sample a [batch_size] set of failure indicators
      new_fs = tfd.Bernoulli(logits=bern_logits).sample()
      # Update Tensorarrays
      new_zs_ta = zs_ta.write(t, tf.where(failed,
                                           tf.zeros_like(new_zs),
                                           new_zs))
      new_fs_ta = fs_ta.write(t, tf.where(failed,
                                           tf.zeros_like(new_fs),
                                           new_fs))
      # Update failure indicators
      new_failed = tf.logical_or(failed, tf.equal(new_fs, 1))
      # Update lengths (add one only if the process hasn't failed)
      new_lens = lens + (1 - tf.to_int32(new_failed))
      return t+1, new_failed, new_lens, new_zs, new_zs_ta, new_fs_ta
    
    _, _, lens, _, zs_ta, fs_ta = tf.while_loop(
        while_predicate,
        while_step,
        loop_vars=(t0, failed, lens, z0, zs_ta, fs_ta),
        parallel_iterations=1
    )
    
    zs = zs_ta.stack()
    fs = fs_ta.stack()
    xs = tfd.Normal(loc=zs, scale=self.x_scale).sample()

    return zs, xs, fs, lens

class ForwardQ(object):
  
  def __init__(self,
               state_size,
               sigma_min=1e-5,
               dtype=tf.float32,
               random_seed=None):
    self.dtype=dtype
    self.state_size=state_size
    with tf.variable_scope("forward_q", reuse=tf.AUTO_REUSE):
      self.W_mu = tf.get_variable(name="W_mu",
                                  shape=[state_size*2,state_size],
                                  dtype=dtype)
      self.b_mu = tf.get_variable(name="b_mu",
                                  shape=[state_size],
                                  dtype=dtype)
      self.log_sigma = tf.get_variable(name="log_sigma",
                                  shape=[state_size],
                                  dtype=dtype)
    self.sigma_min = sigma_min
   
  def sample(self, batch_size, xs, lens):
    
    max_seq_len = tf.reduce_max(lens)
    zs_ta = tf.TensorArray(dtype=self.dtype,
                           size=max_seq_len,
                           dynamic_size=False,
                           name="sample_zs")
    log_q_z_ta = tf.TensorArray(dtype=self.dtype,
                                 size=max_seq_len,
                                 dynamic_size=False,
                                 name="log_q_z_ta")
    xs_ta = tf.TensorArray(dtype=self.dtype,
                           size=max_seq_len,
                           dynamic_size=False,
                           name="xs").unstack(tf.transpose(xs,[1,0,2]))
    z0 = tf.zeros([batch_size, self.state_size], dtype=self.dtype)
    t0 = 0
    
    def while_predicate(t, *unused_args):
      return t < max_seq_len
    
    def while_step(t, prev_z, log_q_z_ta, zs_ta):
      x = xs_ta.read(t)
      # Concatenate the previous z and current x along state dimension
      # z and x are currently [batch, state_size]
      q_input = tf.concat([prev_z, x], 1)
      # Multiply by parameters to create mean vector
      q_loc = tf.matmul(q_input, self.W_mu) + self.b_mu[tf.newaxis, :]
      # Create scale vector by softplussing parameters
      q_scale = tf.math.maximum(tf.math.softplus(self.log_sigma), self.sigma_min)
      # Sample and compute logprob
      q_z = tfd.Normal(loc=q_loc, scale=q_scale)
      new_z = q_z.sample()
      # Update TensorArray
      new_zs_ta = zs_ta.write(t, tf.where(t < lens,
                                          new_z,
                                          tf.zeros_like(new_z)))
      new_log_q_z = q_z.log_prob(new_z)
      new_log_q_z_ta = log_q_z_ta.write(t, tf.where(t < lens,
                                                    new_log_q_z,
                                                    tf.zeros_like(new_log_q_z)))
      return (t+1, new_z, new_log_q_z_ta, new_zs_ta)
    
    # xs are currently [batch_size, steps, state_size].
    # we transpose to [steps, batch_size, state_size] so that scan unpacks along 
    # the first dimension.
    _, _, log_q_z_ta, zs_ta  = tf.while_loop(
        while_predicate, 
        while_step,
        loop_vars=(t0, z0, log_q_z_ta, zs_ta),
        parallel_iterations=1
    )
    
    # zs are currently [time, batch_size, state_dim].
    # We transpose to [batch_size, time, state_dim] to be consistent.
    zs = tf.transpose(zs_ta.stack(), [1,0,2])
    # Sum the log q(z) over the state dimension and then transpose,
    # resulting in a [batch_size, time] Tensor.
    log_q_z = tf.transpose(tf.reduce_sum(log_q_z_ta.stack(), axis=-1), [1,0])
    return log_q_z, zs

def compute_RMSE_at_t(p, q, t, batch_size=10, state_size=1):
  # get a batch of uncensored data
  xs, fs, lens, _ = data.get_forward_process_dataset(
    state_size=state_size, 
    batch_size=batch_size,
    censor=False)
  mask = lens > t
  xs = tf.boolean_mask(xs, mask)
  fs = tf.boolean_mask(fs, mask)
  lens = tf.boolean_mask(lens, mask)
  batch_size = tf.shape(lens)[0]
  
  trunc_lens = lens - t
  trunc_max_len = tf.reduce_max(trunc_lens)
  trunc_xs = xs[:,:trunc_max_len,:]
  log_q_z, zs = q.sample(batch_size=batch_size,
                         xs=trunc_xs,
                         lens=trunc_lens)

  print_op = tf.print([p.drift, p.W_f, p.b_f], summarize=-1)
  # Pull out the final zs to condition the model on.
  # zs is [batch_size, time, state_dim]
  # trunc_lens is [batch_size] inds in [0,trunc_max_len]
  with tf.control_dependencies([print_op]):
    r = tf.range(0, batch_size)
    inds = tf.stack([r,trunc_lens-1], axis=-1)
    final_zs = tf.gather_nd(zs, inds)

  _,_,_, pred_lens = p.sample(batch_size=batch_size, z0=final_zs, max_length=50)
  rmse = tf.sqrt(tf.reduce_mean(tf.square(tf.to_float(pred_lens - t))))
  return rmse
