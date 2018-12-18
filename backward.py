import tensorflow as tf
import tensorflow.distributions as tfd
import numpy as np
import functools
import data

class BackwardP(object):
  
  def __init__(self,
               state_size,
               T_max,
               dtype=tf.float32,
               random_seed=None):
    self.dtype=dtype
    self.state_size=state_size
    with tf.variable_scope("backward_p", reuse=tf.AUTO_REUSE):
      self.z0_mu = tf.get_variable(name="z0_mu",
                                   shape=[state_size],
                                   dtype=dtype)
      self.drift = tf.get_variable(name="drift",
                                   shape=[state_size],
                                   dtype=dtype)
      self.T_logits = tf.get_variable(name="T_logits",
                                      shape=[T_max],
                                      dtype=dtype)
    self.T_max = T_max
    self.z_scale = 0.1
    self.x_scale = 0.1
   
  def log_prob(self, zs, xs, T, z_lens, x_lens):
    """Computes the log probability of a set of samples.

    Args:
      zs: A set of [batch_size, max_z_num_timesteps, state_dim] latent states.
      xs: A set of [batch_size, max_x_num_timesteps, state_dim] observations.
      T: A set of [batch_size] integers denoting the number of censored steps.
      z_lens: A set of [batch_size] integers denoting the length of each 
        sequence of zs.
      x_lens: A set of [batch_size] integers denoting the length of each
        sequence of observations. Note that T must equal z_lens - x_lens.
    Returns:
      log_p_z: A [batch_size, max_z_num_timesteps] set of logprobs of zs.
      log_p_x_given_z: A [batch_size, max_x_num_timesteps] set of logprobs of xs.
      log_p_T: A [batch_size] set of logprobs of T.
    """
    # First, reverse the zs
    rev_zs = tf.reverse_sequence(zs, z_lens, seq_axis=1, batch_axis=0)
    batch_size = tf.shape(zs)[0]
    # Compute means of z locations by adding drift to each z
    rev_z_locs = rev_zs[:,:-1,:] + self.drift[tf.newaxis, tf.newaxis,:]
    z0_mu = tf.tile(self.z0_mu[tf.newaxis,tf.newaxis,:], [batch_size,1,1])
    rev_z_locs = tf.concat([z0_mu, rev_z_locs], axis=1)
    # Compute z log probs.
    rev_log_p_z = tfd.Normal(loc=rev_z_locs, scale=self.z_scale).log_prob(rev_zs)
    rev_log_p_z *= tf.sequence_mask(z_lens, dtype=rev_log_p_z.dtype)[:,:,tf.newaxis]
    # Reverse the log probs back
    log_p_z = tf.reverse_sequence(rev_log_p_z, z_lens, seq_axis=1, batch_axis=0)
    log_p_z = tf.reduce_sum(log_p_z, axis=-1)

    # To compute the prob of xs, mask out all zs beyond the first x_len
    masked_zs = zs * tf.sequence_mask(x_lens, maxlen=tf.reduce_max(z_lens), dtype=zs.dtype)[:,:,tf.newaxis]
    masked_zs = masked_zs[:,:tf.reduce_max(x_lens),:]
    log_p_x_given_z = tfd.Normal(loc=masked_zs, scale=self.x_scale).log_prob(xs)
    log_p_x_given_z *= tf.sequence_mask(x_lens, dtype=log_p_x_given_z.dtype)[:,:,tf.newaxis]
    log_p_x_given_z = tf.reduce_sum(log_p_x_given_z, axis=-1)

    log_p_T = tfd.Categorical(logits=self.T_logits).log_prob(T)
    return log_p_z, log_p_x_given_z, log_p_T

class BackwardQ(object):
  
  def __init__(self,
               state_size,
               T_max,
               sigma_min=1e-5,
               dtype=tf.float32,
               random_seed=None):
    self.dtype=dtype
    self.state_size=state_size
    self.z_scale = 0.1
    self.x_scale = 0.1
    self.T_max = T_max
    self.sigma_min = sigma_min
    with tf.variable_scope("backward_q", reuse=tf.AUTO_REUSE):
      self.W_T = tf.get_variable(name="W_T",
                                  shape=[state_size, T_max],
                                  dtype=dtype)
      self.b_T = tf.get_variable(name="b_T",
                                  shape=[T_max],
                                  dtype=dtype)
      self.W_z = tf.get_variable(name="W_z",
                                 shape=[state_size*2+1,state_size])
      self.b_z = tf.get_variable(name="b_z",
                                 shape=[state_size],
                                 dtype=dtype)
      self.log_sigma = tf.get_variable(name="log_sigma",
                                 shape=[T_max+1, state_size],
                                 dtype=dtype)
      
   
  def sample(self, batch_size, xs, x_lens):

    max_seq_len = tf.reduce_max(x_lens)
    rev_xs = tf.reverse_sequence(xs, x_lens, seq_axis=1, batch_axis=0)

    # Sample T
    T_logits = tf.matmul(rev_xs[:,0,:], self.W_T) + self.b_T[tf.newaxis,:]
    q_T = tfd.Categorical(logits=T_logits)
    T = tf.stop_gradient(q_T.sample())
    z_lens = T + x_lens
    log_q_T = q_T.log_prob(T)

    rev_zs_ta = tf.TensorArray(dtype=self.dtype,
                               size=max_seq_len,
                               dynamic_size=True,
                               name="sample_zs")
    rev_log_q_z_ta = tf.TensorArray(dtype=self.dtype,
                                    size=max_seq_len,
                                    dynamic_size=True,
                                    name="log_q_z_ta")
    z0 = tf.zeros([batch_size, self.state_size], dtype=self.dtype)
    t0 = 0
    
    def while_predicate(t, *unused_args):
      return tf.reduce_any(t < T + x_lens)
    
    def while_step(t, prev_z, rev_log_q_z_ta, rev_zs_ta):
      # Compute the distribution over z_{T-t}
      
      # [batch_size] steps till next x
      steps_till_next_x = tf.maximum(T-t,0)
      # Fetch the next x value.
      next_x_ind = tf.minimum(tf.maximum(t-T, 0), x_lens-1)
      r = tf.range(0, batch_size)
      inds = tf.stack([r, next_x_ind], axis=-1)
      x = tf.gather_nd(rev_xs, inds)

      z_loc_input = tf.concat([x, prev_z, tf.to_float(steps_till_next_x)[:,tf.newaxis]], axis=1)
      z_loc = tf.matmul(z_loc_input, self.W_z) + self.b_z[tf.newaxis,:]
      log_sigmas = tf.gather(self.log_sigma, steps_till_next_x)
      z_scale = tf.math.maximum(tf.math.softplus(log_sigmas), self.sigma_min)
      q_z = tfd.Normal(loc=z_loc, scale=z_scale)
      new_z = q_z.sample()
      log_q_new_z = q_z.log_prob(new_z)

      new_z = tf.where(t < z_lens, new_z, tf.zeros_like(new_z))
      log_q_new_z = tf.where(t < z_lens, log_q_new_z, tf.zeros_like(log_q_new_z))
      
      new_rev_log_q_z_ta = rev_log_q_z_ta.write(t, log_q_new_z)
      new_rev_zs_ta = rev_zs_ta.write(t, new_z)
      return t+1, new_z, new_rev_log_q_z_ta, new_rev_zs_ta
    
    # xs are currently [batch_size, steps, state_size].
    # we transpose to [steps, batch_size, state_size] so that scan unpacks along 
    # the first dimension.
    _, _, rev_log_q_z_ta, rev_zs_ta  = tf.while_loop(
        while_predicate, 
        while_step,
        loop_vars=(t0, z0, rev_log_q_z_ta, rev_zs_ta),
        parallel_iterations=1
    )
    
    # rev_zs are currently [time, batch_size, state_dim].
    # We transpose to [batch_size, time, state_dim] to be consistent.
    rev_zs = tf.transpose(rev_zs_ta.stack(), [1,0,2])
    zs = tf.reverse_sequence(rev_zs, z_lens, seq_axis=1, batch_axis=0)
    # Sum the log q(z) over the state dimension and then transpose,
    # resulting in a [batch_size, time] Tensor.
    rev_log_q_z = tf.transpose(tf.reduce_sum(rev_log_q_z_ta.stack(), axis=-1), [1,0])
    log_q_z = tf.reverse_sequence(rev_log_q_z, z_lens, seq_axis=1, batch_axis=0)
    return T, log_q_T, zs, log_q_z

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
  trunc_xs *= tf.sequence_mask(trunc_lens, dtype=trunc_xs.dtype)[:,:,tf.newaxis]

  T, _, _, _ = q.sample(batch_size=batch_size,
                        xs=trunc_xs,
                        x_lens=trunc_lens)

  rmse = tf.sqrt(tf.reduce_mean(tf.square(tf.to_float(T - t))))
  return rmse
# def compute_RMSE_at_t(p, q, t, batch_size=10, state_size=1):
#   # get a batch of uncensored data
#   xs, fs, lens, _ = get_forward_process_dataset(state_size=state_size, 
#                                                 batch_size=batch_size,
#                                                 censor=False)
#   mask = lens > t
#   xs = tf.boolean_mask(xs, mask)
#   fs = tf.boolean_mask(fs, mask)
#   lens = tf.boolean_mask(lens, mask)
#   batch_size = tf.shape(lens)[0]
  
#   trunc_lens = lens - t
#   trunc_max_len = tf.reduce_max(trunc_lens)
#   trunc_xs = xs[:,:trunc_max_len,:]
#   log_q_z, zs = q.sample(batch_size=batch_size,
#                          xs=trunc_xs,
#                          lens=trunc_lens)

#   print_op = tf.print([p.drift, p.W_f, p.b_f], summarize=-1)
#   # Pull out the final zs to condition the model on.
#   # zs is [batch_size, time, state_dim]
#   # trunc_lens is [batch_size] inds in [0,trunc_max_len]
#   with tf.control_dependencies([print_op]):
#     r = tf.range(0, batch_size)
#     inds = tf.stack([r,trunc_lens-1], axis=-1)
#     final_zs = tf.gather_nd(zs, inds)

#   _,_,_, pred_lens = p.sample(batch_size=batch_size, z0=final_zs, max_length=50)
#   rmse = tf.sqrt(tf.reduce_mean(tf.square(tf.to_float(pred_lens - t))))
#   return rmse
