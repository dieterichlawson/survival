import tensorflow as tf
import tensorflow.distributions as tfd
import numpy as np
import functools
# local imports
import forward
import backward
import data

def make_forward_model_graph(batch_size=4, state_size=1, lr=1e-4):
  xs, fs, lens, censored = data.get_forward_process_dataset(
    state_size=1, batch_size=batch_size)
  
  p = forward.ForwardP(state_size=state_size)
  q = forward.ForwardQ(state_size=state_size)

  log_q, q_zs = q.sample(batch_size=batch_size, xs=xs, lens=lens)
  log_p = p.log_prob(q_zs, xs, fs, lens)

  # log_p is currently [batch_size, time]
  # log_q is currently [batch_size, time]
  # take mean in both time and batch dimensions.
  elbo = tf.reduce_mean(tf.reduce_mean(log_p - log_q, axis=-1))
  
  global_step = tf.train.get_or_create_global_step()
  opt = tf.train.AdamOptimizer(lr)
  grads = opt.compute_gradients(-elbo)
  apply_grads_op = opt.apply_gradients(grads, global_step=global_step)
  rmse_at_5 = forward.compute_RMSE_at_t(p, q, 5, batch_size=100, state_size=state_size)
  return elbo, rmse_at_5, apply_grads_op, global_step


def run_forward_model(batch_size=4, state_size=1, lr=1e-3, num_steps=int(1e5), print_every=1000):
  loss, rmse_at_5, train_op, global_step = make_forward_model_graph(
      batch_size, state_size=state_size, lr=lr)
  sess.run(tf.initializers.global_variables())
  for i in range(1, num_steps):
    if i % print_every == 0:
      _, rmse, l, step = sess.run([train_op, rmse_at_5, loss, global_step])
      print("Step %d: %f, %f" % (step, l, rmse))
    else:
      sess.run([train_op])

def make_backward_model_graph(batch_size=4, state_size=1, lr=1e-4):
  xs, fs, lens, censored = data.get_forward_process_dataset(
    state_size=1, batch_size=batch_size)
  
  p = backward.BackwardP(state_size=state_size, T_max=25)
  q = backward.BackwardQ(state_size=state_size, T_max=25)

  T, log_q_T, zs, log_q_z = q.sample(batch_size=batch_size, xs=xs, x_lens=lens)
  log_p_z, log_p_x_given_z, log_p_T = p.log_prob(zs, xs, T, lens+T, lens)

  log_p_z_per_t = tf.reduce_sum(log_p_z, axis=-1)/tf.to_float(T+lens)
  log_p_x_given_z_per_t = tf.reduce_sum(log_p_x_given_z, axis=-1)/tf.to_float(lens)
  log_q_z = tf.reduce_sum(log_q_z, axis=-1)/tf.to_float(T+lens)
  
  elbo = log_p_T + log_p_z_per_t + log_p_x_given_z_per_t - log_q_T - log_q_z
  elbo_sg = tf.stop_gradient(tf.reduce_mean(elbo))
  elbo_ema = tf.train.ExponentialMovingAverage(decay=0.99)
  maintain_elbo_ema_op = elbo_ema.apply([elbo_sg])

  loss = - tf.reduce_mean(elbo + tf.stop_gradient(elbo - elbo_ema.average(elbo_sg))*log_q_T)
  
  global_step = tf.train.get_or_create_global_step()
  opt = tf.train.AdamOptimizer(lr)
  grads = opt.compute_gradients(loss)
  with tf.control_dependencies([maintain_elbo_ema_op]):
    apply_grads_op = opt.apply_gradients(grads, global_step=global_step)
  rmse_at_5 = backward.compute_RMSE_at_t(p, q, 5, batch_size=100, state_size=state_size)
  print_params_op = tf.print([p.z0_mu, p.drift], summarize=-1)
  return tf.reduce_mean(elbo), print_params_op, rmse_at_5, apply_grads_op, global_step

def run_backward_model(batch_size=4, state_size=1, lr=1e-3, num_steps=int(1e5), print_every=1000):
  loss, print_op, rmse_op, train_op, global_step = make_backward_model_graph(batch_size, state_size=state_size, lr=lr)
  sess.run(tf.initializers.global_variables())
  for i in range(1, num_steps):
    if i % print_every == 0:
      _, _, rmse, l, step = sess.run([train_op, print_op, rmse_op, loss, global_step])
      print("Step %d: %f, %f" % (step, l, rmse))
    else:
      sess.run([train_op])

with tf.Session() as sess:
  run_forward_model(batch_size=16)