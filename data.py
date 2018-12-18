import tensorflow as tf
import tensorflow.distributions as tfd
import numpy as np
import functools

def sample_forward_process(drift=1, failure_z=10, z_scale=0.1, x_scale=0.1, state_size=1, censor=True, car_prob=0.98):
  z_1 = np.random.normal(loc=0., scale=z_scale, size=state_size)
  zs = [z_1]
  censored = False
  while np.all(zs[-1] < failure_z):
    zs.append(np.random.normal(loc=zs[-1] + drift, scale=z_scale, size=state_size))
    if censor and np.random.uniform() > car_prob:
      censored=True
      break
  xs = []
  for z in zs:
    xs.append(np.random.normal(loc=z, scale=x_scale, size=state_size))
  
  fs = np.zeros(len(xs), dtype=np.int32)
  if not censored:
    fs[-1] = 1
  return np.array(xs), fs, len(xs), censored

def forward_process_generator(drift=1, failure_z=10, z_scale=0.1, x_scale=0.1, state_size=1, censor=True, car_prob=0.98):
  while True:
    yield sample_forward_process(drift=drift, 
                                 failure_z=failure_z,
                                 z_scale=z_scale, 
                                 x_scale=x_scale, 
                                 state_size=state_size, 
                                 car_prob=car_prob)
    

def get_forward_process_dataset(state_size, batch_size, censor=True):
  """Creates a set of samples from a survival process.
  
  Args:
    state_size: The dimension of the state and observation of the process.
    batch_size: The number of samples to take.
    
  Returns:
    xs: A [batch_size, max_seq_len, state_size] float64 Tensor containing the 
      sequence  of observations.
    lens: A [batch_size] int32 Tensor containing the lengths of each seqeunce.
    censored: A [batch_size] boolean Tensor containing an indicator of whether
      each sequence was censored or not.
  """
  gen = functools.partial(forward_process_generator, 
                          state_size=state_size,
                          censor=censor)
  dataset = tf.data.Dataset.from_generator(
      gen,
      output_types=(tf.float32, tf.int32, tf.int32, tf.bool),
      output_shapes=([None, state_size], [None], [], []))

  dataset = dataset.padded_batch(
      batch_size, padded_shapes=([None, state_size], [None], [], []))
  
  dataset = dataset.prefetch(10)
  itr = dataset.make_one_shot_iterator()
  xs, fs, lens, censored = itr.get_next()
  return xs, fs, lens, censored
