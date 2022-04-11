import collections
import jax
from jax import numpy as jnp

TargetState = collections.namedtuple('TargetState', ['done', 'reward', 'obs', 'step'])

class TargetEnv(object):

  def __init__(self, size=1, dim=1):
    self._size = size
    self._dim = dim
    self.action_size = dim * 2

  def reset(self, seed=None):
    return TargetState(done=False, reward=jnp.zeros([]), obs=jnp.zeros([self._dim], dtype=jnp.int32), step=jnp.zeros([], dtype=jnp.int32))

  def step(self, state, action):
    # action is an int between 0 and dim*2. 
    action_dim = action % self._dim
    action_dir = (action // self._dim) * 2 - 1
    delta = action_dir * jax.nn.one_hot(action_dim,
                                        num_classes=self._dim,
                                        dtype=state.obs.dtype)
    new_state_pos = state.obs + delta
    new_state_pos = jnp.minimum(jnp.maximum(new_state_pos, -self._size * jnp.ones_like(new_state_pos)), self._size * jnp.ones_like(new_state_pos))
    new_state_done = jnp.all(new_state_pos == self._size * jax.nn.one_hot(0, num_classes=self._dim))
    return TargetState(
      obs=jnp.where(state.done, state.obs, new_state_pos),
      done=jnp.where(state.done, state.done, new_state_done),
      reward=jnp.where(state.done, 0., jnp.where(new_state_done, 1., 0.)),
      step=jnp.where(state.done, state.step, state.step + 1))