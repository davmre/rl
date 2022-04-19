import collections
import functools
from typing import Any, Dict, Callable, Sequence, Tuple, Optional


import numpy as np
import jax
from jax import numpy as jnp

from flax import struct

import optax

from daves_rl_lib import networks

def rollout_scan_body(learner, step, step_fn):
    (_, learner, diagnostics) = step_fn(
          learner=learner,
          step=step)
    return learner, (learner.states, diagnostics)

def rollout_trajectory(learner, step_fn, max_num_steps):
  return jax.lax.scan(
    functools.partial(rollout_scan_body, step_fn=step_fn),
    init=learner,
    xs=jnp.arange(max_num_steps))

def discounted_return(rollout, discount_factor):
  steps = jnp.arange(rollout.reward.shape[0])[..., None]
  discounted_rewards = rollout.reward * (discount_factor ** steps)
  return discounted_rewards[0] + jnp.sum(
    jnp.where(rollout.done[:-1], 0, discounted_rewards[1:]), axis=0)




def trajectory_diagnostics(initial_learner, 
                           rollout,
                           step_diagnostics,
                           discount_factor,
                           summary_stats=True):
  d = {
    'return': discounted_return(rollout, discount_factor),
    'trajectory_length': jnp.argmax(
      jnp.concatenate(
        [rollout.done, jnp.ones_like(rollout.done[-1:])], axis=0),
      axis=0) + 1,
    'initial_value': (
      initial_learner.states.reward + initial_learner.state_values),
    'other': step_diagnostics}
  return jax.tree_util.tree_map(summarize, d) if summary_stats else d

def advance_epoch(learner, env, value_net, step_fn, max_num_steps,
                  discount_factor, return_states=False):
  learner = reset_trajectory(env, learner, value_net)
  updated_learner, (rollout, step_diagnostics) = rollout_trajectory(
    learner, step_fn=step_fn, max_num_steps=max_num_steps)
  diagnostics = trajectory_diagnostics(
      learner, rollout, step_diagnostics, discount_factor)
  if return_states:
    return updated_learner, rollout, diagnostics
  return updated_learner, diagnostics