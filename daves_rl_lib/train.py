import collections
import functools

import numpy as np
import jax
from jax import numpy as jnp

import optax

from daves_rl_lib import networks

Learner = collections.namedtuple(
  'Learner',
  ['states', 'state_values',
   'value_weights', 'value_trace', 'value_optimizer_state',
   'policy_weights', 'policy_trace', 'policy_optimizer_state',
   'seeds'])

def reset_trajectory(env, learner, value_net):
  batch_size = learner.seeds.shape[0]
  policy_trace, value_trace = jax.tree_util.tree_map(
    lambda x: jnp.zeros((batch_size,) + x.shape),
    (learner.policy_weights, learner.value_weights))
  
  seeds, new_seeds = jax.vmap(jax.random.split, in_axes=[0, None], out_axes=1)(
    learner.seeds, 2)
  states = jax.vmap(env.reset, in_axes=0)(seeds)
  state_values = jax.vmap(
    lambda x: value_net.apply(learner.value_weights, x)[0])(states.obs)  
  
  return learner._replace(policy_trace=policy_trace,
                          value_trace=value_trace,
                          states=states,
                          state_values=state_values,
                          seeds=new_seeds)


def initialize_learner(
    env,
    policy_layer_sizes,  # Should include final logits.
    value_layer_sizes,  # Should include final [1] output.
    policy_optimizer,
    value_optimizer,
    policy_activate_final=networks.categorical_from_logits,
    batch_size=32,
    seed=jax.random.PRNGKey(0)):
  policy_seed, value_seed, state_seed = jax.random.split(seed, 3)
  state_seeds = jax.random.split(state_seed, batch_size)
  state = jax.vmap(env.reset, in_axes=0)(state_seeds)

  action_size = env.action_size
  obs_size = state.obs.shape[-1]

  policy_net = networks.make_model(
      layer_sizes=policy_layer_sizes + [action_size],
      obs_size=obs_size,
      activate_final=policy_activate_final)

  value_net = networks.make_model(
      layer_sizes=value_layer_sizes,
      obs_size=obs_size)
  
  policy_weights = policy_net.init(policy_seed)
  value_weights = value_net.init(value_seed)
  policy_trace, value_trace = jax.tree_util.tree_map(
      lambda x: jnp.zeros((batch_size,) + x.shape),
      (policy_weights, value_weights))
    
  state_value = jax.vmap(
    lambda x: value_net.apply(value_weights, x)[0])(state.obs)

  policy_optimizer_state = policy_optimizer.init(policy_weights)
  value_optimizer_state = value_optimizer.init(value_weights)

  return policy_net, value_net, Learner(
      states=state,
      state_values=state_value,
      value_weights=value_weights,
      value_trace=value_trace,
      value_optimizer_state=value_optimizer_state,
      policy_weights=policy_weights,
      policy_trace=policy_trace,
      policy_optimizer_state=policy_optimizer_state,
      seeds=state_seeds)

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


SummaryStats = collections.namedtuple(
  'SummaryStats',
  ['mean', 'std', 'median', 'min', 'max'])

def summarize(x):
  return SummaryStats(mean=jnp.mean(x),
                 std=jnp.std(x),
                 median=jnp.median(x),
                 min=jnp.min(x),
                 max=jnp.max(x))

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
                  discount_factor):
  learner = reset_trajectory(env, learner, value_net)
  updated_learner, (rollout, diagnostics) = rollout_trajectory(
    learner, step_fn=step_fn, max_num_steps=max_num_steps)
  return updated_learner, trajectory_diagnostics(
      learner, rollout, diagnostics, discount_factor)