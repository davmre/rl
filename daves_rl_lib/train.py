import jax
from jax import numpy as jnp

import optax

from daves_rl_lib import networks

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
  
  state_value = jax.vmap(
    lambda x: value_net.apply(value_weights, x)[0])(state.obs)

  policy_optimizer_state = policy_optimizer.init(policy_weights)
  value_optimizer_state = value_optimizer.init(value_weights)

  return (
      (state, state_value, state_seeds),
      (policy_net, policy_weights, policy_optimizer_state),
      (value_net, value_weights, value_optimizer_state))

def rollout_trajectory(env,
                       initial_state,
                       initial_state_value,
                       step_fn, 
                       max_num_steps,
                       seeds,
                       policy_weights,
                       policy_optimizer_state,
                       value_weights,
                       value_optimizer_state,
                       done_fn=lambda s: jnp.all(s.done)):
  batch_size = seeds.shape[0]
  policy_trace, value_trace = jax.tree_util.tree_map(
    lambda x: jnp.zeros((batch_size,) + x.shape), (policy_weights, value_weights))
  rollout = [initial_state]
  states = initial_state
  state_values = initial_state_value
  for step in range(max_num_steps):  # TODO rewrite as jittable scan
    (_,
     states, state_values,
     value_weights, value_trace, value_optimizer_state,
     policy_weights, policy_trace, policy_optimizer_state,
     seeds) = step_fn(
          states=states,
          state_values=state_values,
          step=step,
          policy_weights=policy_weights,
          policy_trace=policy_trace,
          policy_optimizer_state=policy_optimizer_state,
          value_weights=value_weights,
          value_trace=value_trace,
          value_optimizer_state=value_optimizer_state,
          seeds=seeds)
    rollout.append(states)
    if done_fn(states):
      break
  return rollout, seeds,  policy_weights, policy_optimizer_state, value_weights, value_optimizer_state


def discounted_return(rollout, discount_factor):
  r = rollout[0].reward
  d = 1.
  for i in range(1, len(rollout)):
    r += jnp.where(rollout[i - 1].done, 0., d * rollout[i].reward)
    d *= discount_factor
  return r