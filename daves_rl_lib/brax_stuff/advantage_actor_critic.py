import dataclasses
from typing import Any, Callable
import jax
import jax.numpy as jnp

from flax import struct
import optax

from daves_rl_lib import networks
from daves_rl_lib import train
from daves_rl_lib import util


@struct.dataclass
class AgentState:
    state: Any
    step: jnp.ndarray
    accumulated_reward: jnp.ndarray
    last_episode_return: jnp.ndarray
    last_episode_length: jnp.ndarray
    seed: jnp.ndarray


@struct.dataclass
class TraceableQuantities:
    actions: jnp.ndarray
    agent_states: Any
    state_values: jnp.ndarray
    value_weights: Any
    value_grad: Any
    returns: jnp.ndarray
    advantage: jnp.ndarray
    policy_weights: Any
    policy_grad: Any
    policy_entropy: jnp.ndarray
    entropy_grad: Any
    regularized_policy_grad: Any
    num_nonterminal_steps: jnp.ndarray


@struct.dataclass
class Learner:
    agent_states: AgentState
    policy_weights: Any
    policy_optimizer_state: Any
    value_weights: Any
    value_optimizer_state: Any


def initialize_learner(
        env,
        policy_layer_sizes,  # Should include final logits.
        value_layer_sizes,  # Should include final [1] output.
        policy_optimizer,
        value_optimizer,
        policy_activate_final=networks.categorical_from_logits,
        batch_size=32,
        seed=jax.random.PRNGKey(0),
):
    policy_seed, value_seed, state_seed = jax.random.split(seed, 3)
    state_seeds = jax.random.split(state_seed, batch_size)
    agent_state = AgentState(state=jax.vmap(env.reset, in_axes=0)(state_seeds),
                             step=jnp.zeros([batch_size], dtype=jnp.int32),
                             accumulated_reward=jnp.zeros([batch_size]),
                             last_episode_return=jnp.zeros([batch_size]),
                             last_episode_length=jnp.zeros([batch_size],
                                                           dtype=jnp.int32),
                             seed=state_seeds)

    obs_size = agent_state.state.obs.shape[-1]

    policy_net = networks.make_model(
        layer_sizes=policy_layer_sizes,
        obs_size=obs_size,
        activate_final=policy_activate_final,
    )

    value_net = networks.make_model(layer_sizes=value_layer_sizes,
                                    obs_size=obs_size)

    policy_weights = policy_net.init(policy_seed)
    value_weights = value_net.init(value_seed)
    policy_optimizer_state = policy_optimizer.init(policy_weights)
    value_optimizer_state = value_optimizer.init(value_weights)
    return (
        policy_net,
        value_net,
        Learner(agent_states=agent_state,
                value_weights=value_weights,
                value_optimizer_state=value_optimizer_state,
                policy_weights=policy_weights,
                policy_optimizer_state=policy_optimizer_state),
    )


def make_take_action(env, policy_net, policy_weights, discount_factor):

    def take_action(agent_state, _):
        seed, next_seed = jax.random.split(agent_state.seed, 2)
        state = agent_state.state

        action_dist = policy_net.apply(policy_weights, state.obs)
        if len(action_dist.batch_shape):
            raise ValueError(
                'A single input produced a batch policy: {}. You may need to '
                'wrap the output distribution with `tfd.Independent`'.format(
                    action_dist))
        action = action_dist.sample(seed=seed)

        next_state = env.step(state, action)
        accumulated_reward = (
            agent_state.accumulated_reward +
            discount_factor**agent_state.step * next_state.reward)
        next_agent_state = AgentState(
            state=next_state,
            step=agent_state.step + 1,
            accumulated_reward=accumulated_reward,
            last_episode_return=jnp.where(next_state.done, accumulated_reward,
                                          agent_state.last_episode_return),
            last_episode_length=jnp.where(next_state.done, agent_state.step + 1,
                                          agent_state.last_episode_length),
            seed=next_seed)
        no_op = dataclasses.replace(agent_state,
                                    state=dataclasses.replace(
                                        state,
                                        reward=jnp.zeros_like(state.reward)))

        next_agent_state = util.tree_where(state.done, no_op, next_agent_state)

        return next_agent_state, (action, next_agent_state)

    return take_action


def enforce_max_num_steps(agent_state, max_num_steps):
    exceeded = agent_state.step >= max_num_steps
    return dataclasses.replace(
        agent_state,
        state=dataclasses.replace(agent_state.state,
                                  done=jnp.where(exceeded, True,
                                                 agent_state.state.done)),
        last_episode_length=jnp.where(exceeded, agent_state.step,
                                      agent_state.last_episode_length),
        last_episode_return=jnp.where(exceeded, agent_state.step,
                                      agent_state.last_episode_return))


def maybe_reset_agent_state(env, agent_state):
    seed, reset_seed = jax.random.split(agent_state.seed, 2)
    initial_agent_state = AgentState(
        state=env.reset(seed=reset_seed),
        accumulated_reward=jnp.zeros_like(agent_state.accumulated_reward),
        step=jnp.zeros_like(agent_state.step),
        last_episode_length=agent_state.last_episode_length,
        last_episode_return=agent_state.last_episode_return,
        seed=seed)
    return util.tree_where(agent_state.state.done, initial_agent_state,
                           agent_state)


def rewards_to_go(rewards, discount_factor):
    size = rewards.shape[0]
    discount_square = jnp.array(
        [[discount_factor**(j - i) if j >= i else 0.0
          for j in range(size)]
         for i in range(size)])
    return jnp.sum(rewards[None, ...] * discount_square, axis=-1)


def batch_policy_gradient(policy_net, policy_weights, batch_obs, batch_actions,
                          batch_advantages):
    """
    Computes `mean(advantages * scores, axis=0)` where
    `scores[i] = grad(policy_lp[i], policy_weights)`
    is the batch of score function vectors.
    """

    def fn_of_weights(w):

        def scaled_lp(adv, action, obs):
            return adv * policy_net.apply(w, obs).log_prob(action)

        scaled_lps = jax.vmap(scaled_lp)(batch_advantages, batch_actions,
                                         batch_obs)
        return jnp.mean(scaled_lps)

    return jax.grad(fn_of_weights)(policy_weights)


def make_advantage_actor_critic_single_minibatch(
        env,
        num_steps: int,
        policy_net: networks.FeedForwardModel,
        policy_weights,
        value_net: networks.FeedForwardModel,
        value_weights,
        discount_factor=0.97,
        entropy_regularization=0.01,
        max_episode_length=None,
        trace_fn=None):

    if trace_fn is None:
        trace_fn = lambda tq: ()

    def run_minibatch(agent_state):
        if max_episode_length is not None:
            agent_state = enforce_max_num_steps(
                agent_state, max_num_steps=max_episode_length)
        agent_state = maybe_reset_agent_state(env, agent_state)
        take_action = make_take_action(env=env,
                                       policy_net=policy_net,
                                       policy_weights=policy_weights,
                                       discount_factor=discount_factor)
        final_agent_state, (actions, almost_all_agent_states) = jax.lax.scan(
            take_action, init=agent_state, xs=jnp.arange(num_steps))

        # length `n + 1` including both initial and final (pre-reset) states
        all_agent_states = jax.tree_util.tree_map(
            lambda a, b: jnp.concatenate([a[None, ...], b], axis=0),
            agent_state, almost_all_agent_states)
        num_nonterminal_steps = (num_steps -
                                 jnp.sum(all_agent_states.state.done[:-1]))
        state_values, values_grad = jax.vmap(lambda s: jax.value_and_grad(
            lambda w: value_net.apply(w, s)[0])(value_weights))(
                all_agent_states.state.obs)

        final_state_values = jnp.where(all_agent_states.state.done[-1], 0.,
                                       state_values[-1])
        td_horizon = jnp.arange(num_steps, 0, -1)
        returns = (rewards_to_go(all_agent_states.state.reward[1:],
                                 discount_factor=discount_factor) +
                   discount_factor**td_horizon * final_state_values)
        advantage = jnp.where(all_agent_states.state.done[:-1], 0.,
                              returns - state_values[:-1])
        value_grad = jax.tree_util.tree_map(
            (lambda x: jnp.sum(util.batch_multiply(-advantage, x[:-1]), axis=0)
             / num_nonterminal_steps),
            values_grad,
        )

        policy_grad = batch_policy_gradient(
            policy_net,
            policy_weights,
            batch_obs=all_agent_states.state.obs[:-1, ...],
            batch_actions=actions,
            batch_advantages=advantage)
        policy_entropy, entropy_grad = jax.value_and_grad(lambda w: jnp.sum(
            jnp.where(
                all_agent_states.state.done[:-1], 0.,
                policy_net.apply(w, all_agent_states.state.obs[:-1]).entropy())
            / num_nonterminal_steps))(policy_weights)
        regularized_policy_grad = jax.tree_util.tree_map(
            lambda a, b: a + entropy_regularization * b, policy_grad,
            entropy_grad)

        diagnostics = trace_fn(
            TraceableQuantities(actions, all_agent_states, state_values,
                                value_weights, value_grad, returns, advantage,
                                policy_weights, policy_grad, policy_entropy,
                                entropy_grad, regularized_policy_grad,
                                num_nonterminal_steps))

        return final_agent_state, value_grad, regularized_policy_grad, diagnostics

    return run_minibatch


def make_advantage_actor_critic_batch_step(
        env,
        num_steps,
        policy_net,
        policy_optimizer,
        value_net,
        value_optimizer,
        entropy_regularization=0.,
        discount_factor=1.,
        max_episode_length=None,
        trace_fn=lambda *a, **kw: (),
):

    def scan_body(learner, _=None):
        new_agent_states, value_grads, policy_grads, diagnostics = jax.vmap(
            make_advantage_actor_critic_single_minibatch(
                env=env,
                num_steps=num_steps,
                policy_net=policy_net,
                policy_weights=learner.policy_weights,
                value_net=value_net,
                value_weights=learner.value_weights,
                discount_factor=discount_factor,
                entropy_regularization=entropy_regularization,
                max_episode_length=max_episode_length,
                trace_fn=trace_fn))(learner.agent_states)
        value_grad, policy_grad = jax.tree_util.tree_map(
            lambda x: jnp.mean(x, axis=0), (value_grads, policy_grads))

        value_updates, value_optimizer_state = value_optimizer.update(
            value_grad, learner.value_optimizer_state)
        value_weights = optax.apply_updates(learner.value_weights,
                                            value_updates)
        policy_updates, policy_optimizer_state = policy_optimizer.update(
            # We want to *maximize* reward, rather than minimize loss, so
            # pass negative gradients to the optimizer.
            jax.tree_util.tree_map(lambda g: -g, policy_grad),
            learner.policy_optimizer_state)
        policy_weights = optax.apply_updates(learner.policy_weights,
                                             policy_updates)

        return (
            Learner(agent_states=new_agent_states,
                    policy_weights=policy_weights,
                    policy_optimizer_state=policy_optimizer_state,
                    value_weights=value_weights,
                    value_optimizer_state=value_optimizer_state),
            diagnostics,
        )

    return scan_body
