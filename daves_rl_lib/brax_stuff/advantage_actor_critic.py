import dataclasses
from typing import Any, Callable, Optional
import jax
import jax.numpy as jnp

from flax import struct
import optax

from daves_rl_lib import networks
from daves_rl_lib import train
from daves_rl_lib import util
from daves_rl_lib.brax_stuff import environment_lib


@struct.dataclass
class TraceableQuantities:
    actions: jnp.ndarray
    agent_states: environment_lib.State
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
class A2CLearner:
    agent_states: environment_lib.State
    policy_weights: Any
    policy_optimizer_state: Any
    value_weights: Any
    value_optimizer_state: Any


def initialize_learner(
        env,
        policy_net,  # Should include final logits.
        value_net,  # Should include final [1] output.
        policy_optimizer,
        value_optimizer,
        batch_size=32,
        seed=jax.random.PRNGKey(0),
) -> A2CLearner:
    policy_seed, value_seed, state_seed = jax.random.split(seed, 3)
    policy_weights = policy_net.init(policy_seed)
    value_weights = value_net.init(value_seed)
    return A2CLearner(
        agent_states=environment_lib.initialize_batch(env,
                                                      batch_size=batch_size,
                                                      seed=state_seed),
        value_weights=value_weights,
        value_optimizer_state=value_optimizer.init(value_weights),
        policy_weights=policy_weights,
        policy_optimizer_state=policy_optimizer.init(policy_weights))


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
        env: environment_lib.JAXEnvironment,
        num_steps: int,
        policy_net: networks.FeedForwardModel,
        policy_weights,
        value_net: networks.FeedForwardModel,
        value_weights,
        entropy_regularization: float = 0.01,
        trace_fn: Optional[Callable] = None) -> Callable:

    if trace_fn is None:
        trace_fn = lambda tq: ()

    def run_minibatch(agent_state):
        initial_agent_state = env.reset_if_done(agent_state)

        def loop_body(agent_state, _):
            action, next_state = env.step_policy(
                agent_state,
                policy_fn=lambda obs: policy_net.apply(policy_weights, obs))
            return next_state, (action, next_state)

        final_state, (actions, non_initial_agent_states) = jax.lax.scan(
            loop_body, init=initial_agent_state, xs=jnp.arange(num_steps))

        # length `n + 1` including both initial and final (pre-reset) states
        agent_states = jax.tree_util.tree_map(
            lambda a, b: jnp.concatenate([a[None, ...], b], axis=0),
            initial_agent_state, non_initial_agent_states)
        num_nonterminal_steps = (num_steps - jnp.sum(agent_states.done[:-1]))
        state_values, values_grad = jax.vmap(lambda s: jax.value_and_grad(
            lambda w: value_net.apply(w, s)[0])(value_weights))(
                agent_states.observation)

        final_state_values = jnp.where(agent_states.done[-1], 0.,
                                       state_values[-1])
        td_horizon = jnp.arange(num_steps, 0, -1)
        returns = (rewards_to_go(agent_states.reward[1:],
                                 discount_factor=env.discount_factor) +
                   env.discount_factor**td_horizon * final_state_values)
        advantage = jnp.where(agent_states.done[:-1], 0.,
                              returns - state_values[:-1])
        value_grad = jax.tree_util.tree_map(
            (lambda x: jnp.sum(util.batch_multiply(-advantage, x[:-1]), axis=0)
             / num_nonterminal_steps),
            values_grad,
        )

        policy_grad = batch_policy_gradient(
            policy_net,
            policy_weights,
            batch_obs=agent_states.observation[:-1, ...],
            batch_actions=actions,
            batch_advantages=advantage)
        policy_entropy, entropy_grad = jax.value_and_grad(lambda w: jnp.sum(
            jnp.where(
                agent_states.done[:-1], 0.,
                policy_net.apply(w, agent_states.observation[:-1]).entropy()) /
            num_nonterminal_steps))(policy_weights)
        regularized_policy_grad = jax.tree_util.tree_map(
            lambda a, b: a + entropy_regularization * b, policy_grad,
            entropy_grad)

        diagnostics = trace_fn(
            TraceableQuantities(actions, agent_states, state_values,
                                value_weights, value_grad, returns, advantage,
                                policy_weights, policy_grad, policy_entropy,
                                entropy_grad, regularized_policy_grad,
                                num_nonterminal_steps))

        return final_state, value_grad, regularized_policy_grad, diagnostics

    return run_minibatch


def make_advantage_actor_critic_batch_step(
        env: environment_lib.JAXEnvironment,
        num_steps: int,
        policy_net: networks.FeedForwardModel,
        policy_optimizer,
        value_net: networks.FeedForwardModel,
        value_optimizer,
        entropy_regularization: float = 0.,
        trace_fn=lambda *a, **kw: (),
) -> Callable:

    def scan_body(learner, _=None):
        new_agent_states, value_grads, policy_grads, diagnostics = jax.vmap(
            make_advantage_actor_critic_single_minibatch(
                env=env,
                num_steps=num_steps,
                policy_net=policy_net,
                policy_weights=learner.policy_weights,
                value_net=value_net,
                value_weights=learner.value_weights,
                entropy_regularization=entropy_regularization,
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
            A2CLearner(agent_states=new_agent_states,
                       policy_weights=policy_weights,
                       policy_optimizer_state=policy_optimizer_state,
                       value_weights=value_weights,
                       value_optimizer_state=value_optimizer_state),
            diagnostics,
        )

    return scan_body
