import dataclasses
from statistics import mean
from typing import Optional

from absl.testing import parameterized
from importlib_metadata import distribution
import gym

import jax
from jax import numpy as jnp
import numpy as np

import optax
from tensorflow_probability.substrates import jax as tfp

from daves_rl_lib import drivers
from daves_rl_lib import networks
from daves_rl_lib.algorithms import distributional
from daves_rl_lib.environments import environment_lib
from daves_rl_lib.environments import gym_environment
from daves_rl_lib.environments import trivial_environment
from daves_rl_lib.internal import test_util


class ImplicitQuantileTests(test_util.TestCase):

    def test_implicit_quantile_network(self):
        quantile_sample_size = 8

        target = lambda x: tfp.distributions.Normal(x, 1)
        quantile_net = distributional.ImplicitQuantileNetwork(
            state_embedding_network=networks.MLP([32]), layer_sizes=[16, 16, 1])
        weights = quantile_net.init(test_util.test_seed(), [0.], 0.5)
        optimizer = optax.adam(0.01)
        optimizer_state = optimizer.init(weights)

        xs = jnp.linspace(-5., 5., 21)

        def loss_fn(w, seed):
            x_seed, q_seed, y_seed = jax.random.split(seed, 3)
            qs = jax.random.uniform(q_seed, shape=(quantile_sample_size,))
            ys = target(xs).sample(seed=y_seed)
            ysq = quantile_net.apply_at_quantiles(w, xs[..., jnp.newaxis],
                                                  qs)[..., 0]

            qloss = lambda i: (jax.vmap(lambda u: distributional.quantile_loss(
                u, quantile=qs[i]))(ys - ysq[i]))
            return jnp.mean(jax.vmap(qloss)(jnp.arange(quantile_sample_size)))

        gold = target(xs).quantile(jnp.asarray([0.2, 0.8])[..., None])

        @jax.jit
        def do_update(weights, optimizer_state, seed):
            this_seed, seed = jax.random.split(seed, 2)
            loss, grad = jax.value_and_grad(loss_fn)(weights, this_seed)
            updates, optimizer_state = optimizer.update(grad, optimizer_state)
            weights = optax.apply_updates(weights, updates)
            return weights, optimizer_state, seed

        seed = test_util.test_seed()
        for idx in range(512):
            weights, optimizer_state, seed = do_update(weights, optimizer_state,
                                                       seed)
        estimated = quantile_net.apply_at_quantiles(weights, xs[...,
                                                                jnp.newaxis],
                                                    jnp.array([0.2, 0.8]))[...,
                                                                           0]
        self.assertAllClose(estimated, gold, atol=1.0)
        self.assertLessEqual(jnp.linalg.norm(estimated - gold), 3.5)

    @parameterized.named_parameters([('', None), ('_noisy', 0.2)])
    def test_learns_in_trivial_discrete_environment(self, action_noise_prob):

        env = trivial_environment.DiscreteTargetEnvironment(
            size=2,
            dim=1,
            discount_factor=0.6,
            one_hot_features=True,
            action_noise_prob=action_noise_prob)
        buffer_size = 256
        epsilon = 0.2
        states = env.reset(seed=test_util.test_seed(), batch_size=8)
        initial_state_obs = states.observation[0, ...]

        agent = distributional.ImplicitQuantileAgent(
            quantile_net=distributional.ImplicitQuantileNetwork(
                state_embedding_network=networks.MLP([16, 32]),
                layer_sizes=[16, env.action_space.num_actions]),  # type: ignore
            quantile_optimizer=optax.adam(0.01),
            replay_buffer_size=buffer_size,
            num_quantile_samples=6,
            num_target_return_samples=7,
            gradient_batch_size=8,
            epsilon=epsilon,
            target_weights_decay=0.9,
            discount_factor=env.discount_factor)

        weights = agent.init_weights(
            seed=test_util.test_seed(),
            dummy_observation=env.reset(test_util.test_seed()).observation,
            dummy_action=env.action_space.dummy_action())

        step_fn = jax.jit(drivers.jax_driver(env=env, agent=agent))

        seed = test_util.test_seed()
        done = []
        returns = []
        for idx in range(256):
            states, weights, seed = step_fn(states, weights, seed)
            done.append(states.done)
            returns.append(states.episode_return)
        done, returns = jnp.array(done[-128:]), jnp.array(returns[-128:])

        r, d = jnp.reshape(returns, [-1]), jnp.reshape(done, [-1])
        final_returns = r[d]
        mean_return = jnp.mean(final_returns)
        self.assertLess(mean_return, env.discount_factor)
        self.assertGreater(mean_return, (1 - epsilon)**2 * env.discount_factor)

        qs = agent.quantile_net.apply_at_quantiles(
            weights.agent_weights.quantile_weights, initial_state_obs,
            jnp.linspace(0, 1, 21))
        if action_noise_prob is None:
            expected_qs = jnp.mean(qs, axis=0)
            self.assertAllClose(expected_qs,
                                [env.discount_factor**3, env.discount_factor],
                                atol=0.03)
        else:
            # Quantiles should be mostly increasing.
            diffs = jnp.diff(qs, axis=0)
            self.assertGreater(jnp.mean(diffs > 0), 0.8)

    def test_gym_cartpole(self):
        buffer_size = 64
        epsilon = 0.1
        discount_factor = 0.6
        env = gym_environment.GymEnvironment(gym.make("CartPole-v1"),
                                             discount_factor=discount_factor)
        state = env.reset(seed=test_util.test_seed())
        initial_state_obs = state.observation

        agent = distributional.ImplicitQuantileAgent(
            quantile_net=distributional.ImplicitQuantileNetwork(
                state_embedding_network=networks.MLP([32]),
                layer_sizes=[env.action_space.num_actions],  # type: ignore
            ),
            quantile_optimizer=optax.adam(0.1),
            replay_buffer_size=buffer_size,
            num_quantile_samples=6,
            num_target_return_samples=7,
            gradient_batch_size=8,
            epsilon=epsilon,
            target_weights_decay=0.9,
            discount_factor=env.discount_factor)

        weights = agent.init_weights(
            seed=test_util.test_seed(),
            dummy_observation=env.reset(test_util.test_seed()).observation,
            dummy_action=env.action_space.dummy_action())

        step_fn = drivers.stateful_driver(env=env, agent=agent)

        seed = test_util.test_seed()
        for _ in range(256):
            state, weights, seed = step_fn(state, weights, seed)