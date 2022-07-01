import jax
from jax import numpy as jnp
import numpy as np

import optax
from tensorflow_probability.substrates import jax as tfp

from daves_rl_lib import drivers
from daves_rl_lib import networks
from daves_rl_lib.algorithms import policy_gradient
from daves_rl_lib.environments import trivial_environment
from daves_rl_lib.internal import test_util


class EpisodicPolicyGradientTests(test_util.TestCase):

    def test_learns_in_trivial_discrete_environment(self):

        discount_factor = 0.6
        env = trivial_environment.DiscreteTargetEnvironment(
            size=2,
            dim=1,
            discount_factor=discount_factor,
            one_hot_features=True)

        states = env.reset(seed=test_util.test_seed())
        initial_state_obs = states.observation

        seed = test_util.test_seed()

        agent = policy_gradient.EpisodicPolicyGradientAgent(
            policy_net=networks.make_model(
                [32, env.action_space.num_actions],  # type: ignore
                obs_size=env.observation_size,
                activate_final=networks.categorical_from_logits),
            policy_optimizer=optax.adam(0.1),
            discount_factor=discount_factor,
            max_num_steps=100)

        weights = agent.init_weights(
            seed=test_util.test_seed(),
            dummy_observation=states.observation,
            dummy_action=env.action_space.dummy_action())

        step_fn = jax.jit(drivers.jax_driver(env=env, agent=agent))

        done = []
        returns = []
        for _ in range(64):
            states, weights, seed = step_fn(states, weights, seed)
            done.append(states.done)
            returns.append(states.episode_return)

        warmup_steps = 16
        done = jnp.array(done[warmup_steps:])
        returns = jnp.array(returns[warmup_steps:])

        num_episodes = jnp.sum(done)
        final_returns = jnp.where(done, returns, 0.)
        mean_return = jnp.sum(final_returns) / num_episodes

        self.assertAllClose(env.discount_factor, mean_return, rtol=0.1)
