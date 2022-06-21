import functools

from absl.testing import absltest
from absl.testing import parameterized

import jax
from jax import numpy as jnp

from daves_rl_lib.internal import test_util
from daves_rl_lib.environments import trivial_environment


class TrivialEnvironmentTests(test_util.TestCase):

    def test_one_dim_transitions(self):
        env = trivial_environment.DiscreteTargetEnvironment(size=3, dim=1)
        self.assertEqual(env.action_space.num_actions, 2)

        state = env.reset(seed=test_util.test_seed())
        self.assertSequenceEqual(state.observation, [0])
        # Move left until we bounce off the edge.
        for expected_idx in (-1, -2, -3, -3, -3):
            state = env.step(state, jnp.array(0))
            self.assertSequenceEqual(state.observation, [expected_idx])
            self.assertFalse(state.done)
            self.assertEqual(state.reward, 0.)

        # Now move right
        for expected_idx in (-2, -1, 0, 1, 2):
            state = env.step(state, jnp.array(1))
            self.assertSequenceEqual(state.observation, [expected_idx])
            self.assertFalse(state.done)
            self.assertEqual(state.reward, 0.)

        # Transition to terminal state
        state = env.step(state, jnp.array(1))
        self.assertSequenceEqual(state.observation, [3])
        self.assertTrue(state.done)
        self.assertEqual(state.reward, 1.)

        # Actions in terminal state are no-ops with no reward.
        state = env.step(state, jnp.array(0))
        self.assertSequenceEqual(state.observation, [3])
        self.assertTrue(state.done)
        self.assertEqual(state.reward, 0.)

    @parameterized.named_parameters([('_position_features', False),
                                     ('_one_hot_features', True)])
    def test_multi_dim_transitions(self, one_hot_features):
        env = trivial_environment.DiscreteTargetEnvironment(
            size=2, dim=2, one_hot_features=one_hot_features)
        as_position = lambda obs: env._from_features(
            obs) if one_hot_features else obs
        self.assertEqual(env.action_space.num_actions, 4)

        state = env.reset(seed=test_util.test_seed())
        self.assertEqual(state.observation.shape[0], env.observation_size)
        self.assertSequenceEqual(as_position(state.observation), (0, 0))
        for (action, expected_obs) in ((1, (0, -1)), (1, (0, -2)), (1, (0, -2)),
                                       (0, (-1, -2)), (2, (0, -2)),
                                       (2, (1, -2)), (2, (2, -2)), (3, (2,
                                                                        -1))):
            state = env.step(state, jnp.asarray(action))
            self.assertSequenceEqual(as_position(state.observation),
                                     expected_obs)
            self.assertFalse(state.done)
            self.assertEqual(state.reward, 0.)

        # Transition to terminal state
        state = env.step(state, jnp.asarray(3))
        self.assertSequenceEqual(as_position(state.observation), (2, 0))
        self.assertTrue(state.done)
        self.assertEqual(state.reward, 1.)

        # Actions in terminal state are no-ops with no reward.
        state = env.step(state, jnp.asarray(0))
        self.assertSequenceEqual(as_position(state.observation), (2, 0))
        self.assertTrue(state.done)
        self.assertEqual(state.reward, 0.)


if __name__ == '__main__':
    absltest.main()