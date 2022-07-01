import dataclasses

from absl.testing import absltest
from absl.testing import parameterized

import jax
from jax import numpy as jnp
import numpy as np

from daves_rl_lib.algorithms import agent_lib
from daves_rl_lib.internal import test_util


class AgentLibTests(test_util.TestCase):

    def test_episode_reward_to_go(self):
        rewards = [3., 7., -2., 1., 5.]
        done = [True, False, False, True, False]
        rtg = agent_lib.episode_reward_to_go(jnp.array(rewards),
                                             jnp.array(done),
                                             discount_factor=0.5)
        expected = jnp.array([3., 7 - 0.5 * 2. + 0.25, -2. + 0.5, 1., 5.])
        self.assertTrue(jnp.all(jnp.equal(rtg, expected)))