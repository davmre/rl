import functools
import gym

import environment
import mcts
import tree_policy

import unittest

from absl.testing import parameterized

class MCTSTests(parameterized.TestCase):

    def test_cartpole(self):
        seed = 0
        env = environment.DeterministicGymEnvironment(gym.make('CartPole-v1'), discount_factor=0.8)
        env.reset(seed=seed)
        tree = mcts.initialize_tree(env)

        n = 0
        done = False
        while n < 20:
            tree, action, reward, done, observation = mcts.select_action(tree, env, tree_policy=tree_policy.UCTreePolicy(C=1.4 * 5.0))
            n += 1
        qs = {a: q for a, (q, n_a) in tree.action_statistics.items()}
        value = max(qs.values())
        self.assertGreater(value, 4.8)  # 5.0 is the best possible future reward with a discount factor of 0.8.
        
if __name__ == '__main__':
    unittest.main()