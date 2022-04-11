import functools
import gym
import pyspiel

from daves_rl_lib import environment
from daves_rl_lib.mcts import mcts
from daves_rl_lib.mcts import tree_policy

from absl.testing import absltest
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
            tree, action, values, rewards, done, observation = mcts.select_action(tree, env, tree_policy=tree_policy.UCTreePolicy(C=1.4 * 5.0))
            print("took action {} with expected value {} received rewards {}".format(action, values, rewards))
            n += 1
        qs = {a: q for a, (q, n_a) in tree.action_statistics.items()}
        value = max(qs.values())
        self.assertGreater(value, 4.8)  # 5.0 is the best possible future reward with a discount factor of 0.8.
        
    def test_tic_tac_toe(self):
        seed = 0
        env = environment.SpielGameEnvironment(pyspiel.load_game("tic_tac_toe"))
        env.reset(seed=seed)
        tree = mcts.initialize_tree(env)

        done = False
        while not done:
            current_player = env.current_player()
            tree, action, values, rewards, done, observation = mcts.select_action(tree, env, tree_policy=tree_policy.UCTreePolicy(C=1.4))
            print("player {} took action {} with expected value {} received rewards {}".format(current_player, action, values, rewards))
            print(env.get_state())
        self.assertEqual(rewards[0], 0.0)

        
if __name__ == '__main__':
    absltest.main()