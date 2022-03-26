import copy
from pydoc import resolve

import numpy as np


import tree

def resolve_chance_node(state, rng=np.random):
    while state.is_chance_node():
        actions, probs = state.chance_outcomes()
        state.apply_action(rng.choice(actions, p=probs))

class Environment(object):
    
    def step(self, action):
        """
        Returns:
          observation
          rewards: vector of length `num_players`
          done
          info
        """
        raise NotImplementedError
    
    def reset(self, seed=None):
        raise NotImplementedError
    
    def get_state(self):
        raise NotImplementedError
    
    def get_state_key(self):
        state = self.get_state()
        try:
            state = tuple(tree.flatten(state))
        except TypeError:
            pass
        return hash(state)
    
    def set_state(self, state):
        raise NotImplementedError
    
    @property
    def discount_factor(self):
        return self._discount_factor
    
    def copy(self):
        return copy.deepcopy(self)
    
    @property
    def num_players(self):
        return 1
    
    def current_player(self):
        return 0
    
    @property
    def num_actions(self):
        raise NotImplementedError
    
    def legal_actions(self):
        raise NotImplementedError
    
    def sample_action(self, rng=np.random):
        return rng.choice(self.legal_actions())
    
    
class SpielGameEnvironment(Environment):
    """
    How to handle chance nodes?
    If we get to one, just advanbce?
    """
    
    def __init__(self, game):
        self._game = game
        self._discount_factor = 1.
        self._state = None
        self._seed = None
        self._rng = None
        
    @property
    def num_players(self):
        return self._game.num_players()
        
    def reset(self, seed=None):
        self._seed = seed
        self._rng = np.random.RandomState(self._seed)
        self._state = self._game.new_initial_state()
        resolve_chance_node(self._state, rng=self._rng)
        
    def step(self, action):
        self._state.apply_action(action)
        resolve_chance_node(self._state, rng=self._rng)
        if self._state.is_terminal():
            # TODO: do observations always fail at terminal states?
            observation = None
        else:
            observation = self._state.observation_string()
        return (observation,
                self._state.rewards(),
                self._state.is_terminal(),
                {})

    def get_state(self):
        return self._state.clone()
    
    def get_state_key(self):
        return hash(self._state.serialize())
    
    def set_state(self, state):
        self._state = state.clone()
        
    def current_player(self):
        player = self._state.current_player()
        if player < 0:
            raise ValueError("Negative player index {} at state {}".format(player, self._state))
        return player
    
    def copy(self):
        # Just reuse the game and state objects since they are immutable.
        copied_env = SpielGameEnvironment(self._game)
        copied_env._state = self._state.clone()
        copied_env._seed = self._seed
        if self._rng:
            copied_env._rng = copy.deepcopy(self._rng)
        return copied_env
    
    @property
    def num_actions(self):
        return self._game.num_distinct_actions
    
    def legal_actions(self):
        return self._state.legal_actions()

"""
How does mcts represent multiplayer values?
Generally I guess reward is a vector of length num_players
"""

    
class DeterministicGymEnvironment(Environment):
    
    def __init__(self, gym_env, discount_factor=1.0):
        self._gym_env = gym_env
        self._discount_factor = discount_factor
        self._num_players = 1
        
    def step(self, action):
        self._action_sequence.append(action)
        observation, reward, done, info = self._gym_env.step(action=action)
        return observation, [reward], done, info
    
    def reset(self, seed=None):
        self._action_sequence = []
        self._init_seed = seed
        observation = self._gym_env.reset(seed=seed)
        return observation
    
    def get_state(self):
        return self._action_sequence, self._init_seed
    
    def set_state(self, state):
        action_sequence, init_seed = state
        observation = self.reset(seed=init_seed)
        for action in action_sequence:
            self.step(action)
    
    @property
    def num_actions(self):
        return self._gym_env.action_space.n
    
    def legal_actions(self):
        return range(self.num_actions)