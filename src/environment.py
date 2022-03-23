import copy
import tree

class Environment(object):
    
    def step(self, action):
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
    def action_space(self):
        return self._action_space
    
    @property
    def observation_space(self):
        return self._observation_space
    
    @property
    def discount_factor(self):
        return self._discount_factor
    
class DeterministicGymEnvironment(Environment):
    
    def __init__(self, gym_env, discount_factor=1.0):
        self._gym_env = gym_env
        self._action_space = gym_env.action_space
        self._observation_space = gym_env.observation_space
        self._discount_factor = discount_factor
        
    def step(self, action):
        self._action_sequence.append(action)
        observation, reward, done, info = self._gym_env.step(action=action)
        return observation, reward, done, info
    
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
            
    def copy(self):
        env = DeterministicGymEnvironment(
            gym_env=copy.deepcopy(self._gym_env),
            discount_factor=self._discount_factor)
        env._action_sequence = self._action_sequence
        env._init_seed = self._init_seed
        return env