import collections
import copy
from dataclasses import dataclass
from os import environ

from typing import Any, Dict, Optional, Type

from jax import numpy as np

class MCTSNode(object):

    def __init__(self, parent: Any, children: Dict):
        self.parent = parent
        self.children = children
        self.node_statistics = None
        self.action_statistics = {}

def initialize_tree(environment):
    return MCTSNode(parent=None, children={})

def random_rollout_policy(environment):
    return environment.sample_action()

def do_rollout(environment, rollout_policy):
    total_rewards = np.zeros([environment.num_players])
    done = False
    discount = 1.0
    while not done:
        action = rollout_policy(environment)
        observation, rewards, done, info = environment.step(action)
        total_rewards += np.asarray(rewards) * discount
        discount *= environment.discount_factor
    return total_rewards

def descend(tree, environment, tree_policy):
    action_stack = []
    rewards_stack = []
    
    # Descend through the existing tree until we reach a leaf.
    still_descending = True
    while still_descending:
        action = tree_policy(tree, environment)
        observation, rewards, done, info = environment.step(action)
        action_stack.append(action)
        rewards_stack.append(rewards)
        new_state = environment.get_state_key()
        if new_state not in tree.children:
            tree.children[new_state] = MCTSNode(parent=tree, children={})
            still_descending = False
        if done:
            still_descending = False
        tree = tree.children[new_state]
    return tree, action_stack, rewards_stack, done

def backpropagate(tree_leaf, values, action_stack, rewards_stack, tree_policy, discount_factor):
    tree = tree_leaf.parent
    while tree is not None:
        action, rewards = action_stack.pop(), rewards_stack.pop()
        values = values * discount_factor + np.asarray(rewards)
        tree_policy.update_statistics(tree, action, values)
        tree = tree.parent

def update_tree(tree, environment, tree_policy, rollout_policy):
    tree_leaf, action_stack, rewards_stack, done = descend(tree, environment, tree_policy=tree_policy)
    if done:
        remaining_values = np.zeros([environment.num_players])
    else:
        remaining_values = do_rollout(environment, rollout_policy=rollout_policy)
    backpropagate(tree_leaf,
                  values=remaining_values,
                  action_stack=action_stack,
                  rewards_stack=rewards_stack,
                  tree_policy=tree_policy,
                  discount_factor=environment.discount_factor)

def select_action(tree,
                  env,
                  tree_policy,
                  rollout_policy=random_rollout_policy,
                  num_rollouts=100):
    current_state = env.get_state()
    simulation_env = env.copy()
    for k in range(num_rollouts):
        simulation_env.set_state(current_state)
        update_tree(tree, simulation_env, tree_policy=tree_policy, rollout_policy=rollout_policy)
        
    action, values = tree_policy.optimal_action(tree, env)
    observation, rewards, done, info = env.step(action)
    new_tree = tree.children[env.get_state_key()]
    new_tree.parent = None  # Garbage collect the old tree.
    return new_tree, action, values, rewards, done, observation

