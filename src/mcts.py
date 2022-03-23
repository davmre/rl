import collections
import copy
from dataclasses import dataclass

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
    return environment.action_space.sample()

def do_rollout(environment, rollout_policy):
    total_reward = 0.0
    done = False
    discount = 1.0
    while not done:
        action = rollout_policy(environment)
        observation, reward, done, info = environment.step(action)
        total_reward += reward * discount
        discount *= environment.discount_factor
    return total_reward

def descend(tree, environment, tree_policy):
    action_stack = []
    reward_stack = []
    
    # Descend through the existing tree until we reach a leaf.
    still_descending = True
    while still_descending:
        action = tree_policy(tree, environment)
        observation, reward, done, info = environment.step(action)
        action_stack.append(action)
        reward_stack.append(reward)
        new_state = environment.get_state_key()
        if new_state not in tree.children:
            tree.children[new_state] = MCTSNode(parent=tree, children={})
            still_descending = False
        if done:
            still_descending = False
        tree = tree.children[new_state]
    return tree, action_stack, reward_stack, done

def backpropagate(tree_leaf, value, action_stack, reward_stack, tree_policy, discount_factor):
    tree = tree_leaf.parent
    while tree is not None:
        action, reward = action_stack.pop(), reward_stack.pop()
        value = value * discount_factor + reward
        tree_policy.update_statistics(tree, action, value)
        tree = tree.parent

def update_tree(tree, environment, tree_policy, rollout_policy):
    tree_leaf, action_stack, reward_stack, done = descend(tree, environment, tree_policy=tree_policy)
    if done:
        remaining_value = 0.
    else:
        remaining_value = do_rollout(environment, rollout_policy=rollout_policy)
    # print("descended via {} with rewards {}, remaining value {}".format(action_stack, reward_stack, remaining_value))
    backpropagate(tree_leaf,
                  value=remaining_value,
                  action_stack=action_stack,
                  reward_stack=reward_stack,
                  tree_policy=tree_policy,
                  discount_factor=environment.discount_factor)

def select_action(tree,
                  env,
                  tree_policy,
                  rollout_policy=random_rollout_policy,
                  num_rollouts=100):
    current_state = env.get_state()
    simulation_env = env.copy()
    for _ in range(num_rollouts):
        simulation_env.set_state(current_state)
        update_tree(tree, simulation_env, tree_policy=tree_policy, rollout_policy=rollout_policy)
        
    action = tree_policy.optimal_action(tree)
    observation, reward, done, info = env.step(action)
    new_tree = tree.children[env.get_state_key()]
    new_tree.parent = None  # Garbage collect the old tree.
    return new_tree, action, reward, done, observation
