import numpy as np

class UCTreePolicy(object):
    def __init__(self, C=1.4):
        self.C = C

    def __call__(self, tree, environment):
        random_action = environment.action_space.sample()
        if random_action not in tree.action_statistics:
            return random_action

        N_s = tree.node_statistics
        uct_scores = {a: q + self.C * np.sqrt(np.log(N_s) / n_a)
                      for a, (q, n_a) in tree.action_statistics.items()}
        return max(uct_scores, key=uct_scores.get)
    
    def optimal_action(self, tree):
        qvalues = {a: q for a, (q, n_a) in tree.action_statistics.items()}
        return max(qvalues, key=qvalues.get)
    
    def update_statistics(self, tree, action, value):
        # Update node visit count.
        if tree.node_statistics is None:
            tree.node_statistics = 0
        tree.node_statistics += 1

        # Update action value and count.
        if action in tree.action_statistics:
            expected_value, total_count = tree.action_statistics[action]
        else:
            expected_value, total_count = 0.0, 0
        tree.action_statistics[action] = (
            (expected_value * total_count + value) / (total_count + 1),
            total_count + 1)

class NormalThompsonTreePolicy(object):
    
    def __init__(self):
        pass
    
    def __call__(self, tree, environment):
        random_action = environment.action_space.sample()
        if random_action not in tree.action_statistics:
            return random_action
        
        # TODO: Implement Thompson sampling.
        
    def update_statistics(self, tree, action, value):
        # Update node visit count.
        if tree.node_statistics is None:
            tree.node_statistics = 0.0, 0.0, 0
        values, values_squared, count = tree.node_statistics
        tree.node_statistics = (values + value, values_squared + value**2, count + 1)

        # Update action value and count.
        if action not in tree.action_statistics:
            tree.action_statistics[action] = 0.0, 0.0, 0
        qvalues, qvalues_squared, count = tree.action_statistics[action]
        tree.action_statistics[action] = (qvalues + value, qvalues_squared + value**2, count + 1)