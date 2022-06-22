import numpy as np


class UCTreePolicy(object):

    def __init__(self, C=1.4):
        self.C = C

    def __call__(self, tree, environment):
        if not tree.action_statistics:
            tree.action_statistics = {
                a: (np.zeros(environment.num_players), 0)
                for a in environment.legal_actions()
            }

        N_s = tree.node_statistics or 1
        current_player = environment.current_player()
        uct_scores = {
            a: q[current_player] + self.C * np.sqrt(np.log(N_s) / n_a)
            for a, (q, n_a) in tree.action_statistics.items()
        }
        return max(uct_scores, key=uct_scores.get)

    def optimal_action(self, tree, environment):
        current_player = environment.current_player()
        qvalues = {
            a: q[current_player]
            for a, (q, n_a) in tree.action_statistics.items()
        }
        best_action = max(qvalues, key=qvalues.get)
        return best_action, tree.action_statistics[best_action][0]

    def update_statistics(self, tree, action, values):
        # Update node visit count.
        if tree.node_statistics is None:
            tree.node_statistics = 0
        tree.node_statistics += 1

        # Update action value and count.
        if action in tree.action_statistics:
            expected_values, total_count = tree.action_statistics[action]
        else:
            expected_values, total_count = np.zeros_like(values), 0
        tree.action_statistics[action] = (
            (expected_values * total_count + values) / (total_count + 1),
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
        tree.node_statistics = (values + value, values_squared + value**2,
                                count + 1)

        # Update action value and count.
        if action not in tree.action_statistics:
            tree.action_statistics[action] = 0.0, 0.0, 0
        qvalues, qvalues_squared, count = tree.action_statistics[action]
        tree.action_statistics[action] = (qvalues + value,
                                          qvalues_squared + value**2, count + 1)
