"""
Tabular Value Iteration algorithm.
"""

from typing import Dict, Any


def value_iteration(mdp, gamma: float = 0.95, theta: float = 1e-4, max_iterations: int = 1000):
    """
    Generic value iteration

    mdp must provide:
    - mdp.states: iterable of states
    - mdp.get_actions(state)
    - mdp.transition_prob(state, action, next_state)
    - mdp.reward(state, action)
    """
    # Initialize values to zero
    V: Dict[Any, float] = {s: 0.0 for s in mdp.states}

    for _ in range(max_iterations):
        delta = 0.0
        new_V = V.copy()

        for s in mdp.states:
            actions = mdp.get_actions(s)
            if not actions:
                # No available actions (e.g. all outfits used)
                new_V[s] = 0.0
                continue

            best_value = float("-inf")
            for a in actions:
                q_sa = 0.0
                for s_prime in mdp.states:
                    p = mdp.transition_prob(s, a, s_prime)
                    if p > 0.0:
                        r = mdp.reward(s, a)
                        q_sa += p * (r + gamma * V[s_prime])
                if q_sa > best_value:
                    best_value = q_sa

            new_V[s] = best_value
            delta = max(delta, abs(new_V[s] - V[s]))

        V = new_V
        if delta < theta:
            break

    # Derive greedy policy
    policy: Dict[Any, Any] = {}
    for s in mdp.states:
        actions = mdp.get_actions(s)
        if not actions:
            policy[s] = None
            continue

        best_a = None
        best_value = float("-inf")
        for a in actions:
            q_sa = 0.0
            for s_prime in mdp.states:
                p = mdp.transition_prob(s, a, s_prime)
                if p > 0.0:
                    r = mdp.reward(s, a)
                    q_sa += p * (r + gamma * V[s_prime])
            if q_sa > best_value:
                best_value = q_sa
                best_a = a
        policy[s] = best_a

    return V, policy
