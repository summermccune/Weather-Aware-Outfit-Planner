"""
Tabular Policy Iteration algorithm.
"""

from typing import Dict, Any


def policy_iteration(mdp, gamma: float = 0.95, theta: float = 1e-4, max_eval_iterations: int = 1000):
    """
    Generic policy iteration

    mdp must provide:
    - mdp.states: iterable of states
    - mdp.get_actions(state)
    - mdp.transition_prob(state, action, next_state)
    - mdp.reward(state, action)
    """
    # Initialize an arbitrary policy
    policy: Dict[Any, Any] = {}
    for s in mdp.states:
        actions = mdp.get_actions(s)
        policy[s] = actions[0] if actions else None

    # Initialize value function
    V: Dict[Any, float] = {s: 0.0 for s in mdp.states}

    is_stable = False
    while not is_stable:
        # Policy evaluation
        for _ in range(max_eval_iterations):
            delta = 0.0
            new_V = V.copy()
            for s in mdp.states:
                a = policy[s]
                if a is None:
                    new_V[s] = 0.0
                    continue

                v_s = 0.0
                for s_prime in mdp.states:
                    p = mdp.transition_prob(s, a, s_prime)
                    if p > 0.0:
                        r = mdp.reward(s, a)
                        v_s += p * (r + gamma * V[s_prime])
                new_V[s] = v_s
                delta = max(delta, abs(new_V[s] - V[s]))
            V = new_V
            if delta < theta:
                break

        # Policy improvement
        is_stable = True
        for s in mdp.states:
            old_action = policy[s]
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
            if best_a != old_action:
                is_stable = False

    return V, policy
