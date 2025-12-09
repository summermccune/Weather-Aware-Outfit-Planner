"""
Finite-wardrobe MDP with no outfit replacement.
State: (day, weather, used_mask)
Action: outfit index (int)
"""

from typing import List, Dict, Tuple, Any

import numpy as np


class WardrobeMDP:
    def __init__(
        self,
        weather_states: List[str],
        outfits: List[Dict[str, Any]],
        weather_transition_matrix,
        reward_weights: Dict[str, float],
        horizon_days: int = 7,
    ) -> None:
        """
        outfits: list of dicts like
        {
            "name": "Pink Raincoat Fit",
            "category": "Rain-Ready",
            "rain_ready": True,
            "warmth": "warm",
        }
        """
        self.weather_states = weather_states
        self.outfits = outfits
        self.weather_transition_matrix = weather_transition_matrix
        self.reward_weights = reward_weights
        self.horizon_days = horizon_days

        self.days = list(range(horizon_days))
        self.num_outfits = len(outfits)
        self.weather_index = {w: i for i, w in enumerate(self.weather_states)}

        # Build full state space: (day, weather, used_mask)
        self.states: List[Tuple[int, str, int]] = []
        for d in self.days:
            for w in self.weather_states:
                for used_mask in range(1 << self.num_outfits):
                    self.states.append((d, w, used_mask))

    # MDP interface methods

    def get_actions(self, state: Tuple[int, str, int]):
        """Return indices of outfits that have NOT yet been used."""
        _, _, used_mask = state
        available = []
        for i in range(self.num_outfits):
            if not (used_mask & (1 << i)):
                available.append(i)
        return available

    def transition_prob(
        self,
        state: Tuple[int, str, int],
        action: int,
        next_state: Tuple[int, str, int],
    ) -> float:
        """
        P(s' | s, a)

        state = (day, weather, used_mask)
        next_state = (day', weather', used_mask')
        """
        day, weather, used_mask = state
        next_day, next_weather, next_used_mask = next_state

        # Day must advance by 1, capped at last day
        expected_next_day = min(day + 1, self.days[-1])
        if next_day != expected_next_day:
            return 0.0

        # used_mask must update to reflect using outfit 'action'
        expected_mask = used_mask | (1 << action)
        if next_used_mask != expected_mask:
            return 0.0

        # Weather transition probability
        try:
            w_idx = self.weather_index[weather]
            w_next_idx = self.weather_index[next_weather]
        except KeyError:
            return 0.0

        return float(self.weather_transition_matrix[w_idx][w_next_idx])

    def reward(self, state: Tuple[int, str, int], action: int) -> float:
        """
        Reward based on weather–outfit compatibility.
        We assume:
        - Positive comfort reward for good match
        - Negative mismatch penalty otherwise
        """
        _, weather, _ = state
        outfit = self.outfits[action]
        rw = self.reward_weights

        if self.weather_matches(weather, outfit):
            comfort_term = rw.get("comfort", 0.0)
            mismatch_term = 0.0
        else:
            comfort_term = 0.0
            mismatch_term = -rw.get("mismatch_penalty", 0.0)

        return comfort_term + mismatch_term

    def weather_matches(self, weather: str, outfit: Dict[str, Any]) -> bool:
        """Decide if an outfit is appropriate for a given weather."""
        category = outfit.get("category")
        rain_ready = outfit.get("rain_ready", False)
        warmth = outfit.get("warmth", "")

        if weather == "Rainy":
            return rain_ready

        if weather == "Cold":
            return warmth == "warm"

        if weather == "Hot":
            return warmth == "light"

        if weather == "Mild":
            # Many things are okay on a mild day
            return category in {"Casual", "Light", "Formal", "Warm"}

        # Default: not too strict
        return True

def sample_week_wardrobe(mdp, policy, seed: int = 0):
    rng = np.random.default_rng(seed)
    weather_states = mdp.weather_states
    wt = np.array(mdp.weather_transition_matrix)

    current_weather = rng.choice(weather_states)
    used_mask = 0

    week_log = []

    for day in mdp.days:
        state = (day, current_weather, used_mask)
        actions = mdp.get_actions(state)

        if not actions:
            # No outfits left to wear
            week_log.append({
                "day": day,
                "weather": current_weather,
                "outfit_name": None,
                "reward": 0.0,
                "mismatch": None,
            })
            continue

        action = policy.get(state)
        if (action is None) or (action not in actions):
            action = actions[0]

        outfit = mdp.outfits[action]
        r = mdp.reward(state, action)
        mismatch = not mdp.weather_matches(current_weather, outfit)

        week_log.append({
            "day": day,
            "weather": current_weather,
            "outfit_name": outfit["name"],
            "reward": r,
            "mismatch": mismatch,
        })

        # Update used_mask and weather
        used_mask = used_mask | (1 << action)
        w_idx = mdp.weather_index[current_weather]
        probs = wt[w_idx]
        current_weather = rng.choice(weather_states, p=probs)

    return week_log
