"""
Simple MDP where outfits are just categories (Warm, Light, Rain-Ready, etc.)
State: (day, weather, previous_outfit_category)
Action: outfit category (string)
"""

from typing import List, Tuple, Dict, Any

class OutfitMDP:
    def __init__(
        self,
        weather_states: List[str],
        outfit_actions: List[str],
        weather_transition_matrix,
        reward_weights: Dict[str, float],
        horizon_days: int = 7,
    ) -> None:
        self.weather_states = weather_states
        self.outfit_actions = outfit_actions
        self.weather_transition_matrix = weather_transition_matrix
        self.reward_weights = reward_weights
        self.horizon_days = horizon_days

        self.days = list(range(horizon_days))
        self.weather_index = {w: i for i, w in enumerate(self.weather_states)}

        # Build full state space
        self.states: List[Tuple[int, str, str]] = []
        for d in self.days:
            for w in self.weather_states:
                for prev in self.outfit_actions:
                    self.states.append((d, w, prev))

    # MDP interface methods

    def get_actions(self, state: Tuple[int, str, str]):
        """
        Return all possible outfit actions for this state
        """
        # In this simple model, all outfit categories are always available
        return list(self.outfit_actions)

    def transition_prob(
        self,
        state: Tuple[int, str, str],
        action: str,
        next_state: Tuple[int, str, str],
    ) -> float:
        """
        P(s' | s, a)

        state = (day, weather, prev_outfit)
        next_state = (day', weather', prev_outfit')
        """
        day, weather, prev_outfit = state
        next_day, next_weather, next_prev_outfit = next_state

        # Day must advance by 1, capped at last day (absorbing-ish)
        expected_next_day = min(day + 1, self.days[-1])
        if next_day != expected_next_day:
            return 0.0

        # Previous outfit in next_state should be the outfit we just wore.
        if next_prev_outfit != action:
            return 0.0

        # Weather transition according to Markov chain
        try:
            w_idx = self.weather_index[weather]
            w_next_idx = self.weather_index[next_weather]
        except KeyError:
            return 0.0

        return float(self.weather_transition_matrix[w_idx][w_next_idx])

    def reward(self, state: Tuple[int, str, str], action: str) -> float:
        """
        Reward combines:
        - Comfort for matching outfit to weather
        - Variety bonus
        - Penalties for mismatch or repeating outfits
        """
        _, weather, prev_outfit = state
        rw = self.reward_weights

        # Weather comfort term
        if self.weather_matches(weather, action):
            comfort_term = rw.get("comfort", 0.0)
            mismatch_term = 0.0
        else:
            comfort_term = 0.0
            mismatch_term = -rw.get("mismatch_penalty", 0.0)

        # Variety term
        if action != prev_outfit:
            variety_term = rw.get("variety", 0.0)
            repeat_term = 0.0
        else:
            variety_term = 0.0
            repeat_term = -rw.get("repeat_penalty", 0.0)

        return comfort_term + variety_term + mismatch_term + repeat_term

    def weather_matches(self, weather: str, outfit_category: str) -> bool:
        rules = {
            "Cold": {"Warm", "Formal"},
            "Mild": {"Casual", "Light", "Formal"},
            "Hot": {"Light", "Casual"},
            "Rainy": {"Rain-Ready"},
        }
        allowed = rules.get(weather, set())
        return outfit_category in allowed