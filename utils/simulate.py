"""
Simulation utilities for evaluating policies.
"""

from typing import Dict, Any
import numpy as np


def simulate_simple_outfit_mdp(mdp, policy: Dict[Any, Any], num_weeks: int = 100, seed: int = 0):
    """
    Simulate the simple OutfitMDP with a given policy

    Returns a dict with summary statistics:
    - avg_weekly_reward
    - total_mismatches
    - total_repeats
    """
    rng = np.random.default_rng(seed)
    weather_states = mdp.weather_states
    wt = np.array(mdp.weather_transition_matrix)
    num_weeks = int(num_weeks)

    total_reward = 0.0
    total_mismatches = 0
    total_repeats = 0

    for _ in range(num_weeks):
        # Start of week: random initial weather and previous outfit
        current_weather = rng.choice(weather_states)
        prev_outfit = rng.choice(mdp.outfit_actions)

        for day in mdp.days:
            state = (day, current_weather, prev_outfit)
            action = policy.get(state)
            if action is None:
                # Fallback: if policy missing, just choose first action
                action = mdp.outfit_actions[0]

            r = mdp.reward(state, action)
            total_reward += r

            # Track mismatch / repeats
            if not mdp.weather_matches(current_weather, action):
                total_mismatches += 1
            if action == prev_outfit:
                total_repeats += 1

            # Sample next weather
            w_idx = mdp.weather_index[current_weather]
            probs = wt[w_idx]
            current_weather = rng.choice(weather_states, p=probs)

            # Update previous outfit
            prev_outfit = action

    avg_weekly_reward = total_reward / num_weeks
    return {
        "avg_weekly_reward": avg_weekly_reward,
        "total_mismatches": total_mismatches,
        "total_repeats": total_repeats,
    }


def simulate_wardrobe_mdp(mdp, policy: Dict[Any, Any], num_weeks: int = 100, seed: int = 0):
    """
    Simulate the WardrobeMDP with no replacement

    Returns a dict with summary statistics:
    - avg_weekly_reward
    - total_mismatches
    - avg_days_with_outfit (how many days we actually had outfits to wear)
    """
    rng = np.random.default_rng(seed)
    weather_states = mdp.weather_states
    wt = np.array(mdp.weather_transition_matrix)
    num_weeks = int(num_weeks)

    total_reward = 0.0
    total_mismatches = 0
    total_days_with_outfit = 0

    for _ in range(num_weeks):
        current_weather = rng.choice(weather_states)
        used_mask = 0
        days_with_outfit = 0

        for day in mdp.days:
            state = (day, current_weather, used_mask)
            actions = mdp.get_actions(state)
            if not actions:
                # No outfits left to wear
                continue

            days_with_outfit += 1

            action = policy.get(state)
            if (action is None) or (action not in actions):
                # Fallback: choose first available action
                action = actions[0]

            r = mdp.reward(state, action)
            total_reward += r

            # Track mismatch
            if not mdp.weather_matches(current_weather, mdp.outfits[action]):
                total_mismatches += 1

            # Update used_mask
            used_mask = used_mask | (1 << action)

            # Sample next weather
            w_idx = mdp.weather_index[current_weather]
            probs = wt[w_idx]
            current_weather = rng.choice(weather_states, p=probs)

        total_days_with_outfit += days_with_outfit

    avg_weekly_reward = total_reward / num_weeks
    avg_days_with_outfit = total_days_with_outfit / num_weeks

    return {
        "avg_weekly_reward": avg_weekly_reward,
        "total_mismatches": total_mismatches,
        "avg_days_with_outfit": avg_days_with_outfit,
    }

def sample_week_simple(mdp, policy, seed: int = 0):
    """
    A specific human-readable example 7-day schedule with:
        Day
        Weather
        Outfit chosen
        Reward
    """
    rng = np.random.default_rng(seed)
    weather_states = mdp.weather_states
    wt = np.array(mdp.weather_transition_matrix)

    # Random initial weather + previous outfit
    current_weather = rng.choice(weather_states)
    prev_outfit = rng.choice(mdp.outfit_actions)

    week_log = []

    for day in mdp.days:
        state = (day, current_weather, prev_outfit)
        action = policy.get(state)
        if action is None:
            action = mdp.outfit_actions[0]

        r = mdp.reward(state, action)

        week_log.append({
            "day": day,
            "weather": current_weather,
            "prev_outfit": prev_outfit,
            "chosen_outfit": action,
            "reward": r,
        })

        # Transition to next day
        w_idx = mdp.weather_index[current_weather]
        probs = wt[w_idx]
        current_weather = rng.choice(weather_states, p=probs)
        prev_outfit = action

    return week_log