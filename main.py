from algorithms.value_iteration import value_iteration
from algorithms.policy_iteration import policy_iteration
from models.mdp import OutfitMDP
from models.wardrobe_mdp import WardrobeMDP, sample_week_wardrobe
from utils import config
from utils.simulate import simulate_simple_outfit_mdp, simulate_wardrobe_mdp, sample_week_simple


def run_simple_mdp():
    print("=== Running Simple Category-Based Outfit MDP ===")
    mdp = OutfitMDP(
        weather_states=config.weather_states,
        outfit_actions=config.outfit_categories,
        weather_transition_matrix=config.weather_transition_matrix,
        reward_weights=config.reward_weights_simple,
        horizon_days=7,
    )

    # Solve with Value Iteration
    V_vi, policy_vi = value_iteration(mdp)
    print("Value Iteration completed on simple MDP.")

    # Solve with Policy Iteration
    V_pi, policy_pi = policy_iteration(mdp)
    print("Policy Iteration completed on simple MDP.")

    # Evaluate VI policy
    results_vi = simulate_simple_outfit_mdp(mdp, policy_vi, num_weeks=500)
    print("\n[Simple MDP] Evaluation of Value Iteration policy:")
    for k, v in results_vi.items():
        print(f"- {k}: {v:.3f}" if isinstance(v, (float, int)) else f"- {k}: {v}")

    # Evaluate PI policy
    results_pi = simulate_simple_outfit_mdp(mdp, policy_pi, num_weeks=500)
    print("\n[Simple MDP] Evaluation of Policy Iteration policy:")
    for k, v in results_pi.items():
        print(f"- {k}: {v:.3f}" if isinstance(v, (float, int)) else f"- {k}: {v}")

    # After printing evaluation results:
    example_week = sample_week_simple(mdp, policy_vi, seed=0)
    print("\nExample week (simple MDP, VI policy):")
    for d in example_week:
        print(
            f"Day {d['day']}: weather={d['weather']}, "
            f"prev={d['prev_outfit']}, chosen={d['chosen_outfit']}, "
            f"reward={d['reward']:.1f}"
        )


def run_wardrobe_mdp():
    print("=== Running Finite Wardrobe MDP (No Replacement) ===")
    mdp = WardrobeMDP(
        weather_states=config.weather_states,
        outfits=config.outfits,
        weather_transition_matrix=config.weather_transition_matrix,
        reward_weights=config.reward_weights_wardrobe,
        horizon_days=7,
    )

    # Solve with Value Iteration
    V_vi, policy_vi = value_iteration(mdp)
    print("Value Iteration completed on wardrobe MDP.")

    # Solve with Policy Iteration
    V_pi, policy_pi = policy_iteration(mdp)
    print("Policy Iteration completed on wardrobe MDP.")

    # Evaluate VI policy
    results_vi = simulate_wardrobe_mdp(mdp, policy_vi, num_weeks=500)
    print("\n[Wardrobe MDP] Evaluation of Value Iteration policy:")
    for k, v in results_vi.items():
        print(f"- {k}: {v:.3f}" if isinstance(v, (float, int)) else f"- {k}: {v}")

    # Evaluate PI policy
    results_pi = simulate_wardrobe_mdp(mdp, policy_pi, num_weeks=500)
    print("\n[Wardrobe MDP] Evaluation of Policy Iteration policy:")
    for k, v in results_pi.items():
        print(f"- {k}: {v:.3f}" if isinstance(v, (float, int)) else f"- {k}: {v}")

    example_week = sample_week_wardrobe(mdp, policy_vi, seed=0)
    print("\nExample week (wardrobe MDP, VI policy):")
    for d in example_week:
        print(
            f"Day {d['day']}: weather={d['weather']}, "
            f"outfit={d['outfit_name']}, "
            f"reward={d['reward']:.1f}, mismatch={d['mismatch']}"
        )

if __name__ == "__main__":
    # Run both models for comparison
    RUN_SIMPLE = True
    RUN_WARDROBE = True

    if RUN_SIMPLE:
        run_simple_mdp()
        print("\n" + "=" * 60 + "\n")

    if RUN_WARDROBE:
        run_wardrobe_mdp()
