"""
Visualization utilities for MDP results.
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, Any, List, Tuple
import os


def create_reward_comparison_graph(
    results_vi: Dict[str, Any],
    results_pi: Dict[str, Any],
    model_name: str,
    output_dir: str = "experiments_output",
) -> None:
    """
    Create a bar chart comparing Value Iteration vs Policy Iteration rewards.
    """
    os.makedirs(output_dir, exist_ok=True)

    metrics = ["avg_weekly_reward"]
    vi_values = [results_vi.get(m, 0) for m in metrics]
    pi_values = [results_pi.get(m, 0) for m in metrics]

    x = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    bars1 = ax.bar(x - width / 2, vi_values, width, label="Value Iteration", color="#2E86AB")
    bars2 = ax.bar(x + width / 2, pi_values, width, label="Policy Iteration", color="#A23B72")

    ax.set_ylabel("Average Weekly Reward", fontsize=12, fontweight="bold")
    ax.set_title(f"{model_name} - Reward Comparison", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height,
                f"{height:.2f}",
                ha="center",
                va="bottom",
                fontsize=10,
            )

    plt.tight_layout()
    filename = f"{output_dir}/{model_name.replace(' ', '_')}_reward_comparison.png"
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    print(f"[OK] Saved: {filename}")
    plt.close()


def create_mismatches_graph(
    results_vi: Dict[str, Any],
    results_pi: Dict[str, Any],
    model_name: str,
    output_dir: str = "experiments_output",
) -> None:
    """
    Create a bar chart comparing outfit mismatches between VI and PI.
    """
    os.makedirs(output_dir, exist_ok=True)

    if "total_mismatches" not in results_vi:
        print(f"[SKIP] Skipping mismatch graph for {model_name} - no mismatch data")
        return

    metrics = ["total_mismatches"]
    vi_values = [results_vi.get(m, 0) for m in metrics]
    pi_values = [results_pi.get(m, 0) for m in metrics]

    x = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    bars1 = ax.bar(x - width / 2, vi_values, width, label="Value Iteration", color="#F18F01")
    bars2 = ax.bar(x + width / 2, pi_values, width, label="Policy Iteration", color="#C73E1D")

    ax.set_ylabel("Total Mismatches (across 500 weeks)", fontsize=12, fontweight="bold")
    ax.set_title(f"{model_name} - Outfit Mismatches", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height,
                f"{int(height)}",
                ha="center",
                va="bottom",
                fontsize=10,
            )

    plt.tight_layout()
    filename = f"{output_dir}/{model_name.replace(' ', '_')}_mismatches.png"
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    print(f"[OK] Saved: {filename}")
    plt.close()


def create_outfit_usage_graph(
    mdp,
    policy: Dict[Any, Any],
    num_simulations: int = 500,
    model_name: str = "Wardrobe MDP",
    output_dir: str = "experiments_output",
    seed: int = 0,
) -> None:
    """
    Create a bar chart showing outfit usage frequency across simulations.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Only applicable to wardrobe MDP
    if not hasattr(mdp, "outfits"):
        print(f"[SKIP] Skipping outfit usage graph - model doesn't have outfit data")
        return

    outfit_names = [outfit["name"] for outfit in mdp.outfits]
    outfit_counts = {name: 0 for name in outfit_names}

    rng = np.random.default_rng(seed)
    weather_states = mdp.weather_states
    wt = np.array(mdp.weather_transition_matrix)

    for _ in range(num_simulations):
        current_weather = rng.choice(weather_states)
        used_mask = 0

        for day in mdp.days:
            state = (day, current_weather, used_mask)
            actions = mdp.get_actions(state)
            if not actions:
                continue

            action = policy.get(state)
            if (action is None) or (action not in actions):
                action = actions[0]

            outfit_counts[outfit_names[action]] += 1
            used_mask = used_mask | (1 << action)

            w_idx = mdp.weather_index[current_weather]
            probs = wt[w_idx]
            current_weather = rng.choice(weather_states, p=probs)

    fig, ax = plt.subplots(figsize=(12, 6))
    outfits = list(outfit_counts.keys())
    counts = list(outfit_counts.values())

    colors = plt.cm.Set3(np.linspace(0, 1, len(outfits)))
    bars = ax.bar(outfits, counts, color=colors, edgecolor="black", linewidth=1.5)

    ax.set_ylabel("Times Used", fontsize=12, fontweight="bold")
    ax.set_title(f"{model_name} - Outfit Usage Frequency (500 simulations)", fontsize=14, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)

    # Rotate x-axis labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f"{int(height)}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.tight_layout()
    filename = f"{output_dir}/{model_name.replace(' ', '_')}_outfit_usage.png"
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    print(f"[OK] Saved: {filename}")
    plt.close()


def create_repeats_graph(
    results_vi: Dict[str, Any],
    results_pi: Dict[str, Any],
    model_name: str,
    output_dir: str = "experiments_output",
) -> None:
    """
    Create a bar chart comparing outfit repeats between VI and PI.
    """
    os.makedirs(output_dir, exist_ok=True)

    if "total_repeats" not in results_vi:
        print(f"[SKIP] Skipping repeats graph for {model_name} - no repeats data")
        return

    metrics = ["total_repeats"]
    vi_values = [results_vi.get(m, 0) for m in metrics]
    pi_values = [results_pi.get(m, 0) for m in metrics]

    x = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 5))
    bars1 = ax.bar(x - width / 2, vi_values, width, label="Value Iteration", color="#06A77D")
    bars2 = ax.bar(x + width / 2, pi_values, width, label="Policy Iteration", color="#004E89")

    ax.set_ylabel("Total Outfit Repeats (across 500 weeks)", fontsize=12, fontweight="bold")
    ax.set_title(f"{model_name} - Outfit Repetition", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                height,
                f"{int(height)}",
                ha="center",
                va="bottom",
                fontsize=10,
            )

    plt.tight_layout()
    filename = f"{output_dir}/{model_name.replace(' ', '_')}_repeats.png"
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    print(f"✓ Saved: {filename}")
    plt.close()


def create_mdp_comparison_graph(
    results_simple_vi: Dict[str, Any],
    results_wardrobe_vi: Dict[str, Any],
    output_dir: str = "experiments_output",
) -> None:
    """
    Create a comparison graph showing Simple vs Wardrobe MDP performance.
    """
    os.makedirs(output_dir, exist_ok=True)

    models = ["Simple MDP", "Wardrobe MDP"]
    rewards = [
        results_simple_vi.get("avg_weekly_reward", 0),
        results_wardrobe_vi.get("avg_weekly_reward", 0),
    ]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(models, rewards, color=["#2E86AB", "#A23B72"], width=0.6, edgecolor="black", linewidth=2)

    ax.set_ylabel("Average Weekly Reward", fontsize=12, fontweight="bold")
    ax.set_title("Model Comparison - Average Weekly Reward", fontsize=14, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f"{height:.2f}",
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold",
        )

    plt.tight_layout()
    filename = f"{output_dir}/Model_Comparison_Rewards.png"
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    print(f"[OK] Saved: {filename}")
    plt.close()


def create_mismatch_comparison_graph(
    results_simple_vi: Dict[str, Any],
    results_wardrobe_vi: Dict[str, Any],
    output_dir: str = "experiments_output",
) -> None:
    """
    Create a comparison graph showing mismatch rates between MDPs.
    """
    os.makedirs(output_dir, exist_ok=True)

    models = ["Simple MDP", "Wardrobe MDP"]
    mismatches = [
        results_simple_vi.get("total_mismatches", 0),
        results_wardrobe_vi.get("total_mismatches", 0),
    ]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(
        models,
        mismatches,
        color=["#F18F01", "#C73E1D"],
        width=0.6,
        edgecolor="black",
        linewidth=2,
    )

    ax.set_ylabel("Total Mismatches (500 weeks)", fontsize=12, fontweight="bold")
    ax.set_title("Model Comparison - Outfit Mismatches", fontsize=14, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height,
            f"{int(height)}",
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold",
        )

    plt.tight_layout()
    filename = f"{output_dir}/Model_Comparison_Mismatches.png"
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    print(f"[OK] Saved: {filename}")
    plt.close()


def create_summary_table_graph(
    results_simple_vi: Dict[str, Any],
    results_wardrobe_vi: Dict[str, Any],
    output_dir: str = "experiments_output",
) -> None:
    """
    Create a summary table showing key metrics side-by-side.
    """
    os.makedirs(output_dir, exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.axis("tight")
    ax.axis("off")

    # Prepare data
    data = [
        ["Metric", "Simple MDP", "Wardrobe MDP"],
        [
            "Avg Weekly Reward",
            f"{results_simple_vi.get('avg_weekly_reward', 0):.2f}",
            f"{results_wardrobe_vi.get('avg_weekly_reward', 0):.2f}",
        ],
        [
            "Total Mismatches",
            f"{int(results_simple_vi.get('total_mismatches', 0))}",
            f"{int(results_wardrobe_vi.get('total_mismatches', 0))}",
        ],
        [
            "Total Repeats",
            f"{int(results_simple_vi.get('total_repeats', 0))}",
            "N/A (Limited Wardrobe)",
        ],
    ]

    table = ax.table(
        cellText=data,
        cellLoc="center",
        loc="center",
        colWidths=[0.35, 0.3, 0.3],
    )

    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.5)

    # Style header row
    for i in range(3):
        table[(0, i)].set_facecolor("#2E86AB")
        table[(0, i)].set_text_props(weight="bold", color="white")

    # Alternate row colors
    for i in range(1, len(data)):
        for j in range(3):
            if i % 2 == 0:
                table[(i, j)].set_facecolor("#E8F4F8")
            else:
                table[(i, j)].set_facecolor("#FFFFFF")

    plt.title("Performance Summary (500 simulations)", fontsize=14, fontweight="bold", pad=20)
    plt.tight_layout()
    filename = f"{output_dir}/Summary_Table.png"
    plt.savefig(filename, dpi=300, bbox_inches="tight")
    print(f"[OK] Saved: {filename}")
    plt.close()


def create_all_graphs(
    mdp_simple,
    results_simple_vi: Dict[str, Any],
    results_simple_pi: Dict[str, Any],
    policy_simple_vi: Dict[Any, Any],
    mdp_wardrobe,
    results_wardrobe_vi: Dict[str, Any],
    results_wardrobe_pi: Dict[str, Any],
    policy_wardrobe_vi: Dict[Any, Any],
    output_dir: str = "experiments_output",
) -> None:
    """
    Create all visualization graphs.
    """
    print("\n[INFO] Generating visualizations...\n")

    # Model comparison graphs
    create_mdp_comparison_graph(
        results_simple_vi,
        results_wardrobe_vi,
        output_dir,
    )
    create_mismatch_comparison_graph(
        results_simple_vi,
        results_wardrobe_vi,
        output_dir,
    )
    create_summary_table_graph(
        results_simple_vi,
        results_wardrobe_vi,
        output_dir,
    )

    # Simple MDP graphs
    create_repeats_graph(
        results_simple_vi,
        results_simple_pi,
        "Simple MDP",
        output_dir,
    )

    # Wardrobe MDP graphs
    create_outfit_usage_graph(
        mdp_wardrobe,
        policy_wardrobe_vi,
        num_simulations=500,
        model_name="Wardrobe MDP",
        output_dir=output_dir,
    )

    print(f"\n[SUCCESS] All graphs saved to: {output_dir}/")
