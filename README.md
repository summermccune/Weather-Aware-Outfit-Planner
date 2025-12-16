# Weather-Aware Outfit Planner (CS 667 Final Project)

This project models weekly outfit planning under uncertain weather as a Markov Decision Process (MDP).
We implement two formulations:
1) **Simple Category-Based MDP** (outfit categories, reuse allowed)
2) **Finite Wardrobe MDP (No Replacement)** (fixed set of outfits, cannot reuse within a week)

We solve both models using **Value Iteration** and **Policy Iteration**, then evaluate policies over 500 simulated weeks and generate plots.

---

## Requirements
- Python 3.9+ (3.10+ recommended)
- Dependencies listed in `requirements.txt`

Install:
```bash
pip install -r requirements.txt
```

---

## How to Run
```
python main.py
```
The script runs both the Simple MDP and Finite-Wardrobe MDP,
prints evaluation metrics, and saves result plots to the `experiments_output/` directory.

---

## Expected Output
- Printed evaluation results for both MDP models
- Example weeks for each model
- Saved plots in `experiments_output/`

---

## Project Structure
- main.py: runs experiments and visualization
- models/: MDP definitions
- algorithms/: Value Iteration and Policy Iteration
- utils/: simulation and plotting utilities
