# Bot Localization: Learning How Long It Takes to Find Yourself

A robot on a spaceship knows the ship's map but not its own position, and it has no working sensors. It knows only a set of places it might be. Every move it attempts shrinks that set: blocked moves leave it in place, and allowed moves shift it. This project trains a neural network to predict how many moves localization will take from any set of candidate positions. It then uses that prediction to choose better moves.

*Rutgers CS520 (Intro to AI), part 3 of 3. Part 1 is [No-one-can-hear-you](https://github.com/sridevi-mr25/No-one-can-hear-you) and part 2 is [Space-Rats](https://github.com/sridevi-mr25/Space-Rats).*

## Approach

1. **Baseline policy (π₀).** A rule-based strategy that steers the candidate positions toward corners and dead ends, where they collapse fastest.
2. **Training data.** Run π₀ across 200 random 30 × 30 mazes, starting from both full and partial (75%) candidate sets. Every step becomes a training example: the map of candidate positions, labelled with how many moves were still needed. This gave 47,570 examples.
3. **Cost model (C₀).** A small convolutional network that reads a candidate-position map and predicts the moves remaining.
4. **Improved policy (π₁).** At each step, try each of the four moves, use C₀ to estimate the cost of the resulting candidate set, and take the cheapest.
5. **Iterate (π₂).** Train a second model (C₁) on π₁'s own runs and compare all three policies.

## Findings

The one-step lookahead policies gave modest improvements over the rule-based baseline, mainly when the starting set of candidate positions was large. The full analysis is in `Project_3_CS520 BotLocalization.pdf`.

## How to run

```bash
pip install numpy matplotlib torch
```

Run the steps in this order. Each one produces a file the next one needs.

```bash
python data_generation.py   # → mixed_train_data.npz  (training examples from π₀)
python Training.py          # → localization_cnn_model.pt  (cost model C₀)
python Part_4.py            # compare π₀ vs. π₁
python PI1_model.py         # → localization_pi1_model.pt  (cost model C₁)
python Pi2.py               # compare π₀, π₁ and π₂
```

## Repository contents

| File | Contents |
|---|---|
| `environment.py` | Maze generation |
| `L_next.py` | How a move updates the set of candidate positions |
| `solution_two.py`, `part_2_belief_sets.py` | Baseline policy π₀ and data recording |
| `data_generation.py` | Builds the training set |
| `Training.py` | Trains the CNN cost model |
| `Part_4.py` | Policy π₁ and comparison with π₀ |
| `PI1_model.py`, `Pi2.py` | Second iteration (C₁ / π₂) |
| `Project_3_CS520 BotLocalization.pdf` | Full report |

## Team

Sridevi Mudiyanur Ramamurthy and Zeba Afreen.
