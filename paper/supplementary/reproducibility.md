# Reproducibility Statement

## Code availability
Code and processed experimental records will be released upon acceptance.

## Experimental setup
- Python 3.x with PyTorch, Tianshou, gymnasium
- RLTabucol: 10 runs per (instance, k), time limit 18000s
- RLACO: 10 runs per (instance, k), time limit 18000s, ACO local-search OFF
- Actor-only: le450_15d, k=15, Tabucol disabled

## Hardware
Experiments run on CPU (RLTabucol) and CPU (RLACO). No GPU required for
inference.

## Random seeds
RLACO and RLACO_0 are not seed-paired (different run timestamps).
