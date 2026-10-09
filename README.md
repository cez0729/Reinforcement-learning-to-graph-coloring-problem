# RL Guidance Across Tabucol and ACO

This repository contains the code, experiment launchers, trained policy, benchmark instances, and source files for the conference paper:

`From Tabucol to Ant Colony Optimization: Transferring Reinforcement-Learning Guidance for Graph Coloring`

The project studies a frozen PPO policy trained in RLTabucol and reused in ACO through soft pheromone injection. It also contains the Pure Tabucol baseline and the terminal RLACO-to-Tabucol experiment.

## Repository layout

- `src/`: RLTabucol, RLACO, ACO, graph environment, Rust/PyO3 Tabucol backend, benchmark files, and policy weights.
- `experiments/pure_tabucol_baseline/`: standalone Tabucol baseline campaign.
- `experiments/rlaco_terminal_tabucol/`: RLACO-to-terminal-Tabucol campaign and protocol checks.
- `paper/`: latest LNCS/IEEE paper sources, figures, tables, and reproducibility notes.
- `configs/`: experiment protocol metadata.
- `results/`: instructions for storing local results. Formal raw logs are intentionally kept outside the source release.

## Environment

- Python 3.13 is recommended for the supplied experiment protocol.
- Rust and Cargo are required to build the native Tabucol extension.
- Windows also requires the MSVC C++ build tools.

Install the base Python dependencies with:

```bash
python -m pip install -r requirements.txt
```

For the formal RLACO-to-Tabucol experiment, use the setup script in `experiments/rlaco_terminal_tabucol/`. It creates a virtual environment, installs the experiment dependencies, builds the Rust extension, and verifies the installation.

## Quick verification

From the repository root:

```bash
python experiments/rlaco_terminal_tabucol/verify_install.py
python -m unittest discover -s experiments/rlaco_terminal_tabucol/tests -v
```

Run the short smoke tests described in the two experiment README files before starting a formal campaign.

## Formal experiments

The formal settings are recorded in `configs/conference_protocol.json` and in the experiment README files. The main protocol uses ten independent seeds. Formal timing must use the native Rust backend; the Python fallback is for wiring tests only.

The DIMACS instances are included for convenience. They originate from the public DIMACS graph-coloring benchmark collection; retain the original source attribution when redistributing them.

## Paper

The latest paper source is `paper/src/main_lncs_blind.tex`. Compile from the `paper` directory so that the relative paths to `figures`, `tables`, and `supplementary` resolve correctly.

## Reproducibility note

The native Rust solver currently uses its own operating-system randomness internally. The recorded seed controls the Python/NumPy side and the initial coloring, but exact move-by-move replay is not guaranteed by the current native interface.
