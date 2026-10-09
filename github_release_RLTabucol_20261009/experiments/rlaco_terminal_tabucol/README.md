# RLACO + Terminal Tabucol Professor Protocol

This package implements one experiment only:

1. Run the complete learned-policy RLACO stage with the instance-specific budget.
2. Track the coloring corresponding to the lowest conflict count across all RLACO episodes.
3. Start a separate terminal Tabucol stage from that verified coloring.
4. Run repeated 10,000-iteration Tabucol blocks until feasibility or 18,000 seconds.

It is not the per-episode `RL -> ACO -> Tabucol` polish pipeline.

## Locked formal settings

- 10 fixed instances and fixed `k` values in `run_campaign.py`.
- 10 runs, seeds 0 through 9.
- RL: trained `policy.pt`, 300 steps per episode, `keep_rl_best=False`.
- ACO: 40 ants, 50 iterations, learned-policy soft pheromone prior.
- ACO fixed defaults: alpha 1.0, beta 3.0, rho 0.2, tau0 1.0,
  tau_min 0.05, tau_max 8.0, injection strength 8.0.
- No per-episode Tabucol polish.
- Terminal Tabucol: T=10,000, A=10, alpha=1.2, return beta=0.2.
- Terminal budget: 18,000 seconds per run.
- The tabu table resets at every block. The returned coloring seeds the next
  block, while a separate global-best coloring is retained for reporting.

ACO beta (3.0) and Tabucol return beta (0.2) are different parameters.

## Setup

Use Python 3.13 when reproducing the supplied Windows environment. A virtual
environment is recommended.

Windows PowerShell:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\experiments\rlaco_terminal_tabucol\setup_windows.ps1
```

Linux / AutoDL:

```bash
bash experiments/rlaco_terminal_tabucol/setup_linux.sh
```

The setup script creates `.venv` with access to existing system packages and
compiles the Rust/PyO3 Tabucol backend. Formal timing refuses
to run with the slower Python fallback. On Windows, Rust also requires the
Microsoft C++ Build Tools with the Desktop development with C++ workload.

## Verify before a long run

```powershell
.\.venv\Scripts\python.exe experiments\rlaco_terminal_tabucol\verify_install.py
.\.venv\Scripts\python.exe -m unittest discover -s experiments\rlaco_terminal_tabucol\tests -v
```

## Formal command

Windows PowerShell or the PyCharm terminal, from the package root:

```powershell
.\.venv\Scripts\python.exe experiments\rlaco_terminal_tabucol\run_campaign.py --instances all --num-seeds 10 --concurrency 2 --output-root experiments\rlaco_terminal_tabucol\results_formal
```

Linux / AutoDL:

```bash
./.venv/bin/python experiments/rlaco_terminal_tabucol/run_campaign.py --instances all --num-seeds 10 --concurrency 2 --output-root experiments/rlaco_terminal_tabucol/results_formal
```

The formal launcher rejects parameter changes that would violate the professor
protocol. `--concurrency` may be changed because it affects scheduling, not an
individual run's algorithm or budget.

## Short wiring test

This is not publishable data. It may use the Python fallback and shortened
budgets only to verify paths and output files:

```powershell
.\.venv\Scripts\python.exe experiments\rlaco_terminal_tabucol\run_campaign.py --instances DSJC500.1 --num-seeds 1 --concurrency 1 --smoke-rlaco-time-limit 2 --terminal-time-limit 0.1 --tabu-iters 50 --allow-python-fallback --output-root experiments\rlaco_terminal_tabucol\results_smoke
```

## Output and resume behavior

Each `instance/k/seed` directory contains:

- `rlaco_summary.json`: verified RLACO global-best score and coloring.
- `rlaco_stdout.txt`, `rlaco_stderr.txt`: complete RLACO process logs.
- `terminal_tabucol/checkpoint.json`: checkpoint after every Tabucol block.
- `terminal_tabucol/block_trace.csv`: per-block scores, iterations and seeds.
- `terminal_tabucol/summary.json`: final per-seed result.

The output root also contains:

- `protocol.json`: locked parameters and SHA-256 of `policy.pt`.
- `per_seed_results.csv`: one row per completed seed.
- `per_instance_summary.csv`: success count, mean successful time, and conflict statistics.

Rerun the same command to resume. Completed stages are skipped. A terminal
checkpoint resumes from the exact block number and accumulated active runtime.
Changing the experiment protocol requires a new output directory.

## Runtime interpretation

The worst-case total is approximately 870 worker-hours for 100 runs: about 370
hours of configured RLACO budgets plus 500 hours of terminal Tabucol budgets.
With concurrency 2, the worst-case wall time is about 435 hours. Feasible runs
stop early, so actual time can be lower. RL inference can use a GPU, while ACO
and terminal Tabucol remain primarily CPU workloads.
