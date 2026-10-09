# Pure Tabucol control experiment

This experiment removes RL completely while retaining the paper's Tabucol
implementation and parameters.

## Exact episode behavior

1. Seed 0--9 creates one initial uniform random `k`-coloring.
2. `TabuColSolver.set_solution(previous_solution)` is called at the beginning
   of every episode. In both the Rust implementation and Python fallback this
   resets `iters` to zero and clears the complete tabu matrix.
3. Unmodified Tabucol runs for at most `T=10000` iterations using `A=10`,
   `alpha=1.2`, and return probability `beta=0.2`.
4. The coloring returned by Tabucol is passed to the next episode.
5. A run stops at zero independently verified edge conflicts or after 18000
   seconds. Ten independent runs are used for each `(instance, k)`.

No policy checkpoint, neural network, Gym environment, RL action, ACO call, or
`keep_rl_best` behavior is present.

## Formal 11-instance campaign

Run from the `RLTCol-main` project directory with Python 3.13:

For the fixed-target comparison requested in the paper, run exactly one
reported RLTabucol `k_best` per instance (11 instances x 10 seeds):

```powershell
py -3.13 experiments\pure_tabucol_baseline\run_campaign.py --instances all --mode rltabucol-k --concurrency 2 --output-root experiments\pure_tabucol_baseline\results_rltabucol_k_5h
```

This mode measures pure Tabucol success and time at the same `k` already
reported for RLTabucol. It does not search for, and must not be labelled as,
pure Tabucol's own `k_best`. Use
`results_rltabucol_k_5h/pure_tabucol_matched_rltabucol_k.csv` for this
comparison.

To test exactly one harder target per instance without going below the
benchmark best-known color number, the default target is
`max(RLTabucol k_best - 1, best-known k)`. Instances already at the lower
bound reuse their completed matched-`k` results and are not launched again.
`flat1000_76_0` is an explicit exception: because pure Tabucol failed all ten
runs at `k=87`, its next informative probe is `k=88`, not `k=86`:

```powershell
py -3.13 experiments\pure_tabucol_baseline\run_campaign.py --instances all --mode rltabucol-k-minus-one --concurrency 2 --output-root experiments\pure_tabucol_baseline\results_rltabucol_k_5h
```

The dedicated summary is `pure_tabucol_bounded_k_minus_one.csv`.

To search for pure Tabucol's own `k_best`, use the more expensive adaptive
campaign:

```powershell
py -3.13 experiments\pure_tabucol_baseline\run_campaign.py --instances all --mode adaptive-kbest --concurrency 2
```

`adaptive-kbest` starts from the easier `k_best` shown in the existing paper
table, decreases `k` one at a time, and stops after the first lower level where
all ten runs fail. If the starting level unexpectedly has no success, it
searches upward for up to five levels. This produces a defensible pure-Tabucol
`k_best`; testing only one `k` does not.

For a shorter direct control restricted to the `k` values already shown in the
paper table:

```powershell
py -3.13 experiments\pure_tabucol_baseline\run_campaign.py --instances all --mode table-k --concurrency 2
```

Run selected instances by passing comma-separated names:

```powershell
py -3.13 experiments\pure_tabucol_baseline\run_campaign.py --instances DSJC250.5,flat300_28_0 --mode adaptive-kbest --concurrency 2
```

Completed seeds are skipped. Interrupted seeds resume from `checkpoint.json`,
including the previous coloring and already consumed runtime.

## Smoke test

This checks wiring only and must not be reported as a formal result:

```powershell
py -3.13 experiments\pure_tabucol_baseline\run_campaign.py --instances DSJC250.5 --mode table-k --num-seeds 2 --time-limit 5 --tabu-iters 100 --concurrency 2 --output-root experiments\pure_tabucol_baseline\smoke_results
```

## Outputs

- `results/<instance>/k<k>/seed<seed>/summary.json`: final run record.
- `episode_trace.csv`: episode-level handoff and reset evidence.
- `stdout.txt` and `stderr.txt`: process logs.
- `results/per_k_summary.csv`: every tested `k`, success count, time, scores.
- `results/per_seed_results.csv`: combined row for every completed seed.
- `results/pure_tabucol_main_table.csv`: paper-ready pure-Tabucol columns.
- `results/protocol.json`: exact campaign parameters.

The table's `Time (s)` is the mean wall-clock time among successful runs at
`k_best`. The `Success` field is the number of zero-conflict runs out of ten.

This is a wall-clock-matched solver comparison: pure Tabucol can complete more
Tabucol calls in five hours because it does not spend time on 300 RL actions.
That is appropriate for comparing end-to-end performance under the paper's
budget, but it is not an equal-call-count attribution experiment. If a strict
compute-matched attribution claim is needed, report it as a separate secondary
control rather than replacing this main-table baseline.

Do not reuse an output directory with different seeds, time limits, Tabucol
parameters, or concurrency. The launcher checks `protocol.json` and refuses to
mix incompatible records.

Formal timing requires the compiled Rust/PyO3 Tabucol extension. The Python
fallback is accepted only when `--allow-python-fallback` is explicitly supplied
for debugging; such timings must not be mixed with the paper results.

The existing native Rust solver uses `thread_rng()` internally and does not
expose a seed parameter. The recorded seed controls the initial coloring and
Python/NumPy randomness, while each native process obtains independent internal
randomness from the operating system. This matches the current solver behavior
but means exact move-by-move replay requires a future seeded Rust interface.
