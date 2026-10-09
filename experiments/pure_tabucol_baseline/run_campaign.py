"""Run and summarize the 11-instance pure-Tabucol control campaign."""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
SRC_DIR = PROJECT_ROOT / "src"
SINGLE_RUNNER = SCRIPT_DIR / "pure_tabucol_run.py"

# start_k is the easier of the two k_best values in the paper table.  The
# adaptive campaign decreases k one level at a time until ten runs all fail.
INSTANCES = {
    "DSJC250.5": {"graph": "DSJC250.5.col", "start_k": 28, "rltabucol_k": 28, "best_known_k": 28, "table_k": [28]},
    "DSJC500.1": {"graph": "DSJC500.1.col", "start_k": 13, "rltabucol_k": 12, "best_known_k": 12, "table_k": [13, 12]},
    "DSJC500.5": {"graph": "DSJC500.5.col", "start_k": 50, "rltabucol_k": 49, "best_known_k": 47, "table_k": [50, 49]},
    "DSJC500.9": {"graph": "DSJC500.9.col", "start_k": 130, "rltabucol_k": 128, "best_known_k": 126, "table_k": [130, 128]},
    "le450_15c": {"graph": "le450_15c.col", "start_k": 16, "rltabucol_k": 16, "best_known_k": 15, "table_k": [16, 15]},
    "le450_15d": {"graph": "le450_15d.col", "start_k": 16, "rltabucol_k": 16, "best_known_k": 15, "table_k": [16, 15]},
    "le450_25c": {"graph": "le450_25c.col", "start_k": 26, "rltabucol_k": 26, "best_known_k": 25, "table_k": [26]},
    "le450_25d": {"graph": "le450_25d.col", "start_k": 26, "rltabucol_k": 26, "best_known_k": 25, "table_k": [26]},
    "flat300_26_0": {"graph": "flat300_26_0.col", "start_k": 26, "rltabucol_k": 26, "best_known_k": 26, "table_k": [26]},
    "flat300_28_0": {"graph": "flat300_28_0.col", "start_k": 31, "rltabucol_k": 31, "best_known_k": 28, "table_k": [31, 30]},
    "flat1000_76_0": {"graph": "flat1000_76_0.col", "start_k": 90, "rltabucol_k": 87, "best_known_k": 81, "bounded_probe_k": 88, "table_k": [90, 87]},
}


def bounded_minus_one_k(config: dict) -> int:
    if "bounded_probe_k" in config:
        return int(config["bounded_probe_k"])
    return max(int(config["rltabucol_k"]) - 1, int(config["best_known_k"]))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--instances",
        default="all",
        help="Comma-separated instance names, or all.",
    )
    parser.add_argument(
        "--mode",
        choices=[
            "rltabucol-k",
            "rltabucol-k-minus-one",
            "adaptive-kbest",
            "table-k",
        ],
        default="adaptive-kbest",
        help=(
            "rltabucol-k runs one matched target per instance using the paper's "
            "reported RLTabucol k_best; rltabucol-k-minus-one runs exactly one "
            "target at max(reported k_best minus one, benchmark best-known k); "
            "adaptive-kbest searches downward to the "
            "first all-failure level; table-k runs every k appearing in the table."
        ),
    )
    parser.add_argument("--output-root", type=Path, default=SCRIPT_DIR / "results")
    parser.add_argument("--num-seeds", type=int, default=10)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--time-limit", type=float, default=18000.0)
    parser.add_argument("--tabu-iters", type=int, default=10000)
    parser.add_argument("--tabu-a", type=int, default=10)
    parser.add_argument("--tabu-alpha", type=float, default=1.2)
    parser.add_argument("--beta", type=float, default=0.2)
    parser.add_argument("--max-upward-steps", type=int, default=5)
    parser.add_argument("--min-k", type=int, default=1)
    parser.add_argument("--allow-python-fallback", action="store_true")
    parser.add_argument("--summarize-only", action="store_true")
    return parser.parse_args()


def selected_instances(value: str) -> list[str]:
    if value.strip().lower() == "all":
        return list(INSTANCES)
    names = [item.strip() for item in value.split(",") if item.strip()]
    unknown = [name for name in names if name not in INSTANCES]
    if unknown:
        raise ValueError(f"Unknown instances: {', '.join(unknown)}")
    return names


def run_dir(output_root: Path, instance: str, k: int, seed: int) -> Path:
    return output_root / instance / f"k{k}" / f"seed{seed}"


def load_level(
    output_root: Path, instance: str, k: int, seeds: list[int]
) -> list[dict]:
    rows = []
    for seed in seeds:
        path = run_dir(output_root, instance, k, seed) / "summary.json"
        if path.exists():
            rows.append(json.loads(path.read_text(encoding="utf-8")))
    return rows


def invoke_one(args: argparse.Namespace, instance: str, k: int, seed: int) -> int:
    cfg = INSTANCES[instance]
    graph = SRC_DIR / cfg["graph"]
    output = run_dir(args.output_root, instance, k, seed)
    command = [
        sys.executable,
        str(SINGLE_RUNNER),
        "--graph",
        str(graph),
        "--k",
        str(k),
        "--seed",
        str(seed),
        "--output-dir",
        str(output),
        "--time-limit",
        str(args.time_limit),
        "--tabu-iters",
        str(args.tabu_iters),
        "--tabu-a",
        str(args.tabu_a),
        "--tabu-alpha",
        str(args.tabu_alpha),
        "--beta",
        str(args.beta),
    ]
    if args.allow_python_fallback:
        command.append("--allow-python-fallback")
    output.mkdir(parents=True, exist_ok=True)
    stdout_path = output / "stdout.txt"
    stderr_path = output / "stderr.txt"
    print(f"[LAUNCH] {instance} k={k} seed={seed}", flush=True)
    with stdout_path.open("a", encoding="utf-8") as stdout, stderr_path.open(
        "a", encoding="utf-8"
    ) as stderr:
        process = subprocess.run(command, stdout=stdout, stderr=stderr, check=False)
    print(
        f"[EXIT {process.returncode}] {instance} k={k} seed={seed}", flush=True
    )
    return process.returncode


def run_level(args: argparse.Namespace, instance: str, k: int, seeds: list[int]) -> list[dict]:
    complete = load_level(args.output_root, instance, k, seeds)
    completed_seeds = {int(row["seed"]) for row in complete}
    pending = [seed for seed in seeds if seed not in completed_seeds]
    if pending and not args.summarize_only:
        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            futures = {
                pool.submit(invoke_one, args, instance, k, seed): seed for seed in pending
            }
            for future in as_completed(futures):
                code = future.result()
                if code not in (0, 130):
                    seed = futures[future]
                    print(
                        f"[ERROR] {instance} k={k} seed={seed}; inspect stderr.txt",
                        flush=True,
                    )
    return load_level(args.output_root, instance, k, seeds)


def run_bounded_minus_one_global(
    args: argparse.Namespace,
    instances: list[str],
    seeds: list[int],
    summary_instances: list[str],
) -> None:
    """Keep the worker pool full across all fixed target instances."""
    pending: list[tuple[str, int, int]] = []
    for instance in instances:
        cfg = INSTANCES[instance]
        target_k = bounded_minus_one_k(cfg)
        if "bounded_probe_k" not in cfg and target_k >= int(cfg["rltabucol_k"]):
            continue
        complete = load_level(args.output_root, instance, target_k, seeds)
        completed_seeds = {int(row["seed"]) for row in complete}
        pending.extend(
            (instance, target_k, seed)
            for seed in seeds
            if seed not in completed_seeds
        )

    print(
        f"[GLOBAL QUEUE] pending={len(pending)} concurrency={args.concurrency}",
        flush=True,
    )
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = {
            pool.submit(invoke_one, args, instance, k, seed): (instance, k, seed)
            for instance, k, seed in pending
        }
        for completed_count, future in enumerate(as_completed(futures), start=1):
            instance, k, seed = futures[future]
            code = future.result()
            if code not in (0, 130):
                print(
                    f"[ERROR] {instance} k={k} seed={seed}; inspect stderr.txt",
                    flush=True,
                )
            if completed_count % args.concurrency == 0 or completed_count == len(futures):
                refresh_summaries(args.output_root, summary_instances, seeds)


def aggregate_level(instance: str, k: int, rows: list[dict], expected: int) -> dict:
    successes = [row for row in rows if bool(row["success"])]
    scores = [int(row["score"]) for row in rows]
    all_times = [float(row["time_s"]) for row in rows]
    success_times = [float(row["time_s"]) for row in successes]
    return {
        "instance": instance,
        "k": k,
        "completed_runs": len(rows),
        "expected_runs": expected,
        "successes": len(successes),
        "success_rate": len(successes) / expected if expected else math.nan,
        "mean_success_time_s": statistics.fmean(success_times) if success_times else "",
        "median_success_time_s": statistics.median(success_times) if success_times else "",
        "mean_all_time_s": statistics.fmean(all_times) if all_times else "",
        "best_score": min(scores) if scores else "",
        "mean_score": statistics.fmean(scores) if scores else "",
        "native_runs": sum(bool(row.get("native_tabucol")) for row in rows),
    }


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def refresh_summaries(output_root: Path, instances: list[str], seeds: list[int]) -> None:
    level_rows = []
    main_rows = []
    matched_rows = []
    minus_one_rows = []
    seed_rows = []
    for instance in instances:
        instance_dir = output_root / instance
        levels = []
        if instance_dir.exists():
            for k_dir in instance_dir.glob("k*"):
                try:
                    k = int(k_dir.name[1:])
                except ValueError:
                    continue
                rows = load_level(output_root, instance, k, seeds)
                if rows:
                    levels.append(aggregate_level(instance, k, rows, len(seeds)))
                    seed_rows.extend(
                        {
                            "instance": instance,
                            "k": k,
                            "seed": int(row["seed"]),
                            "score": int(row["score"]),
                            "success": bool(row["success"]),
                            "time_s": float(row["time_s"]),
                            "episodes": int(row["episodes"]),
                            "best_episode": int(row["best_episode"]),
                            "native_tabucol": bool(row.get("native_tabucol")),
                            "tabu_reset_each_episode": bool(
                                row.get("tabu_reset_each_episode")
                            ),
                            "solution_handoff_each_episode": bool(
                                row.get("solution_handoff_each_episode")
                            ),
                            "rl_used": bool(row.get("rl_used")),
                        }
                        for row in rows
                    )
        levels.sort(key=lambda row: int(row["k"]), reverse=True)
        level_rows.extend(levels)
        solved = [row for row in levels if int(row["successes"]) > 0]
        kbest_row = min(solved, key=lambda row: int(row["k"])) if solved else None
        search_complete = bool(kbest_row) and any(
            int(row["completed_runs"]) == len(seeds)
            and int(row["successes"]) == 0
            and int(row["k"]) < int(kbest_row["k"])
            for row in levels
        )
        main_rows.append(
            {
                "instance": instance,
                "k_best": kbest_row["k"] if search_complete else "",
                "time_s": kbest_row["mean_success_time_s"] if search_complete else "",
                "success": (
                    f"{kbest_row['successes']}/{kbest_row['expected_runs']}"
                    if search_complete
                    else f"0/{len(seeds)}"
                ),
                "best_score_at_k_best": kbest_row["best_score"] if search_complete else "",
                "search_complete": search_complete,
            }
        )
        target_k = int(INSTANCES[instance]["rltabucol_k"])
        target_row = next(
            (row for row in levels if int(row["k"]) == target_k), None
        )
        matched_rows.append(
            {
                "instance": instance,
                "target_k": target_k,
                "completed_runs": target_row["completed_runs"] if target_row else 0,
                "time_s": target_row["mean_success_time_s"] if target_row else "",
                "mean_success_time_s": (
                    target_row["mean_success_time_s"] if target_row else ""
                ),
                "mean_all_time_s": target_row["mean_all_time_s"] if target_row else "",
                "success": (
                    f"{target_row['successes']}/{target_row['expected_runs']}"
                    if target_row
                    else f"0/{len(seeds)}"
                ),
                "best_score": target_row["best_score"] if target_row else "",
                "mean_score": target_row["mean_score"] if target_row else "",
            }
        )
        minus_one_k = bounded_minus_one_k(INSTANCES[instance])
        minus_one_row = next(
            (row for row in levels if int(row["k"]) == minus_one_k), None
        )
        minus_one_rows.append(
            {
                "instance": instance,
                "target_k": minus_one_k,
                "completed_runs": minus_one_row["completed_runs"] if minus_one_row else 0,
                "time_s": minus_one_row["mean_success_time_s"] if minus_one_row else "",
                "mean_success_time_s": (
                    minus_one_row["mean_success_time_s"] if minus_one_row else ""
                ),
                "mean_all_time_s": (
                    minus_one_row["mean_all_time_s"] if minus_one_row else ""
                ),
                "success": (
                    f"{minus_one_row['successes']}/{minus_one_row['expected_runs']}"
                    if minus_one_row
                    else f"0/{len(seeds)}"
                ),
                "best_score": minus_one_row["best_score"] if minus_one_row else "",
                "mean_score": minus_one_row["mean_score"] if minus_one_row else "",
            }
        )

    level_fields = [
        "instance",
        "k",
        "completed_runs",
        "expected_runs",
        "successes",
        "success_rate",
        "mean_success_time_s",
        "median_success_time_s",
        "mean_all_time_s",
        "best_score",
        "mean_score",
        "native_runs",
    ]
    write_csv(output_root / "per_k_summary.csv", level_rows, level_fields)
    seed_rows.sort(key=lambda row: (str(row["instance"]), -int(row["k"]), int(row["seed"])))
    write_csv(
        output_root / "per_seed_results.csv",
        seed_rows,
        [
            "instance",
            "k",
            "seed",
            "score",
            "success",
            "time_s",
            "episodes",
            "best_episode",
            "native_tabucol",
            "tabu_reset_each_episode",
            "solution_handoff_each_episode",
            "rl_used",
        ],
    )
    write_csv(
        output_root / "pure_tabucol_main_table.csv",
        main_rows,
        [
            "instance",
            "k_best",
            "time_s",
            "success",
            "best_score_at_k_best",
            "search_complete",
        ],
    )
    write_csv(
        output_root / "pure_tabucol_matched_rltabucol_k.csv",
        matched_rows,
        [
            "instance",
            "target_k",
            "completed_runs",
            "time_s",
            "mean_success_time_s",
            "mean_all_time_s",
            "success",
            "best_score",
            "mean_score",
        ],
    )
    write_csv(
        output_root / "pure_tabucol_rltabucol_k_minus_one.csv",
        minus_one_rows,
        [
            "instance",
            "target_k",
            "completed_runs",
            "time_s",
            "mean_success_time_s",
            "mean_all_time_s",
            "success",
            "best_score",
            "mean_score",
        ],
    )
    write_csv(
        output_root / "pure_tabucol_bounded_k_minus_one.csv",
        minus_one_rows,
        [
            "instance",
            "target_k",
            "completed_runs",
            "time_s",
            "mean_success_time_s",
            "mean_all_time_s",
            "success",
            "best_score",
            "mean_score",
        ],
    )


def main() -> int:
    args = parse_args()
    if args.num_seeds < 1 or args.concurrency < 1:
        raise ValueError("num-seeds and concurrency must be positive")
    instances = selected_instances(args.instances)
    seeds = list(range(args.seed_start, args.seed_start + args.num_seeds))
    args.output_root = args.output_root.resolve()
    args.output_root.mkdir(parents=True, exist_ok=True)

    protocol_path = args.output_root / "protocol.json"
    protocol = {
        "method": "PureTabucol-episodic-reset",
        "instances": instances,
        "mode": args.mode,
        "modes": [args.mode],
        "seeds": seeds,
        "concurrency": args.concurrency,
        "time_limit_s_per_seed": args.time_limit,
        "tabucol_iterations_per_episode": args.tabu_iters,
        "tabu_a": args.tabu_a,
        "tabu_alpha": args.tabu_alpha,
        "tabucol_return_beta": args.beta,
        "tabu_reset_each_episode": True,
        "solution_handoff_each_episode": True,
        "rl_used": False,
    }
    if protocol_path.exists():
        previous = json.loads(protocol_path.read_text(encoding="utf-8"))
        protected_keys = [
            "method",
            "seeds",
            "concurrency",
            "time_limit_s_per_seed",
            "tabucol_iterations_per_episode",
            "tabu_a",
            "tabu_alpha",
            "tabucol_return_beta",
            "tabu_reset_each_episode",
            "solution_handoff_each_episode",
            "rl_used",
        ]
        mismatches = [
            key for key in protected_keys if previous.get(key) != protocol.get(key)
        ]
        if mismatches:
            raise RuntimeError(
                "Output directory contains a different protocol. Use a new "
                f"--output-root. Mismatched fields: {', '.join(mismatches)}"
            )
        protocol["instances"] = sorted(
            set(previous.get("instances", [])) | set(instances)
        )
        previous_modes = previous.get("modes", [previous.get("mode")])
        protocol["modes"] = sorted(
            {mode for mode in [*previous_modes, args.mode] if mode}
        )
    protocol_path.write_text(json.dumps(protocol, indent=2), encoding="utf-8")
    summary_instances = sorted(
        set(instances)
        | {
            path.name
            for path in args.output_root.iterdir()
            if path.is_dir() and path.name in INSTANCES
        }
    )

    print(
        f"[PROTOCOL] mode={args.mode} runs={args.num_seeds} concurrency={args.concurrency} "
        f"budget={args.time_limit}s T={args.tabu_iters} A={args.tabu_a} "
        f"alpha={args.tabu_alpha} beta={args.beta}",
        flush=True,
    )

    if not args.summarize_only:
        if args.mode == "rltabucol-k-minus-one":
            run_bounded_minus_one_global(
                args, instances, seeds, summary_instances
            )
            refresh_summaries(args.output_root, summary_instances, seeds)
            print(f"[DONE] Summaries written under {args.output_root}", flush=True)
            return 0
        for instance in instances:
            cfg = INSTANCES[instance]
            if args.mode == "rltabucol-k":
                run_level(args, instance, int(cfg["rltabucol_k"]), seeds)
                refresh_summaries(args.output_root, summary_instances, seeds)
                continue
            if args.mode == "table-k":
                for k in cfg["table_k"]:
                    run_level(args, instance, int(k), seeds)
                    refresh_summaries(args.output_root, summary_instances, seeds)
                continue

            start_k = int(cfg["start_k"])
            start_rows = run_level(args, instance, start_k, seeds)
            refresh_summaries(args.output_root, summary_instances, seeds)
            if len(start_rows) != len(seeds):
                print(f"[STOP] Incomplete level: {instance} k={start_k}", flush=True)
                continue

            if not any(bool(row["success"]) for row in start_rows):
                for step in range(1, args.max_upward_steps + 1):
                    k = start_k + step
                    rows = run_level(args, instance, k, seeds)
                    refresh_summaries(args.output_root, summary_instances, seeds)
                    if len(rows) != len(seeds) or any(bool(row["success"]) for row in rows):
                        break
                continue

            k = start_k - 1
            while k >= args.min_k:
                rows = run_level(args, instance, k, seeds)
                refresh_summaries(args.output_root, summary_instances, seeds)
                if len(rows) != len(seeds):
                    print(f"[STOP] Incomplete level: {instance} k={k}", flush=True)
                    break
                if not any(bool(row["success"]) for row in rows):
                    print(
                        f"[BOUNDARY] {instance}: all 10 runs failed at k={k}; "
                        "lower k values are not scheduled.",
                        flush=True,
                    )
                    break
                k -= 1

    refresh_summaries(args.output_root, summary_instances, seeds)
    print(f"[DONE] Summaries written under {args.output_root}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
