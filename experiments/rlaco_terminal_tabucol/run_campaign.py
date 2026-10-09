"""Launch and summarize the professor-requested 10-instance experiment."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import statistics
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
SRC_DIR = PROJECT_ROOT / "src"
RUNNER = SRC_DIR / "runner.py"
TERMINAL_RUNNER = SCRIPT_DIR / "terminal_tabucol.py"
PROTOCOL_VERSION = 1

EXPECTED_PARAMETERS = {
    "num_seeds": 10,
    "seed_start": 0,
    "terminal_time_limit": 18000.0,
    "tabu_iters": 10000,
    "tabu_a": 10,
    "tabu_alpha": 1.2,
    "tabucol_beta": 0.2,
    "aco_ants": 40,
    "aco_iter": 50,
    "rl_steps": 300,
}

ACO_FIXED_PARAMETERS = {
    "alpha": 1.0,
    "beta": 3.0,
    "rho": 0.2,
    "q": 1.0,
    "tau0": 1.0,
    "tau_min": 0.05,
    "tau_max": 8.0,
    "q0": 0.85,
    "elite_count": 3,
    "global_best_weight": 2.0,
    "injection_strength": 8.0,
}


# Fixed k and RLACO budgets reproduce the current RLACO experiment table.
INSTANCES = {
    "DSJC250.5": {"graph": "DSJC250.5.col", "k": 28, "rlaco_time": 14400},
    "DSJC500.1": {"graph": "DSJC500.1.col", "k": 12, "rlaco_time": 7200},
    "DSJC500.5": {"graph": "DSJC500.5.col", "k": 48, "rlaco_time": 18000},
    "DSJC500.9": {"graph": "DSJC500.9.col", "k": 126, "rlaco_time": 21600},
    "flat300_26_0": {"graph": "flat300_26_0.col", "k": 26, "rlaco_time": 14400},
    "flat300_28_0": {"graph": "flat300_28_0.col", "k": 28, "rlaco_time": 14400},
    "le450_15c": {"graph": "le450_15c.col", "k": 15, "rlaco_time": 10800},
    "le450_15d": {"graph": "le450_15d.col", "k": 15, "rlaco_time": 10800},
    "le450_25c": {"graph": "le450_25c.col", "k": 25, "rlaco_time": 10800},
    "le450_25d": {"graph": "le450_25d.col", "k": 25, "rlaco_time": 10800},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--instances", default="all")
    parser.add_argument("--output-root", type=Path, default=SCRIPT_DIR / "results")
    parser.add_argument("--policy", type=Path, default=SRC_DIR / "policy.pt")
    parser.add_argument("--num-seeds", type=int, default=10)
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--concurrency", type=int, default=2)
    parser.add_argument("--terminal-time-limit", type=float, default=18000.0)
    parser.add_argument("--tabu-iters", type=int, default=10000)
    parser.add_argument("--tabu-a", type=int, default=10)
    parser.add_argument("--tabu-alpha", type=float, default=1.2)
    parser.add_argument("--beta", type=float, default=0.2)
    parser.add_argument("--aco-ants", type=int, default=40)
    parser.add_argument("--aco-iter", type=int, default=50)
    parser.add_argument("--rl-steps", type=int, default=300)
    parser.add_argument(
        "--smoke-rlaco-time-limit",
        type=float,
        default=None,
        help="Override every RLACO budget for wiring tests only.",
    )
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


def run_dir(root: Path, instance: str, seed: int) -> Path:
    cfg = INSTANCES[instance]
    return root / instance / f"k{cfg['k']}" / f"seed{seed}"


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def solution_digest(solution: list[int]) -> str:
    payload = ",".join(str(int(value)) for value in solution).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def dimacs_score(path: Path, solution: list[int]) -> int:
    graph_order = None
    conflicts = 0
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for raw in handle:
            fields = raw.strip().split()
            if not fields or fields[0] == "c":
                continue
            if fields[0] == "p":
                graph_order = int(fields[2])
            elif fields[0] == "e":
                u, v = int(fields[1]) - 1, int(fields[2]) - 1
                if solution[u] == solution[v]:
                    conflicts += 1
    if graph_order is None or len(solution) != graph_order:
        raise RuntimeError(
            f"Coloring length {len(solution)} does not match DIMACS order "
            f"{graph_order} for {path.name}"
        )
    return conflicts


def preflight(args: argparse.Namespace, instances: list[str]) -> None:
    if args.num_seeds < 1 or args.concurrency < 1:
        raise ValueError("num-seeds and concurrency must be positive")
    if args.terminal_time_limit <= 0 or args.tabu_iters < 1:
        raise ValueError("terminal-time-limit and tabu-iters must be positive")
    if not 0.0 <= args.beta <= 1.0:
        raise ValueError("beta must be in [0, 1]")
    smoke_mode = args.smoke_rlaco_time_limit is not None
    if args.allow_python_fallback and not smoke_mode:
        raise ValueError(
            "--allow-python-fallback is restricted to smoke tests; formal runs "
            "must use the native Rust Tabucol backend."
        )
    if not smoke_mode:
        actual = {
            "num_seeds": args.num_seeds,
            "seed_start": args.seed_start,
            "terminal_time_limit": args.terminal_time_limit,
            "tabu_iters": args.tabu_iters,
            "tabu_a": args.tabu_a,
            "tabu_alpha": args.tabu_alpha,
            "tabucol_beta": args.beta,
            "aco_ants": args.aco_ants,
            "aco_iter": args.aco_iter,
            "rl_steps": args.rl_steps,
        }
        mismatches = {
            key: (actual[key], expected)
            for key, expected in EXPECTED_PARAMETERS.items()
            if actual[key] != expected
        }
        if mismatches:
            raise ValueError(
                "Formal protocol parameters cannot be changed: " + repr(mismatches)
            )
    if not args.policy.exists():
        raise FileNotFoundError(args.policy)
    for instance in instances:
        graph = SRC_DIR / INSTANCES[instance]["graph"]
        if not graph.exists():
            raise FileNotFoundError(graph)

    environment = os.environ.copy()
    if args.allow_python_fallback:
        environment["RLTABUCOL_FORCE_PYTHON"] = "1"
    check = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; sys.path.insert(0, r'%s'); import tabucol; "
                "assert hasattr(tabucol.TabuColSolver, 'get_best_solution'); "
                "assert tabucol.IS_NATIVE or %s; "
                "tabucol.TabuColSolver([[]], 1, seed=0); "
                "print(tabucol.IS_NATIVE, tabucol.BACKEND_ORIGIN)"
            )
            % (SRC_DIR, repr(args.allow_python_fallback)),
        ],
        cwd=PROJECT_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    if check.returncode != 0:
        raise RuntimeError(
            "Tabucol preflight failed. Build the packaged Rust extension.\n"
            + check.stdout
            + check.stderr
        )
    print(f"[PREFLIGHT] Tabucol backend: {check.stdout.strip()}", flush=True)


def run_rlaco(args: argparse.Namespace, instance: str, seed: int) -> int:
    cfg = INSTANCES[instance]
    output = run_dir(args.output_root, instance, seed)
    output.mkdir(parents=True, exist_ok=True)
    result_path = output / "rlaco_summary.json"
    if result_path.exists():
        print(f"[SKIP RLACO] {instance} seed={seed}", flush=True)
        return 0
    budget = (
        args.smoke_rlaco_time_limit
        if args.smoke_rlaco_time_limit is not None
        else cfg["rlaco_time"]
    )
    command = [
        sys.executable,
        str(RUNNER),
        str(args.policy.resolve()),
        str((SRC_DIR / cfg["graph"]).resolve()),
        str(cfg["k"]),
        "--RL",
        "True",
        "--use-aco",
        "True",
        "--inject-mode",
        "soft",
        "--aco-ants",
        str(args.aco_ants),
        "--aco-iter",
        str(args.aco_iter),
        "--max-steps",
        str(args.rl_steps),
        "--beta",
        str(args.beta),
        "--time-limit",
        str(int(budget)),
        "--seed",
        str(seed),
        "--result-json",
        str(result_path.resolve()),
    ]
    print(f"[RLACO] {instance} k={cfg['k']} seed={seed} budget={budget}s", flush=True)
    with (output / "rlaco_stdout.txt").open("w", encoding="utf-8") as stdout, (
        output / "rlaco_stderr.txt"
    ).open("w", encoding="utf-8") as stderr:
        process = subprocess.run(
            command, cwd=SRC_DIR, stdout=stdout, stderr=stderr, check=False
        )
    return process.returncode


def run_terminal(args: argparse.Namespace, instance: str, seed: int) -> int:
    cfg = INSTANCES[instance]
    output = run_dir(args.output_root, instance, seed)
    handoff = output / "rlaco_summary.json"
    terminal_output = output / "terminal_tabucol"
    summary = terminal_output / "summary.json"
    if summary.exists():
        print(f"[SKIP TERMINAL] {instance} seed={seed}", flush=True)
        return 0
    if not handoff.exists():
        return 2
    command = [
        sys.executable,
        str(TERMINAL_RUNNER),
        "--rlaco-result",
        str(handoff.resolve()),
        "--graph",
        str((SRC_DIR / cfg["graph"]).resolve()),
        "--k",
        str(cfg["k"]),
        "--seed",
        str(seed),
        "--output-dir",
        str(terminal_output.resolve()),
        "--time-limit",
        str(args.terminal_time_limit),
        "--tabu-iters",
        str(args.tabu_iters),
        "--tabu-a",
        str(args.tabu_a),
        "--tabu-alpha",
        str(args.tabu_alpha),
        "--beta",
        str(args.beta),
    ]
    environment = os.environ.copy()
    if args.allow_python_fallback:
        command.append("--allow-python-fallback")
        environment["RLTABUCOL_FORCE_PYTHON"] = "1"
    terminal_output.mkdir(parents=True, exist_ok=True)
    print(
        f"[TERMINAL] {instance} k={cfg['k']} seed={seed} "
        f"budget={args.terminal_time_limit}s",
        flush=True,
    )
    with (terminal_output / "stdout.txt").open("a", encoding="utf-8") as stdout, (
        terminal_output / "stderr.txt"
    ).open("a", encoding="utf-8") as stderr:
        process = subprocess.run(
            command,
            cwd=PROJECT_ROOT,
            env=environment,
            stdout=stdout,
            stderr=stderr,
            check=False,
        )
    return process.returncode


def run_one(args: argparse.Namespace, instance: str, seed: int) -> tuple[str, int, int]:
    code = run_rlaco(args, instance, seed)
    if code != 0:
        return instance, seed, code
    return instance, seed, run_terminal(args, instance, seed)


def validate_completed_result(
    root: Path, instance: str, seed: int, row: dict
) -> None:
    cfg = INSTANCES[instance]
    if row["instance"] != instance or int(row["k"]) != cfg["k"]:
        raise RuntimeError(f"Result identity mismatch for {instance} seed={seed}")
    if int(row["seed"]) != seed:
        raise RuntimeError(f"Result seed mismatch for {instance} seed={seed}")

    graph = SRC_DIR / cfg["graph"]
    final_solution = [int(value) for value in row["global_best_solution"]]
    final_score = dimacs_score(graph, final_solution)
    if final_score != int(row["score"]):
        raise RuntimeError(
            f"Final score/coloring mismatch for {instance} seed={seed}: "
            f"{row['score']} != {final_score}"
        )

    handoff_path = run_dir(root, instance, seed) / "rlaco_summary.json"
    handoff = json.loads(handoff_path.read_text(encoding="utf-8"))
    handoff_solution = [int(value) for value in handoff["rlaco_best_solution"]]
    handoff_score = dimacs_score(graph, handoff_solution)
    if handoff_score != int(row["handoff_score"]):
        raise RuntimeError(f"Handoff score mismatch for {instance} seed={seed}")
    if solution_digest(handoff_solution) != row["handoff_digest"]:
        raise RuntimeError(f"Handoff coloring digest mismatch for {instance} seed={seed}")

    protocol = json.loads((root / "protocol.json").read_text(encoding="utf-8"))
    if protocol["run_type"] == "formal" and not bool(row["tabucol_native"]):
        raise RuntimeError(f"Formal result used Python Tabucol: {instance} seed={seed}")


def load_results(root: Path, instances: list[str], seeds: list[int]) -> list[dict]:
    rows = []
    for instance in instances:
        for seed in seeds:
            path = run_dir(root, instance, seed) / "terminal_tabucol" / "summary.json"
            if path.exists():
                row = json.loads(path.read_text(encoding="utf-8"))
                validate_completed_result(root, instance, seed, row)
                rows.append(row)
    return rows


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def summarize(root: Path, instances: list[str], seeds: list[int]) -> None:
    rows = load_results(root, instances, seeds)
    seed_rows = []
    for row in rows:
        seed_rows.append(
            {
                "instance": row["instance"],
                "k": row["k"],
                "seed": row["seed"],
                "handoff_conflicts": row["handoff_score"],
                "final_conflicts": row["score"],
                "success": int(bool(row["success"])),
                "rlaco_time_s": row["rlaco_elapsed_s"],
                "terminal_time_s": row["terminal_elapsed_s"],
                "terminal_time_to_feasibility_s": row["terminal_time_to_feasibility_s"],
                "end_to_end_time_to_feasibility_s": row["end_to_end_time_to_feasibility_s"],
                "blocks": row["blocks"],
            }
        )
    write_csv(
        root / "per_seed_results.csv",
        seed_rows,
        [
            "instance",
            "k",
            "seed",
            "handoff_conflicts",
            "final_conflicts",
            "success",
            "rlaco_time_s",
            "terminal_time_s",
            "terminal_time_to_feasibility_s",
            "end_to_end_time_to_feasibility_s",
            "blocks",
        ],
    )

    summary_rows = []
    for instance in instances:
        group = [row for row in seed_rows if row["instance"] == instance]
        scores = [int(row["final_conflicts"]) for row in group]
        successful = [
            float(row["terminal_time_to_feasibility_s"])
            for row in group
            if row["success"] and row["terminal_time_to_feasibility_s"] is not None
        ]
        summary_rows.append(
            {
                "instance": instance,
                "k": INSTANCES[instance]["k"],
                "completed_runs": len(group),
                "best_conflicts": min(scores) if scores else None,
                "success_out_of_10": sum(int(row["success"]) for row in group),
                "mean_time_to_feasibility_s": (
                    statistics.fmean(successful) if successful else None
                ),
                "mean_conflicts": statistics.fmean(scores) if scores else None,
                "std_conflicts": statistics.stdev(scores) if len(scores) > 1 else 0.0,
                "mean_handoff_conflicts": (
                    statistics.fmean(float(row["handoff_conflicts"]) for row in group)
                    if group else None
                ),
            }
        )
    write_csv(
        root / "per_instance_summary.csv",
        summary_rows,
        [
            "instance",
            "k",
            "completed_runs",
            "best_conflicts",
            "success_out_of_10",
            "mean_time_to_feasibility_s",
            "mean_conflicts",
            "std_conflicts",
            "mean_handoff_conflicts",
        ],
    )


def main() -> int:
    args = parse_args()
    instances = selected_instances(args.instances)
    seeds = list(range(args.seed_start, args.seed_start + args.num_seeds))
    args.output_root = args.output_root.resolve()
    args.output_root.mkdir(parents=True, exist_ok=True)
    preflight(args, instances)

    protocol = {
        "protocol_version": PROTOCOL_VERSION,
        "run_type": (
            "smoke" if args.smoke_rlaco_time_limit is not None else "formal"
        ),
        "method": "RLACO+TerminalTabucol",
        "instances": instances,
        "fixed_k": {name: INSTANCES[name]["k"] for name in instances},
        "rlaco_time_limit_s": {
            name: (
                args.smoke_rlaco_time_limit
                if args.smoke_rlaco_time_limit is not None
                else INSTANCES[name]["rlaco_time"]
            )
            for name in instances
        },
        "seeds": seeds,
        "policy_file": args.policy.name,
        "policy_sha256": file_sha256(args.policy.resolve()),
        "rl_steps_per_episode": args.rl_steps,
        "keep_rl_best": False,
        "aco_ants": args.aco_ants,
        "aco_iterations_per_episode": args.aco_iter,
        "aco_fixed_parameters": ACO_FIXED_PARAMETERS,
        "aco_injection": "learned-policy soft prior",
        "per_episode_tabucol_polish": False,
        "terminal_time_limit_s": args.terminal_time_limit,
        "terminal_tabucol_iterations_per_block": args.tabu_iters,
        "tabu_a": args.tabu_a,
        "tabu_alpha": args.tabu_alpha,
        "tabucol_return_beta": args.beta,
        "tabu_reset_each_block": True,
        "returned_solution_handoff": True,
        "external_global_best_retained": True,
        "deterministic_seed_per_terminal_block": True,
        "formal_native_tabucol_required": not args.allow_python_fallback,
    }
    protocol_path = args.output_root / "protocol.json"
    if protocol_path.exists():
        previous = json.loads(protocol_path.read_text(encoding="utf-8"))
        if previous != protocol:
            raise RuntimeError(
                "Output directory contains a different protocol. Use a new "
                "--output-root instead of mixing experiments."
            )
    else:
        atomic_json(protocol_path, protocol)

    if not args.summarize_only:
        jobs = [(instance, seed) for instance in instances for seed in seeds]
        with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
            futures = {
                pool.submit(run_one, args, instance, seed): (instance, seed)
                for instance, seed in jobs
            }
            for future in as_completed(futures):
                requested_instance, requested_seed = futures[future]
                try:
                    instance, seed, code = future.result()
                except Exception as exc:
                    print(
                        f"[ERROR] {requested_instance} seed={requested_seed}: {exc}",
                        flush=True,
                    )
                    continue
                if code not in (0, 130):
                    print(
                        f"[ERROR] {instance} seed={seed} exited {code}; inspect logs.",
                        flush=True,
                    )
                summarize(args.output_root, instances, seeds)

    summarize(args.output_root, instances, seeds)
    print(f"[DONE] Results and summaries: {args.output_root}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
