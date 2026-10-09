"""Run the terminal Tabucol stage from a verified RLACO global-best coloring."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rlaco-result", type=Path, required=True)
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--time-limit", type=float, default=18000.0)
    parser.add_argument("--tabu-iters", type=int, default=10000)
    parser.add_argument("--tabu-a", type=int, default=10)
    parser.add_argument("--tabu-alpha", type=float, default=1.2)
    parser.add_argument("--beta", type=float, default=0.2)
    parser.add_argument("--allow-python-fallback", action="store_true")
    return parser.parse_args()


def read_dimacs(path: Path) -> list[list[int]]:
    n = None
    edges: list[tuple[int, int]] = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for raw in handle:
            fields = raw.strip().split()
            if not fields or fields[0] == "c":
                continue
            if fields[0] == "p":
                n = int(fields[2])
            elif fields[0] == "e":
                edges.append((int(fields[1]) - 1, int(fields[2]) - 1))
    if n is None:
        raise ValueError(f"Missing DIMACS problem line in {path}")
    adjacency = [[] for _ in range(n)]
    for u, v in edges:
        if not (0 <= u < n and 0 <= v < n):
            raise ValueError(f"Edge ({u + 1}, {v + 1}) is outside graph order {n}")
        adjacency[u].append(v)
        adjacency[v].append(u)
    return adjacency


def score(adjacency: list[list[int]], solution: list[int]) -> int:
    if len(solution) != len(adjacency):
        raise ValueError("Coloring length does not match graph order")
    doubled = sum(
        solution[u] == solution[v]
        for u, neighbors in enumerate(adjacency)
        for v in neighbors
    )
    return int(doubled // 2)


def validate_solution(solution: list[int], n: int, k: int) -> None:
    if len(solution) != n:
        raise RuntimeError(f"Expected {n} colors, received {len(solution)}")
    if any(color < 0 or color >= k for color in solution):
        raise RuntimeError(f"Coloring contains a value outside 0..{k - 1}")


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def solution_digest(solution: list[int]) -> str:
    payload = ",".join(str(value) for value in solution).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def block_seed(seed: int, block: int) -> int:
    return ((int(seed) & 0xFFFFFFFF) << 32) | (int(block) & 0xFFFFFFFF)


def main() -> int:
    args = parse_args()
    if args.k < 1 or args.time_limit <= 0 or args.tabu_iters < 1:
        raise ValueError("k, time-limit, and tabu-iters must be positive")
    if not 0.0 <= args.beta <= 1.0:
        raise ValueError("beta must be in [0, 1]")

    if args.allow_python_fallback:
        os.environ["RLTABUCOL_FORCE_PYTHON"] = "1"
    import tabucol

    if not tabucol.IS_NATIVE and not args.allow_python_fallback:
        raise RuntimeError(
            "Formal experiments require the compiled Rust/PyO3 Tabucol backend. "
            "Run the package setup script before launching the campaign."
        )
    if not hasattr(tabucol.TabuColSolver, "get_best_solution"):
        raise RuntimeError(
            "The Tabucol extension is stale. Rebuild it from the packaged Rust "
            "source so get_best_solution() is available."
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "summary.json"
    checkpoint_path = args.output_dir / "checkpoint.json"
    trace_path = args.output_dir / "block_trace.csv"
    if summary_path.exists():
        print(f"[SKIP] Completed terminal result exists: {summary_path}", flush=True)
        return 0

    adjacency = read_dimacs(args.graph.resolve())
    handoff = json.loads(args.rlaco_result.read_text(encoding="utf-8"))
    handoff_graph = Path(str(handoff["graph"])).name
    if handoff_graph != args.graph.name:
        raise RuntimeError(f"Graph mismatch: {handoff_graph} != {args.graph.name}")
    if int(handoff["k"]) != args.k:
        raise RuntimeError(f"k mismatch: {handoff['k']} != {args.k}")
    if int(handoff.get("seed", args.seed)) != args.seed:
        raise RuntimeError(f"seed mismatch: {handoff.get('seed')} != {args.seed}")

    handoff_solution = [int(value) for value in handoff["rlaco_best_solution"]]
    validate_solution(handoff_solution, len(adjacency), args.k)
    handoff_score = score(adjacency, handoff_solution)
    if handoff_score != int(handoff["rlaco_best_score"]):
        raise RuntimeError(
            "RLACO handoff score/solution mismatch: "
            f"reported={handoff['rlaco_best_score']}, recomputed={handoff_score}"
        )
    handoff_digest = solution_digest(handoff_solution)

    random.seed(args.seed)
    np.random.seed(args.seed)
    elapsed_before = 0.0
    blocks = 0
    current_solution = list(handoff_solution)
    global_best_solution = list(handoff_solution)
    global_best_score = handoff_score
    best_block = 0
    total_iterations = 0
    time_to_feasibility = 0.0 if handoff_score == 0 else None

    if checkpoint_path.exists():
        saved = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        expected = (
            str(args.graph.resolve()),
            args.k,
            args.seed,
            handoff_score,
            handoff_digest,
        )
        actual = (
            saved["graph"],
            int(saved["k"]),
            int(saved["seed"]),
            int(saved["handoff_score"]),
            saved["handoff_digest"],
        )
        if actual != expected:
            raise RuntimeError(f"Checkpoint protocol mismatch: {actual} != {expected}")
        elapsed_before = float(saved["elapsed_s"])
        blocks = int(saved["blocks"])
        current_solution = [int(v) for v in saved["current_solution"]]
        global_best_solution = [int(v) for v in saved["global_best_solution"]]
        global_best_score = int(saved["global_best_score"])
        best_block = int(saved["best_block"])
        total_iterations = int(saved["total_iterations"])
        time_to_feasibility = saved.get("time_to_feasibility_s")
        print(
            f"[RESUME] blocks={blocks} elapsed={elapsed_before:.3f}s "
            f"global_best={global_best_score}",
            flush=True,
        )

    validate_solution(current_solution, len(adjacency), args.k)
    validate_solution(global_best_solution, len(adjacency), args.k)
    if score(adjacency, global_best_solution) != global_best_score:
        raise RuntimeError("Checkpoint global-best score/solution mismatch")

    session_start = time.perf_counter()

    def elapsed() -> float:
        return elapsed_before + (time.perf_counter() - session_start)

    def save_checkpoint() -> None:
        atomic_json(
            checkpoint_path,
            {
                "graph": str(args.graph.resolve()),
                "k": args.k,
                "seed": args.seed,
                "handoff_score": handoff_score,
                "handoff_digest": handoff_digest,
                "elapsed_s": elapsed(),
                "blocks": blocks,
                "current_solution": current_solution,
                "global_best_solution": global_best_solution,
                "global_best_score": global_best_score,
                "best_block": best_block,
                "total_iterations": total_iterations,
                "time_to_feasibility_s": time_to_feasibility,
            },
        )

    trace_exists = trace_path.exists() and trace_path.stat().st_size > 0
    with trace_path.open("a", newline="", encoding="utf-8") as trace_handle:
        trace = csv.writer(trace_handle)
        if not trace_exists:
            trace.writerow(
                [
                    "block",
                    "elapsed_s",
                    "start_score",
                    "returned_score",
                    "block_best_score",
                    "global_best_score",
                    "iterations",
                    "block_seed",
                    "tabu_reset",
                ]
            )
            trace_handle.flush()

        print(
            f"[START] graph={args.graph.name} k={args.k} seed={args.seed} "
            f"handoff={handoff_score} backend={tabucol.BACKEND_ORIGIN} "
            f"T={args.tabu_iters} A={args.tabu_a} alpha={args.tabu_alpha} "
            f"beta={args.beta} budget={args.time_limit}s",
            flush=True,
        )
        try:
            while global_best_score != 0 and elapsed() < args.time_limit:
                blocks += 1
                start_score = score(adjacency, current_solution)
                current_block_seed = block_seed(args.seed, blocks)
                solver = tabucol.TabuColSolver(
                    adjacency,
                    args.k,
                    max_iterations=args.tabu_iters,
                    tabu_a=args.tabu_a,
                    tabu_alpha=args.tabu_alpha,
                    beta=args.beta,
                    seed=current_block_seed,
                )
                solver.set_solution(current_solution)
                returned, _ = solver.solve()
                returned_solution = [int(value) for value in returned]
                block_best_solution = [
                    int(value) for value in solver.get_best_solution()
                ]
                validate_solution(returned_solution, len(adjacency), args.k)
                validate_solution(block_best_solution, len(adjacency), args.k)
                returned_score = score(adjacency, returned_solution)
                block_best_score = score(adjacency, block_best_solution)
                solver_best_score = int(solver.get_best_score())
                if block_best_score != solver_best_score:
                    raise RuntimeError(
                        "Tabucol block best mismatch: "
                        f"reported={solver_best_score}, recomputed={block_best_score}"
                    )

                # Preserve beta=0.2 handoff exactly: the returned coloring,
                # which can differ from the block best, seeds the next block.
                current_solution = returned_solution
                if block_best_score < global_best_score:
                    global_best_score = block_best_score
                    global_best_solution = block_best_solution
                    best_block = blocks
                total_iterations += int(solver.get_iters())
                if global_best_score == 0 and time_to_feasibility is None:
                    time_to_feasibility = elapsed()

                trace.writerow(
                    [
                        blocks,
                        f"{elapsed():.6f}",
                        start_score,
                        returned_score,
                        block_best_score,
                        global_best_score,
                        int(solver.get_iters()),
                        current_block_seed,
                        1,
                    ]
                )
                trace_handle.flush()
                save_checkpoint()
                print(
                    f"block={blocks} start={start_score} returned={returned_score} "
                    f"block_best={block_best_score} global_best={global_best_score} "
                    f"elapsed={elapsed():.3f}s",
                    flush=True,
                )
        except KeyboardInterrupt:
            save_checkpoint()
            print("[INTERRUPTED] Terminal checkpoint saved.", flush=True)
            return 130

    final_elapsed = elapsed()
    save_checkpoint()
    result = {
        "method": "RLACO+TerminalTabucol",
        "graph": str(args.graph.resolve()),
        "instance": args.graph.stem,
        "k": args.k,
        "seed": args.seed,
        "success": global_best_score == 0,
        "score": global_best_score,
        "handoff_score": handoff_score,
        "handoff_digest": handoff_digest,
        "rlaco_elapsed_s": float(handoff["rlaco_elapsed"]),
        "rlaco_best_episode": int(handoff["rlaco_best_episode"]),
        "terminal_elapsed_s": final_elapsed,
        "terminal_time_to_feasibility_s": time_to_feasibility,
        "end_to_end_time_to_feasibility_s": (
            float(handoff["rlaco_elapsed"]) + float(time_to_feasibility)
            if time_to_feasibility is not None else None
        ),
        "blocks": blocks,
        "best_block": best_block,
        "total_tabucol_iterations": total_iterations,
        "tabucol_iterations_per_block": args.tabu_iters,
        "tabu_a": args.tabu_a,
        "tabu_alpha": args.tabu_alpha,
        "tabucol_return_beta": args.beta,
        "tabu_reset_each_block": True,
        "returned_solution_handoff": True,
        "external_global_best_retained": True,
        "deterministic_seed_per_block": True,
        "terminal_time_limit_s": args.time_limit,
        "tabucol_native": bool(tabucol.IS_NATIVE),
        "tabucol_backend": tabucol.BACKEND_ORIGIN,
        "global_best_solution": global_best_solution,
        "final_returned_solution": current_solution,
        "rlaco_result": str(args.rlaco_result.resolve()),
        "trace": str(trace_path.resolve()),
    }
    if score(adjacency, result["global_best_solution"]) != result["score"]:
        raise RuntimeError("Final global-best score/solution invariant failed")
    atomic_json(summary_path, result)
    print(
        f"[DONE] success={result['success']} score={result['score']} "
        f"terminal_elapsed={final_elapsed:.3f}s",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
