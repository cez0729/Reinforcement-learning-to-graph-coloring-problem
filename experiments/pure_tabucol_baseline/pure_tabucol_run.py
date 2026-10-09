"""Run one episodic pure-Tabucol baseline trial.

No RL policy or Gym environment is created.  The coloring returned by one
Tabucol call is passed to the next call, while set_solution() resets the
iteration counter and tabu matrix before every call.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import random
import sys
import time
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
SRC_DIR = PROJECT_ROOT / "src"
NATIVE_TABUCOL_DIR = SRC_DIR / "tabucol" / "python"


def load_tabucol(allow_python_fallback: bool):
    # Prefer the same compiled Rust/PyO3 implementation used by the paper.
    sys.path.insert(0, str(NATIVE_TABUCOL_DIR))
    sys.path.append(str(SRC_DIR))
    try:
        import tabucol  # type: ignore
    except (ImportError, ModuleNotFoundError):
        if not allow_python_fallback:
            raise RuntimeError(
                "The compiled tabucol extension is unavailable for this Python. "
                "Use Python 3.13 with src/tabucol/python/tabucol/tabucol.cp313-"
                "win_amd64.pyd, or pass --allow-python-fallback only for a smoke test."
            ) from None
        sys.path.remove(str(NATIVE_TABUCOL_DIR))
        sys.modules.pop("tabucol", None)
        import tabucol  # type: ignore

    extension_spec = importlib.util.find_spec("tabucol.tabucol")
    extension_origin = str(extension_spec.origin) if extension_spec else ""
    native = extension_origin.lower().endswith((".pyd", ".so"))
    if not native and not allow_python_fallback:
        raise RuntimeError(
            "A Python Tabucol fallback was loaded. Formal timing experiments "
            "require the compiled Rust/PyO3 extension."
        )
    return tabucol, native, extension_origin or str(getattr(tabucol, "__file__", ""))


def read_dimacs_graph(path: Path) -> tuple[list[list[int]], int]:
    declared_n = None
    edges: list[tuple[int, int]] = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("c"):
                continue
            fields = line.split()
            if fields[0] == "p":
                if len(fields) < 4:
                    raise ValueError(f"Malformed DIMACS problem line: {line}")
                declared_n = int(fields[2])
            elif fields[0] == "e":
                if len(fields) < 3:
                    raise ValueError(f"Malformed DIMACS edge line: {line}")
                u, v = int(fields[1]) - 1, int(fields[2]) - 1
                edges.append((u, v))

    if declared_n is None:
        raise ValueError(f"No DIMACS problem line found in {path}")
    adjacency = [[] for _ in range(declared_n)]
    for u, v in edges:
        if not (0 <= u < declared_n and 0 <= v < declared_n):
            raise ValueError(f"Edge ({u + 1}, {v + 1}) is outside 1..{declared_n}")
        adjacency[u].append(v)
        adjacency[v].append(u)
    return adjacency, len(edges)


def conflict_score(adjacency: list[list[int]], solution: list[int]) -> int:
    if len(solution) != len(adjacency):
        raise ValueError("Solution length does not match graph order")
    doubled = sum(
        solution[node] == solution[neighbor]
        for node, neighbors in enumerate(adjacency)
        for neighbor in neighbors
    )
    return int(doubled // 2)


def atomic_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def prepare_trace_for_resume(path: Path, checkpoint_episodes: int) -> None:
    """Remove rows written after the last durable checkpoint.

    A process can stop after flushing trace rows but before its next checkpoint.
    Keeping those rows and then resuming from the older checkpoint duplicates
    episode numbers.  The latest row for each episode is retained and the trace
    is truncated to the checkpoint boundary before appending new work.
    """
    if not path.exists() or path.stat().st_size == 0:
        return
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames
        if not fieldnames:
            return
        rows_by_episode = {
            int(row["episode"]): row
            for row in reader
            if int(row["episode"]) <= checkpoint_episodes
        }
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for episode in sorted(rows_by_episode):
            writer.writerow(rows_by_episode[episode])
    temporary.replace(path)


def validate_solution(solution: list[int], n: int, k: int) -> None:
    if len(solution) != n:
        raise RuntimeError(f"Tabucol returned {len(solution)} colors for {n} vertices")
    if any(color < 0 or color >= k for color in solution):
        raise RuntimeError(f"Tabucol returned a color outside 0..{k - 1}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--k", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--time-limit", type=float, default=18000.0)
    parser.add_argument("--tabu-iters", type=int, default=10000)
    parser.add_argument("--tabu-a", type=int, default=10)
    parser.add_argument("--tabu-alpha", type=float, default=1.2)
    parser.add_argument("--beta", type=float, default=0.2)
    parser.add_argument("--checkpoint-every", type=int, default=10)
    parser.add_argument("--allow-python-fallback", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.k < 1 or args.time_limit <= 0 or args.tabu_iters < 1:
        raise ValueError("k, time-limit, and tabu-iters must be positive")
    if not 0.0 <= args.beta <= 1.0:
        raise ValueError("beta must be in [0, 1]")

    graph_path = args.graph.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "summary.json"
    checkpoint_path = args.output_dir / "checkpoint.json"
    trace_path = args.output_dir / "episode_trace.csv"

    if summary_path.exists():
        print(f"[SKIP] Completed result exists: {summary_path}", flush=True)
        return 0

    tabucol, native_backend, backend_origin = load_tabucol(args.allow_python_fallback)
    adjacency, edge_count = read_dimacs_graph(graph_path)

    random.seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)

    elapsed_before = 0.0
    episodes = 0
    best_episode = 0
    current_solution = rng.integers(0, args.k, size=len(adjacency)).astype(int).tolist()
    best_solution = list(current_solution)
    best_score = conflict_score(adjacency, current_solution)
    total_tabucol_iterations = 0
    resumed_from_checkpoint = False

    if checkpoint_path.exists():
        resumed_from_checkpoint = True
        saved = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        expected = (str(graph_path), args.k, args.seed)
        actual = (saved["graph"], saved["k"], saved["seed"])
        if actual != expected:
            raise RuntimeError(f"Checkpoint configuration mismatch: {actual} != {expected}")
        elapsed_before = float(saved["elapsed_s"])
        episodes = int(saved["episodes"])
        best_episode = int(saved["best_episode"])
        current_solution = [int(value) for value in saved["current_solution"]]
        best_solution = [int(value) for value in saved["best_solution"]]
        best_score = int(saved["best_score"])
        total_tabucol_iterations = int(saved.get("total_tabucol_iterations", 0))
        print(
            f"[RESUME] episode={episodes}, elapsed={elapsed_before:.3f}s, "
            f"best_score={best_score}",
            flush=True,
        )

    if resumed_from_checkpoint:
        prepare_trace_for_resume(trace_path, episodes)

    solver = tabucol.TabuColSolver(
        adjacency,
        args.k,
        max_iterations=args.tabu_iters,
        tabu_a=args.tabu_a,
        tabu_alpha=args.tabu_alpha,
        beta=args.beta,
    )

    trace_exists = trace_path.exists() and trace_path.stat().st_size > 0
    trace_handle = trace_path.open("a", newline="", encoding="utf-8")
    trace = csv.writer(trace_handle)
    if not trace_exists:
        trace.writerow(
            [
                "episode",
                "elapsed_s",
                "start_score",
                "returned_score",
                "tabucol_best_score",
                "running_best_score",
                "tabucol_iterations",
                "tabu_reset",
            ]
        )
        trace_handle.flush()

    session_start = time.perf_counter()

    def total_elapsed() -> float:
        return elapsed_before + (time.perf_counter() - session_start)

    def save_checkpoint() -> None:
        atomic_json(
            checkpoint_path,
            {
                "graph": str(graph_path),
                "k": args.k,
                "seed": args.seed,
                "elapsed_s": total_elapsed(),
                "episodes": episodes,
                "best_episode": best_episode,
                "best_score": best_score,
                "total_tabucol_iterations": total_tabucol_iterations,
                "current_solution": current_solution,
                "best_solution": best_solution,
            },
        )

    print(
        f"[START] graph={graph_path.name} k={args.k} seed={args.seed} "
        f"backend={'native' if native_backend else 'python-fallback'} "
        f"T={args.tabu_iters} A={args.tabu_a} alpha={args.tabu_alpha} "
        f"beta={args.beta} budget={args.time_limit}s",
        flush=True,
    )

    try:
        while best_score != 0 and total_elapsed() < args.time_limit:
            episodes += 1
            start_score = conflict_score(adjacency, current_solution)

            # This call resets tabu[][] and iters to zero but preserves the
            # inter-episode coloring supplied by the preceding solve().
            solver.set_solution(current_solution)
            returned_solution, _ = solver.solve()
            current_solution = [int(value) for value in returned_solution]
            validate_solution(current_solution, len(adjacency), args.k)
            returned_score = conflict_score(adjacency, current_solution)
            tabucol_best_score = int(solver.get_best_score())
            episode_iterations = int(solver.get_iters())
            total_tabucol_iterations += episode_iterations

            if returned_score < best_score:
                best_score = returned_score
                best_solution = list(current_solution)
                best_episode = episodes

            trace.writerow(
                [
                    episodes,
                    f"{total_elapsed():.6f}",
                    start_score,
                    returned_score,
                    tabucol_best_score,
                    best_score,
                    episode_iterations,
                    1,
                ]
            )
            trace_handle.flush()

            if episodes % max(args.checkpoint_every, 1) == 0 or best_score == 0:
                save_checkpoint()
            if episodes == 1 or episodes % 100 == 0 or best_score == 0:
                print(
                    f"episode={episodes} start={start_score} returned={returned_score} "
                    f"tabucol_best={tabucol_best_score} running_best={best_score} "
                    f"elapsed={total_elapsed():.3f}s",
                    flush=True,
                )
    except KeyboardInterrupt:
        save_checkpoint()
        print("[INTERRUPTED] Checkpoint saved.", flush=True)
        return 130
    finally:
        trace_handle.close()

    elapsed = total_elapsed()
    save_checkpoint()
    result = {
        "method": "PureTabucol-episodic-reset",
        "graph": str(graph_path),
        "instance": graph_path.stem,
        "vertices": len(adjacency),
        "edges": edge_count,
        "k": args.k,
        "seed": args.seed,
        "score": best_score,
        "success": best_score == 0,
        "time_s": elapsed,
        "episodes": episodes,
        "best_episode": best_episode,
        "tabucol_iterations_per_episode": args.tabu_iters,
        "total_tabucol_iterations": total_tabucol_iterations,
        "tabu_a": args.tabu_a,
        "tabu_alpha": args.tabu_alpha,
        "tabucol_return_beta": args.beta,
        "tabu_reset_each_episode": True,
        "solution_handoff_each_episode": True,
        "rl_used": False,
        "time_limit_s": args.time_limit,
        "native_tabucol": native_backend,
        "tabucol_backend": backend_origin,
        "seed_scope": "initial coloring and Python/NumPy RNG",
        "native_internal_rng_seedable": False if native_backend else True,
        "best_solution": best_solution,
        "final_solution": current_solution,
        "trace": str(trace_path.resolve()),
    }
    atomic_json(summary_path, result)
    print(
        f"[DONE] score={best_score} success={best_score == 0} "
        f"episodes={episodes} time={elapsed:.3f}s summary={summary_path}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
