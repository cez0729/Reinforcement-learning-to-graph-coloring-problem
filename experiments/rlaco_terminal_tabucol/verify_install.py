"""Fail-fast verification for the formal experiment environment."""

from __future__ import annotations

import inspect
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

import tabucol
from gcp_env.ACO import AntColony


def main() -> int:
    if not tabucol.IS_NATIVE:
        raise RuntimeError(
            "The Python fallback is active. Formal timing requires the native "
            "Rust/PyO3 Tabucol extension."
        )
    if not hasattr(tabucol.TabuColSolver, "get_best_solution"):
        raise RuntimeError("Native Tabucol is stale: get_best_solution() is missing")

    solver = tabucol.TabuColSolver([[]], 1, seed=0)
    if solver.get_best_solution() != [0] or solver.get_best_score() != 0:
        raise RuntimeError("Native Tabucol score/solution API check failed")

    defaults = {
        name: parameter.default
        for name, parameter in inspect.signature(AntColony.__init__).parameters.items()
    }
    expected = {
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
    }
    actual = {name: defaults[name] for name in expected}
    if actual != expected:
        raise RuntimeError(f"ACO defaults changed: {actual} != {expected}")

    print(f"Python: {sys.version.split()[0]}")
    print(f"Tabucol native: {tabucol.IS_NATIVE}")
    print(f"Tabucol backend: {tabucol.BACKEND_ORIGIN}")
    print("Tabucol best-solution API: OK")
    print("ACO fixed parameters: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
