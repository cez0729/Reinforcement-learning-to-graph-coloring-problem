from __future__ import annotations

import inspect
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


TEST_DIR = Path(__file__).resolve().parent
SCRIPT_DIR = TEST_DIR.parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(SRC_DIR))

from experiments.rlaco_terminal_tabucol import run_campaign
from experiments.rlaco_terminal_tabucol.terminal_tabucol import block_seed
from gcp_env.ACO import AntColony


class ProtocolTests(unittest.TestCase):
    def test_fixed_instance_table(self):
        self.assertEqual(len(run_campaign.INSTANCES), 10)
        self.assertEqual(run_campaign.INSTANCES["flat300_28_0"]["k"], 28)
        self.assertEqual(run_campaign.INSTANCES["DSJC500.9"]["rlaco_time"], 21600)

    def test_aco_defaults_match_locked_protocol(self):
        signature = inspect.signature(AntColony.__init__)
        for name, expected in run_campaign.ACO_FIXED_PARAMETERS.items():
            if name == "injection_strength":
                continue
            self.assertEqual(signature.parameters[name].default, expected)

    def test_terminal_block_seed_is_stable_and_unique(self):
        self.assertEqual(block_seed(3, 2), (3 << 32) | 2)
        self.assertNotEqual(block_seed(3, 2), block_seed(3, 3))
        self.assertNotEqual(block_seed(3, 2), block_seed(4, 2))


class TerminalIntegrationTests(unittest.TestCase):
    def test_fallback_smoke_and_checkpoint_resume(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            graph = root / "edge.col"
            graph.write_text("p edge 2 1\ne 1 2\n", encoding="ascii")
            handoff = root / "rlaco_summary.json"
            handoff.write_text(
                json.dumps(
                    {
                        "graph": str(graph),
                        "k": 1,
                        "seed": 7,
                        "rlaco_best_solution": [0, 0],
                        "rlaco_best_score": 1,
                        "rlaco_elapsed": 1.25,
                        "rlaco_best_episode": 1,
                    }
                ),
                encoding="utf-8",
            )
            output = root / "terminal"
            command = [
                sys.executable,
                str(SCRIPT_DIR / "terminal_tabucol.py"),
                "--rlaco-result",
                str(handoff),
                "--graph",
                str(graph),
                "--k",
                "1",
                "--seed",
                "7",
                "--output-dir",
                str(output),
                "--time-limit",
                "0.02",
                "--tabu-iters",
                "25",
                "--allow-python-fallback",
            ]
            environment = os.environ.copy()
            environment["RLTABUCOL_FORCE_PYTHON"] = "1"
            first = subprocess.run(
                command,
                cwd=PROJECT_ROOT,
                env=environment,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(first.returncode, 0, first.stdout + first.stderr)
            summary_path = output / "summary.json"
            first_summary = json.loads(summary_path.read_text(encoding="utf-8"))
            self.assertGreater(first_summary["blocks"], 0)
            self.assertEqual(first_summary["score"], 1)
            self.assertEqual(first_summary["global_best_solution"], [0, 0])
            self.assertTrue(first_summary["external_global_best_retained"])

            blocks_before_resume = first_summary["blocks"]
            summary_path.unlink()
            resumed = subprocess.run(
                command,
                cwd=PROJECT_ROOT,
                env=environment,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(resumed.returncode, 0, resumed.stdout + resumed.stderr)
            resumed_summary = json.loads(summary_path.read_text(encoding="utf-8"))
            self.assertEqual(resumed_summary["blocks"], blocks_before_resume)
            self.assertIn("[RESUME]", resumed.stdout)


if __name__ == "__main__":
    unittest.main()
