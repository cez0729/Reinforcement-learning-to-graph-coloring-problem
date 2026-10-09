# Professor Protocol Audit

## Experimental question

Does a complete learned-policy RLACO run produce a global-best coloring that
becomes a stronger starting point for a standalone terminal Tabucol search?

This experiment compares fixed-k feasibility and residual conflicts. It does
not search progressively for a smaller k inside one run.

## Requirement-to-code mapping

| Requirement | Implementation | Verification evidence |
|---|---|---|
| Use the same 10 RLACO instances | Locked `INSTANCES` table in `run_campaign.py` | Unit test checks table length and selected values |
| Use the paper's fixed k | k is stored beside every graph and cannot be overridden per job | `protocol.json`; graph/k checks before terminal stage |
| 10 independent runs | Formal mode locks seeds 0-9 | Formal preflight rejects other seed counts/ranges |
| Keep RL phase unchanged | Trained `policy.pt`; 300 actions; `keep_rl_best=False` | Runner command and `protocol.json` |
| Use learned soft pheromone prior | `--RL True --use-aco True --inject-mode soft` | RLACO summary records mode and ACO trace |
| Keep ACO configuration unchanged | 40 ants, 50 iterations; ACO defaults are locked and tested | `verify_install.py` introspects constructor defaults |
| Run complete RLACO first | RLACO uses the instance-specific existing time budget | No terminal call exists inside the episode loop |
| Save the true cross-episode best | Runner stores score, coloring and episode together | Terminal stage recomputes score from DIMACS edges |
| Start Tabucol from that coloring | Terminal input is `rlaco_best_solution` | SHA-256 coloring digest protects checkpoint identity |
| Give terminal Tabucol the Pure Tabucol budget | 18,000 active seconds per run | Formal preflight locks this value |
| Use standard Tabucol blocks | T=10,000, A=10, alpha=1.2, beta=0.2 | Per-block CSV trace and protocol lock |
| Reset tabu state each block | A new solver is constructed and then seeded each block | `tabu_reset=1` in every trace row |
| Continue search between blocks | Tabucol's returned coloring seeds the next block | `returned_solution_handoff=True` |
| Do not lose an internal block best | `get_best_solution()` is retained separately | Score/coloring invariant checked after each block |
| Stop at feasibility | Terminal loop exits when global-best score is zero | Success and time-to-feasibility fields |
| Resume interrupted long runs | Atomic checkpoint after every completed block | Integration test covers checkpoint resume |
| Reject invalid timing backend | Formal mode requires compiled Rust/PyO3 Tabucol | Formal preflight intentionally fails on fallback/stale binary |

## Time-budget interpretation

The professor protocol is additive:

```text
instance-specific full RLACO budget + 18,000-second terminal Tabucol budget
```

The terminal stage is not limited to one call of 10,000 iterations. It makes as
many standard 10,000-iteration calls as fit before the terminal budget, stopping
early only at zero conflicts. Since a block is atomic, the measured terminal
runtime can exceed 18,000 seconds by the duration of the final block that began
before the limit. The exact elapsed time and every block duration are recorded.

## Two different beta parameters

- ACO heuristic exponent: `beta_ACO = 3.0` in `AntColony`.
- Tabucol return probability: `beta_Tabucol = 0.2` in the terminal solver.

The campaign passes `0.2` only to Tabucol behavior. It does not replace the ACO
heuristic exponent.

## Why the best coloring API is required

At `beta_Tabucol=0.2`, a Tabucol block returns its final current state with 20%
probability and its block-best state with 80% probability. Therefore one state
cannot safely serve both purposes:

- returned current coloring: starting point of the next block;
- block/global-best coloring: immutable reporting candidate.

The Rust extension now exposes `get_best_solution()` so every saved score always
has its corresponding coloring. The code recomputes conflicts from the graph and
stops immediately if the pair is inconsistent.
