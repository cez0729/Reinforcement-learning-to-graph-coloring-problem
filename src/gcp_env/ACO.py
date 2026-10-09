"""
Pure ACO for fixed-k graph coloring (no local search).

Hard vs soft:
- Hard prior: commits to one color per vertex (argmax of policy).
  Pheromone at the chosen color is boosted to tau_0 * strength.
- Soft prior: keeps the policy's per-vertex color distribution.
  Pheromone is interpolated tau_0 -> tau_0*strength via dist[v,c].
  Confident positions follow the policy; uncertain positions explore freely.

every ant constructs a solution from scratch using pheromone +
heuristic; the policy's signal is delivered only via the pheromone matrix
and the saved best_solution. No ant copies the seed solution and no
local-search repair is applied.

The objective is fixed-k conflict minimization:
    score = number of conflicting edges

Flow: construct solution -> calculate conflicts -> update pheromone -> next iteration
"""

import time
import numpy as np
import networkx as nx


class AntColony:
    def __init__(
        self,
        graph,
        max_colors,
        alpha=1.0,
        beta=3.0,
        rho=0.20,
        q=1.0,
        tau0=1.0,
        tau_min=0.05,
        tau_max=8.0,
        q0=0.85,
        elite_count=3,
        global_best_weight=2.0,
        seed=None,
    ):
        """
        Args:
            graph:
                NetworkX graph. This implementation assumes nodes are 0..n-1,
                which matches your DIMACS reader.
            max_colors:
                Fixed number of colors k.
            alpha:
                Pheromone importance.
            beta:
                Heuristic importance.
            rho:
                Pheromone evaporation rate.
            q:
                Pheromone deposit scale.
            tau0:
                Initial pheromone value.
            tau_min, tau_max:
                Pheromone clipping range.
            q0:
                Greedy probability in pseudo-random proportional rule.
                Higher means more exploitation.
            elite_count:
                Number of iteration-best ants used to update pheromone.
            global_best_weight:
                Extra pheromone weight for the global best solution.
            seed:
                Random seed.
        """
        self.graph = graph
        self.max_colors = int(max_colors)
        self.k = int(max_colors)
        self.num_vertices = len(graph)
        self.n = self.num_vertices

        self.alpha = float(alpha)
        self.beta = float(beta)
        self.rho = float(rho)
        self.Q = float(q)
        self.tau0 = float(tau0)
        self.tau_min = float(tau_min)
        self.tau_max = float(tau_max)
        self.q0 = float(q0)

        self.elite_count = int(elite_count)
        self.global_best_weight = float(global_best_weight)

        self.rng = np.random.default_rng(seed)

        # This project's DIMACS reader creates nodes 0..n-1.
        # Keep this explicit because Gym env indexes solution[node].
        expected_nodes = set(range(self.n))
        actual_nodes = set(graph.nodes())
        if actual_nodes != expected_nodes:
            raise ValueError(
                "AntColony expects graph nodes to be 0..n-1. "
                "Please relabel the graph before calling ACO."
            )

        self.adj_list = [
            np.fromiter(graph.neighbors(v), dtype=np.int32)
            for v in range(self.n)
        ]

        self.edges = np.array(
            [(u, v) for u, v in graph.edges() if u != v],
            dtype=np.int32,
        )
        if len(self.edges) > 0:
            self.edges.sort(axis=1)
            self.edges = np.unique(self.edges, axis=0)

        self.degrees = np.array([graph.degree(v) for v in range(self.n)], dtype=np.int32)
        self.vertex_order = np.argsort(-self.degrees)

        self.Tau = np.full((self.n, self.k), self.tau0, dtype=np.float64)

        self.best_solution = None
        self.best_conflicts = float("inf")

        self.seed_solution = None
        self.seed_conflicts = None

        self.iter_x = []
        self.iter_y = []

    # ---------------------------------------------------------------------
    # Basic utilities
    # ---------------------------------------------------------------------

    def reset_pheromones(self):
        self.Tau.fill(self.tau0)
        self.best_solution = None
        self.best_conflicts = float("inf")
        self.seed_solution = None
        self.seed_conflicts = None
        self.iter_x = []
        self.iter_y = []

    def calculate_conflicts(self, solution):
        solution = np.asarray(solution, dtype=np.int32)
        if len(self.edges) == 0:
            return 0
        endpoint_colors = solution[self.edges]
        return int(np.sum(endpoint_colors[:, 0] == endpoint_colors[:, 1]))
 
    # ---------------------------------------------------------------------
    # RL  interface
    # ---------------------------------------------------------------------

    def bias_from_solution(self, solution, strength=8.0, reset_tau=True):
        """
        Hard prior injection: bias pheromone from a concrete solution.

        Each vertex v has its pheromone at the chosen color solution[v]
        boosted to tau_0 * strength (clipped by tau_max); other colors
        stay at tau_0 (if reset_tau=True).

        Args:
            solution:
                np.ndarray of shape (n,), values in [0, k-1].
            strength:
                How strongly ACO prefers the provided color assignment.
            reset_tau:
                If True, reset pheromone to tau_0 before boosting.
                If False, add boost on top of current Tau (progressive injection).
        """
        solution = np.asarray(solution, dtype=np.int32).copy()
        if solution.shape != (self.n,):
            raise ValueError(
                f"bias_from_solution: expected shape ({self.n},), got {solution.shape}"
            )
        if solution.min () < 0 or solution.max() >= self.k:
            raise ValueError(
                f"bias_from_solution: solution has colors outside [0, {self.k-1}]"
            )

        if reset_tau:
            self.Tau.fill(self.tau0)

        boosted = min(self.tau_max, self.tau0 * float(strength))
        for v in range(self.n):
            self.Tau[v, solution[v]] = boosted

        self.seed_solution = solution.copy()
        self.seed_conflicts = self.calculate_conflicts(solution)

        self.best_solution = solution.copy()
        self.best_conflicts = int(self.seed_conflicts)

        self.iter_x = []
        self.iter_y = []

    def bias_from_distribution(self, dist, solution=None, strength=8.0, reset_tau=True):
        """
        Soft prior injection: bias pheromone from a per-vertex color distribution.

        dist[v, c] is the probability (or attractiveness) that vertex v should
        be color c under the external predictor. Pheromone is set to:

            Tau[v, c] = tau_0 + (boosted - tau_0) * dist[v, c]

        where boosted = min(tau_max, tau_0 * strength). So the most confident
        positions approach the hard-prior boost, while uncertain positions
        remain near tau_0 (free exploration).

        This is the softer, ML-ACO-style injection (Sun et al. 2022, COR).
        It preserves the policy's uncertainty instead of committing to argmax.

        Args:
            dist:
                np.ndarray of shape (n, k). Each row should be a probability
                vector (non-negative, sums to ~1). Will be normalised if not.
            solution:
                np.ndarray of shape (n,), the concrete RL solution used as
                best/seed baseline. If None, argmax(dist) is used as a fallback,
                but this is NOT recommended because argmax(dist) is generally
                NOT the same as the RL-produced solution and may be much worse.
            strength:
                How strongly the most confident (v, c) entries are boosted.
            reset_tau:
                If True, reset pheromone to tau_0 before the additive boost.
        """
        dist = np.asarray(dist, dtype=np.float64)
        if dist.shape != (self.n, self.k):
            raise ValueError(
                f"bias_from_distribution: expected shape ({self.n}, {self.k}), got {dist.shape}"
            )
        if np.any(dist < 0):
            raise ValueError("bias_from_distribution: distribution has negative entries")

        # Normalise each row to a probability vector (tolerate un-normalised input).
        row_sums = dist.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums <= 0, 1.0, row_sums)
        dist = dist / row_sums

        if reset_tau:
            self.Tau.fill(self.tau0)

        boosted = min(self.tau_max, self.tau0 * float(strength))
        # Soft boost: uncertain rows stay near tau_0, confident rows approach `boosted`.
        self.Tau += (boosted - self.tau0) * dist
        np.clip(self.Tau, self.tau_min, self.tau_max, out=self.Tau)

        # Seed/best baseline: prefer the explicit RL solution (typically much
        # better than argmax(dist) when the policy's distribution is shaped by
        # "next-action" attractiveness rather than per-vertex color preference).
        if solution is not None:
            sol = np.asarray(solution, dtype=np.int32).copy()
            if sol.shape != (self.n,):
                raise ValueError(
                    f"bias_from_distribution: solution shape mismatch, got {sol.shape}"
                )
            if sol.min() < 0 or sol.max() >= self.k:
                raise ValueError(
                    f"bias_from_distribution: solution has colors outside [0, {self.k-1}]"
                )
        else:
            sol = np.argmax(dist, axis=1).astype(np.int32)

        self.seed_solution = sol.copy()
        self.seed_conflicts = self.calculate_conflicts(sol)

        self.best_solution = sol.copy()
        self.best_conflicts = int(self.seed_conflicts)

        self.iter_x = []
        self.iter_y = []

    # ---------------------------------------------------------------------
    # Constructive ACO
    # ---------------------------------------------------------------------

    def _make_processing_order(self):
        """
        Mostly degree-descending, with random noise.
        Hard GCP instances benefit from not using exactly the same order
        for every ant.
        """
        if self.n <= 1:
            return self.vertex_order.copy()

        noise_scale = max(1.0, float(np.std(self.degrees)))
        noise = self.rng.normal(0.0, noise_scale * 0.15, size=self.n)
        priority = self.degrees.astype(np.float64) + noise  
        return np.argsort(-priority)

    def _choose_color(self, v, partial_counts, color_sizes):
        """
        Choose a color for vertex v during construction.

        partial_counts[v, c] is the number of already-colored neighbors
        of v that have color c.

        Heuristic:
            fewer immediate conflicts is better.
            slightly prefer smaller color classes for balance.
        """
        conflicts_by_color = partial_counts[v].astype(np.float64)

        # Main graph-coloring heuristic: fewer conflicts.
        eta_conflict = 1.0 / (1.0 + conflicts_by_color)

        # Mild balance term. This prevents very large color classes
        # in early construction but does not dominate conflict minimization.
        eta_balance = 1.0 / np.sqrt(1.0 + color_sizes.astype(np.float64))

        tau = self.Tau[v]
        weights = (tau ** self.alpha) * (eta_conflict ** self.beta) * eta_balance

        if not np.all(np.isfinite(weights)) or weights.sum() <= 0:
            return int(self.rng.integers(0, self.k))

        if self.rng.random() < self.q0:
            max_w = np.max(weights)
            best_colors = np.flatnonzero(weights >= max_w - 1e-12)
            return int(self.rng.choice(best_colors))

        probs = weights / weights.sum()
        return int(self.rng.choice(self.k, p=probs))

    def _construct_solution(self):
        """
        Construct a complete fixed-k coloring using pheromone + heuristic.
        """
        solution = np.full(self.n, -1, dtype=np.int32)
        partial_counts = np.zeros((self.n, self.k), dtype=np.int16)
        color_sizes = np.zeros(self.k, dtype=np.int32)

        order = self._make_processing_order()

        for v in order:
            c = self._choose_color(v, partial_counts, color_sizes)
            solution[v] = c
            color_sizes[c] += 1

            # Once v gets color c, every uncolored neighbor u sees one more
            # neighbor with color c.
            for u in self.adj_list[v]:
                if solution[u] == -1:
                    partial_counts[u, c] += 1

        return solution

    # ---------------------------------------------------------------------
    # Pheromone update
    # ---------------------------------------------------------------------

    def _update_pheromones(self, solutions, conflicts):
        """
        Elite-only pheromone update.

        This is more stable than updating pheromone using every ant,
        especially on hard DIMACS instances.
        """
        self.Tau *= (1.0 - self.rho)

        order = np.argsort(conflicts)
        elite_count = max(1, min(self.elite_count, len(order)))

        for rank in range(elite_count):
            idx = int(order[rank])
            sol = solutions[idx]
            conf = int(conflicts[idx])

            # Better elites deposit more.
            rank_weight = 1.0 / (rank + 1.0)
            delta = self.Q * rank_weight / (1.0 + conf)

            # Avoid excessive explosion when conf is very small.
            delta = min(delta, 1.0)

            self.Tau[np.arange(self.n), sol] += delta

        # Extra global-best reinforcement.
        if self.best_solution is not None:
            delta_best = self.global_best_weight * self.Q / (1.0 + self.best_conflicts)
            delta_best = min(delta_best, 2.0)
            self.Tau[np.arange(self.n), self.best_solution] += delta_best

        np.clip(self.Tau, self.tau_min, self.tau_max, out=self.Tau)

    # ---------------------------------------------------------------------
    # Main run
    # ---------------------------------------------------------------------

    def run(self, m=None, max_iter=30, verbose=True, enable_dynamic_colors=False):
        """
        Run ACO.

        Args:
            m:
                Number of ants. If None, use a size-aware default.
            max_iter:
                Number of ACO iterations.
            verbose:
                Print progress.
            enable_dynamic_colors:
                Ignored. Kept only for compatibility with your previous code.

        Returns:
            np.ndarray of shape (n,), dtype=np.int32.
        """
        if m is None:
            # Stronger default than your earlier 15 ants, but still manageable.
            num_ants = 40 if self.n <= 500 else 60 
        else:
            num_ants = int(m)

        max_iter = int(max_iter)

        if verbose:
            mode = "RL-biased ACO" if self.seed_solution is not None else "pure ACO"
            print(
                f"[ACO] mode={mode}, n={self.n}, k={self.k}, "
                f"ants={num_ants}, iter={max_iter}, "
                f"alpha={self.alpha}, beta={self.beta}, rho={self.rho}"
            )
            if self.seed_solution is not None:
                print(f"[ACO] seed conflicts={self.seed_conflicts}")

        start_time = time.time()

        for iteration in range(max_iter):
            solutions = []
            conflicts = []

            for _ in range(num_ants):
                sol = self._construct_solution()

                conf = self.calculate_conflicts(sol)

                solutions.append(sol)
                conflicts.append(conf)

                if conf < self.best_conflicts:
                    self.best_conflicts = int(conf)
                    self.best_solution = sol.copy()

                if self.best_conflicts == 0:
                    break

            conflicts = np.asarray(conflicts, dtype=np.int32)

            self._update_pheromones(solutions, conflicts)

            self.iter_x.append(iteration)
            self.iter_y.append(int(self.best_conflicts))

            if verbose and (iteration % 10 == 0 or self.best_conflicts == 0):
                elapsed = time.time() - start_time
                print(
                    f"[ACO] iter={iteration:4d}, "
                    f"iter_best={int(np.min(conflicts))}, "
                    f"global_best={int(self.best_conflicts)}, "
                    f"time={elapsed:.2f}s"
                )

            if self.best_conflicts == 0:
                break

        if self.best_solution is None:
            # Fallback should rarely happen.
            self.best_solution = self.rng.integers(0, self.k, size=self.n, dtype=np.int32)
            self.best_conflicts = self.calculate_conflicts(self.best_solution)

        result = np.asarray(self.best_solution, dtype=np.int32)
        result = np.clip(result, 0, self.k - 1)

        if verbose:
            elapsed = time.time() - start_time
            print(f"[ACO] done, best_conflicts={self.best_conflicts}, time={elapsed:.2f}s")

        return result