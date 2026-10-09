import gymnasium as gym
import networkx as nx
import numpy as np
from gymnasium import spaces
import random
import matplotlib.pyplot as plt
import tabucol
from PIL import Image
from gcp_env.ACO import AntColony


class GcpEnv(gym.Env):
    metadata = {"render_modes": ["human", "file"], "render_fps": 60}

    def __init__(
        self,
        graph,
        k,
        tabucol_iters=10000,
        beta=0.2,
        render_mode=None,
        base_filename=None,
        tabucol_init=False,
        aco_num_ants=40,
        aco_max_iter=50,
        disable_tabucol=False,#used in RLACO experiment
        use_aco=False,
        inject_mode="soft",  # "hard" or "soft"
        soft_prior_fn=None,   # callable: (obs_dict) -> np.ndarray (n, k), set from runner
        keep_rl_best=False,   # if True, restore the best score frame seen during the RL phase before invoking ACO/TabuCol

        # TabuCol perturbation options
        aco_then_tabucol=False,  # if True, run TabuCol as a polish stage after ACO inside the same episode
        restart_every=0,      # if > 0, perturb the carried-over solution every K episodes
        restart_perturb=0.2,  # fraction of vertices recolored randomly when restart fires; 1.0 = full random reset
        random_seed=None,
    ):
        self._graph = graph
        self._k = k
        self.k = k
        self._adj_matrix = nx.to_numpy_array(graph, dtype=np.int32)
        self._disable_tabucol = disable_tabucol
        self._use_aco = use_aco
        self._inject_mode = inject_mode
        self._soft_prior_fn = soft_prior_fn
        self._keep_rl_best = keep_rl_best

        self._aco_then_tabucol = aco_then_tabucol
        self._restart_every = int(restart_every)
        self._restart_perturb = float(restart_perturb)
        self._random_seed = None if random_seed is None else int(random_seed)
        self._reset_count = 0
        self._completed_episodes = 0

        n = len(self._graph)

        self.observation_space = spaces.Dict(
            {
                "node_features": spaces.Box(
                    low=0, high=n, shape=(n, 3), dtype=np.float32
                ),
                "col_features": spaces.Box(
                    low=0, high=n, shape=(n, k, 3), dtype=np.float32
                ),
                "k": spaces.Discrete(k, start=1),
            }
        )

        self.solution_space = spaces.Box(low=0, high=k - 1, shape=(n,), dtype=np.int32)
        self.action_space = spaces.MultiDiscrete([n, k])

        self._n = n

        self.color_map = None
        self.layout = None

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        if render_mode == "file":
            assert base_filename is not None
            self.render_iter = 0

        self._step_counter = 0

        adj_list = [[] for i in range(0, len(graph))]
        for node in graph:
            for neighbor in graph.neighbors(node):
                adj_list[node].append(neighbor)

        # TabuCol is built when:
        #   (a) it is the main episode-end local search (RLTabucol mode), OR
        #   (b) it polishes the ACO output (RLACOT pipeline).
        need_tabucol = (
            (not self._disable_tabucol and not self._use_aco)
            or (self._use_aco and self._aco_then_tabucol)
        )
        if need_tabucol:
            self._tabucol = tabucol.TabuColSolver(
                adj_list,
                k,
                max_iterations=tabucol_iters,
                tabu_a=10,
                tabu_alpha=1.2,
                beta=beta,
            )
        else:
            self._tabucol = None

        if self._use_aco:
            self._aco_local = AntColony(graph, k, seed=self._random_seed)
        else:
            self._aco_local = None
        
        self._aco_num_ants = aco_num_ants
        self._aco_max_iter = aco_max_iter
        
        # statistics when running
        self._initial_score = 0
        self._rl_improvement = 0
        self._tabucol_improvement = 0
        self._action_stats = {
            'total_actions': 0,
            'effective_actions': 0,
            'node_selections': np.zeros(n),
            'color_selections': np.zeros(k)
        }# track agent behavior

        self.graph = graph
        self.final_score = 0
        self.final_solution = None
        self.episode_initial_score = 0
        self.episode_rl_improvement = 0
        self.episode_tabucol_improvement = 0

        # RLACO / RL TabuCol local search score tracking
        self.episode_rl_start_score = 0
        self.episode_rl_end_score = 0
        self.episode_rl_actual_end_score = 0 # RL step 300 score before ACO/TabuCol
        self.episode_rl_best_step = 0        # step number of best frame within RL phase
        self.episode_aco_seed_score = 0      # ACO seed conflicts，equal to RL end score
        self.episode_aco_end_score = 0       # score after ACO finishes

        # RLACOT pipeline: post-ACO TabuCol polish stage
        self.episode_polish_seed_score = 0   # TabuCol polish entry score (= ACO end)
        self.episode_polish_end_score = 0    # TabuCol polish final score


        # reset last_episode_*
        self.last_episode_rl_start_score = 0
        self.last_episode_rl_end_score = 0
        self.last_episode_rl_actual_end_score = 0
        self.last_episode_rl_best_step = 0
        self.last_episode_aco_seed_score = 0
        self.last_episode_aco_end_score = 0
        self.last_episode_polish_seed_score = 0
        self.last_episode_polish_end_score = 0
        
        self.last_episode_initial_score = 0
        self.last_episode_final_score = 0
        self.last_episode_rl_improvement = 0
        self.last_episode_tabucol_improvement = 0
        self.last_episode_solution = None
        self.last_episode_action_stats = None

        if tabucol_init and not self._disable_tabucol:
            tabucol_sol, _ = self._tabucol.solve()
            self._solution = tabucol_sol
            self._prev_solution = tabucol_sol
        else:
            self._prev_solution = None
            self._solution = None

        self.render_mode = render_mode
        self.base_filename = base_filename

    def set_soft_prior_fn(self, fn):
        """Inject a callable that maps the env's current obs dict to an (n, k)
        probability distribution. Used only when inject_mode='soft'."""
        self._soft_prior_fn = fn

    def apply_perturbation(self, perturb_frac):
        """Randomly recolor a fraction of vertices in _prev_solution.

        Called from runner.py's main loop at logical-episode boundaries to
        escape search traps without discarding all prior progress.
        Returns (score_before, score_after, n_perturbed).
        """
        if self._prev_solution is None:
            return None, None, 0
        prev = np.array(self._prev_solution, copy=True)
        if perturb_frac >= 1.0:
            perturbed = self.solution_space.sample()
        else:
            perturbed = np.array(prev, copy=True)
            n_perturb = max(1, int(self._n * perturb_frac))
            idx = np.random.choice(self._n, size=n_perturb, replace=False)
            new_colors = np.random.randint(0, self._k, size=n_perturb)
            perturbed[idx] = new_colors
        score_before = self._calculate_score_from(prev)
        score_after = self._calculate_score_from(perturbed)
        n_changed = int(np.sum(prev != perturbed))
        self._prev_solution = perturbed
        self._solution = perturbed
        return score_before, score_after, n_changed

    def get_last_aco_curve(self):
        """Return iter_x, iter_y of the last ACO call inside an episode end.
        Empty lists if no ACO call has happened yet."""
        if self._aco_local is None:
            return [], []
        return list(self._aco_local.iter_x), list(self._aco_local.iter_y)

    def reset(self, seed=None, options=None):
        if seed is None and self._random_seed is not None:
            seed = self._random_seed + self._reset_count
        super().reset(seed=seed)
        self._reset_count += 1

        self.solution_space = spaces.Box(
            low=0, high=self._k - 1, shape=(self._n,), dtype=np.int32, seed=seed
        )
        self.action_space = spaces.MultiDiscrete([self._n, self._k], seed=seed)

        if options is not None and "initial_solution" in options:
            self._solution = options["initial_solution"]
            self._prev_solution = self._solution

        elif options is not None and options.get("random_init", False):
            self._solution = self.solution_space.sample()
            self._prev_solution = self._solution
        # Normal perturbation (carry-over branch)
        elif self._prev_solution is not None:
            self._solution = self._prev_solution
        else:
            self._solution = self.solution_space.sample()
            self._prev_solution = self._solution

        self._step_counter = 0
        self._initialize_obs()  # initialize observation features
        self.score = self._calculate_score()
        
        #reset statistics
        self._initial_score = self.score
        self._rl_improvement = 0
        self._tabucol_improvement = 0
        # Score tracking for RLACO analysis
        self.episode_rl_start_score = self.score
        self.episode_rl_end_score = self.score
        self.episode_rl_actual_end_score = self.score
        self.episode_rl_best_step = 0
        self.episode_aco_seed_score = self.score
        self.episode_aco_end_score = self.score
        self.episode_polish_seed_score = self.score
        self.episode_polish_end_score = self.score
        self.episode_initial_score = 0
        self.episode_rl_improvement = 0
        self.episode_tabucol_improvement = 0
        self.final_score = 0
        self.final_solution = None
        if not hasattr(self, 'last_episode_initial_score'):
            self.last_episode_initial_score = 0
            self.last_episode_final_score = 0
            self.last_episode_rl_improvement = 0
            self.last_episode_tabucol_improvement = 0
            self.last_episode_solution = None
            self.last_episode_action_stats = None

        # Save action stats before reset (for runner.py to read)
        if hasattr(self, '_action_stats') and self._action_stats['total_actions'] > 0:
            self.last_episode_action_stats = {
                'total_actions': self._action_stats['total_actions'],
                'effective_actions': self._action_stats['effective_actions'],
                'node_selections': self._action_stats['node_selections'].copy(),
                'color_selections': self._action_stats['color_selections'].copy(),
            }

        self._action_stats = {
            'total_actions': 0,
            'effective_actions': 0,
            'node_selections': np.zeros(self._n),
            'color_selections': np.zeros(self._k)
        }

        # Track episode-internal best (frame with the lowest score during RL phase)
        self._rl_best_score = self.score
        self._rl_best_solution = self._solution.copy()
        self._rl_best_step = 0
        observation = self._get_obs()

        return observation, {"solution": self._solution}

    def step(self, action):
        action_node, new_group = action[0], action[1]

        prev_group = self._solution[action_node]
        reward = 0
        
        self._action_stats['total_actions'] += 1
        self._action_stats['node_selections'][action_node] += 1
        self._action_stats['color_selections'][new_group] += 1

        if prev_group != new_group:
            self._action_stats['effective_actions'] += 1
            conflicts = self._node_feats[action_node, 1]

            self._solution[action_node] = new_group  # apply the color change

            new_conflicts = 0
            neighbors_groups = set()
            for neighbor in self._graph.neighbors(action_node):
                if self._solution[action_node] == self._solution[neighbor]:
                    new_conflicts += 1

                neighbors_groups.add(self._solution[neighbor])
                self._col_feats[neighbor, prev_group, 0] -= 1
                self._col_feats[neighbor, new_group, 0] += 1

                if self._solution[neighbor] == prev_group:
                    self._node_feats[neighbor, 1] -= 1
                if self._solution[neighbor] == new_group:
                    self._node_feats[neighbor, 1] += 1
                if self._col_feats[neighbor, prev_group, 0] == 0:
                    self._node_feats[neighbor, 2] -= 1  # if neighbor's count for old color becomes 0, decrease color group count
                if self._col_feats[neighbor, new_group, 0] == 1:
                    self._node_feats[neighbor, 2] += 1  # if neighbor's count for new color becomes 1, increase color group count
            
            # Update action_node's own features
            self._node_feats[action_node, 1] = new_conflicts  # update action_node's new conflict count
            self._node_feats[action_node, 2] = len(neighbors_groups)

            self._col_feats[:, prev_group, 1] -= 1  # update global color group size: old color group size -1
            self._col_feats[:, new_group, 1] += 1   # update global color group size: new color group size +1

            reward = conflicts - new_conflicts  # immediate reward
            self._rl_improvement += reward

        self.score -= reward  # update global score

        # Track episode-internal best frame
        if self.score < self._rl_best_score:
            self._rl_best_score = self.score
            self._rl_best_solution = self._solution.copy()
            self._rl_best_step = self._step_counter

        terminated = False
        if self.score == 0:
            terminated = True

        self._step_counter += 1
        observation = self._get_obs()

        tabucol_rew = 0
        if self._step_counter == self.spec.max_episode_steps:
            # Always record what RL actually ended at (before any restoration).
            # This is the "policy view" — useful for analysing degradation.
            # NOTE: self.score is maintained incrementally (self.score -= reward)
            # and can drift from the true conflict count over 300 RL steps.
            # Recompute it exactly here so the recorded actual-end score, the
            # keep_rl_best comparison, and the value handed to Tabucol/ACO all
            # reflect the real conflicts of the current solution.
            self.score = self._calculate_score()
            self.episode_rl_actual_end_score = self.score
            self.episode_rl_best_step = self._rl_best_step

            # Optionally restore the best frame seen during the RL phase.
            # Without this, RL may have degraded a good intermediate solution
            # by the time it reaches max_steps (especially when the policy is
            # operating outside its training distribution).
            if self._keep_rl_best and self._rl_best_score < self.score:
                degraded_score = self.score
                self._solution = self._rl_best_solution.copy()
                self.score = self._rl_best_score
                self._initialize_obs()
                print(
                f"[keep_rl_best] restored frame at step {self._rl_best_step}: "
                f"score {degraded_score} -> {self._rl_best_score}",
                flush=True,
                )

            if self._use_aco:
                score_before_local = self.score
                print(
                f"[ACO local] start: seed_score={score_before_local}, "
                f"ants={self._aco_num_ants}, iter={self._aco_max_iter}, "
                f"inject_mode={self._inject_mode}",
                flush=True,
                )
                self.episode_rl_end_score = score_before_local

                if self._inject_mode == "soft" and self._soft_prior_fn is not None:
                    # Soft prior: pheromone shaped by the policy's per-vertex
                    # color distribution, but best/seed baseline is the RL solution.
                    dist = self._soft_prior_fn(self._get_obs())
                    self._aco_local.bias_from_distribution(dist, solution=self._solution)
                else:
                    # Hard prior (default): use the RL solution as argmax.
                    self._aco_local.bias_from_solution(self._solution)

                self.episode_aco_seed_score = int(self._aco_local.seed_conflicts)
                aco_solution = self._aco_local.run(
                    m=self._aco_num_ants,
                    max_iter=self._aco_max_iter,
                    verbose=False
                )
                self._solution = aco_solution
                self._prev_solution = self._solution

                score_after_local = self._calculate_score()
                self.score = score_after_local

                self.episode_aco_end_score = score_after_local
                self._tabucol_improvement = score_before_local - score_after_local
                print(
                f"[ACO local] done: before={score_before_local}, "
                f"seed={self.episode_aco_seed_score}, "
                f"after={score_after_local}, "
                f"improvement={self._tabucol_improvement}",
                flush=True,
                )

                # Optional TabuCol polish stage after ACO (RLACOT pipeline)
                if self._aco_then_tabucol and self._tabucol is not None and self.score > 0:
                    polish_seed = self.score
                    self.episode_polish_seed_score = polish_seed
                    print(
                        f"[TabuCol polish] start: seed_score={polish_seed}",
                        flush=True,
                    )
                    self._tabucol.set_solution(self._solution)
                    polished_solution, _ = self._tabucol.solve()
                    self._solution = polished_solution
                    self._prev_solution = self._solution
                    polish_end = self._calculate_score()
                    self.score = polish_end
                    self.episode_polish_end_score = polish_end
                    print(
                        f"[TabuCol polish] done: before={polish_seed}, "
                        f"after={polish_end}, "
                        f"improvement={polish_seed - polish_end}",
                        flush=True,
                    )
                else:
                    # No polish (or polish skipped because already at score=0)
                    self.episode_polish_seed_score = self.score
                    self.episode_polish_end_score = self.score
            elif not self._disable_tabucol:
                score_before_tabucol = self.score
                self._tabucol.set_solution(self._solution)
                self._prev_solution, tabucol_rew = self._tabucol.solve()
                self._solution = self._prev_solution
                self._prev_solution = self._solution
                self._tabucol_improvement = score_before_tabucol - self._calculate_score()
                self.score = self._calculate_score()

        if terminated or self._step_counter == self.spec.max_episode_steps:
        # If early termination at score==0 before reaching max_steps,
        # RL end score equals current score.
            if terminated and self._step_counter < self.spec.max_episode_steps:
                self.episode_rl_end_score = self.score
                if self._use_aco:
                    self.episode_aco_seed_score = self.score
                    self.episode_aco_end_score = self.score

            self.final_score = self.score
            self.final_solution = self._solution.copy()

            self.episode_initial_score = self._initial_score
            self.episode_rl_improvement = self._rl_improvement
            self.episode_tabucol_improvement = self._tabucol_improvement

    # Existing last-episode fields
            self.last_episode_initial_score = self._initial_score
            self.last_episode_final_score = self.score
            self.last_episode_rl_improvement = self._rl_improvement
            self.last_episode_tabucol_improvement = self._tabucol_improvement
            self.last_episode_solution = self._solution.copy()

    # New last-episode fields
            self.last_episode_rl_start_score = self.episode_rl_start_score
            self.last_episode_rl_end_score = self.episode_rl_end_score
            self.last_episode_rl_actual_end_score = self.episode_rl_actual_end_score
            self.last_episode_rl_best_step = self.episode_rl_best_step
            self.last_episode_aco_seed_score = self.episode_aco_seed_score
            self.last_episode_aco_end_score = self.episode_aco_end_score
            self.last_episode_polish_seed_score = self.episode_polish_seed_score
            self.last_episode_polish_end_score = self.episode_polish_end_score

            # Perturbation restart: every K logical episodes (defined as
            # completed RL+localsearch episodes inside this env), recolor a
            # fraction of vertices in _prev_solution so the NEXT env.reset()
            # will pick up the perturbed state via the carry-over branch.
            # Score>0 only — if we already solved, no need to perturb.
            self._completed_episodes += 1
            if (self._restart_every > 0
                    and self._completed_episodes % self._restart_every == 0
                    and self.score > 0
                    and self._prev_solution is not None):
                prev = np.array(self._prev_solution, copy=True)
                if self._restart_perturb >= 1.0:
                    perturbed = self.solution_space.sample()
                else:
                    perturbed = np.array(prev, copy=True)
                    n_perturb = max(1, int(self._n * self._restart_perturb))
                    idx = np.random.choice(self._n, size=n_perturb, replace=False)
                    new_colors = np.random.randint(0, self._k, size=n_perturb)
                    perturbed[idx] = new_colors
                score_before = self._calculate_score_from(prev)
                score_after = self._calculate_score_from(perturbed)
                n_changed = int(np.sum(prev != perturbed))
                self._prev_solution = perturbed
                print(
                    f"[RESTART] after episode {self._completed_episodes}: "
                    f"score {score_before} -> {score_after} "
                    f"(perturbed {n_changed} vertices, frac={self._restart_perturb})",
                    flush=True,
                )

        info = {
            "solution": self._solution,
            "graph": self._graph,
            "initial_score": self._initial_score,
            "final_score": self.score,
            "rl_improvement": self._rl_improvement,
            "tabucol_improvement": self._tabucol_improvement,
            # new fields for analysis
            "rl_start_score": self.episode_rl_start_score,
            "rl_end_score": self.episode_rl_end_score,
            "rl_actual_end_score": self.episode_rl_actual_end_score,
            "rl_best_step": self.episode_rl_best_step,
            "aco_seed_score": self.episode_aco_seed_score,
            "aco_end_score": self.episode_aco_end_score,
            "local_improvement": self._tabucol_improvement,
            "local_search": "aco" if self._use_aco else (
        "tabucol" if not self._disable_tabucol else "none"
        ),
        }

        return (
            observation,
            reward + tabucol_rew,
            terminated,
            False,
            info,
        )

    def render(self):
            # clear_output(wait=True)

        import matplotlib.pyplot as plt

        if self.color_map is None:
            self.color_map = dict(
                [
                    (
                        j,
                        f"#{''.join([random.choice('0123456789ABCDEF') for i in range(6)])}",
                    )
                    for j in range(self._k)
                ]
            )

        if self.layout is None:
            self.layout = nx.spring_layout(self._graph)

        fig, ax = plt.subplots(figsize=(10, 10))
        fig.tight_layout()
        ax.axis("off")
        node_colors = [self.color_map[node] for node in self._solution]

        edge_colors = [
            "#ff0000" if self._solution[x] == self._solution[y] else "#000000"
            for x, y in nx.edges(self._graph)
        ]
        alphas = [
            1.0 if self._solution[x] == self._solution[y] else 0.1
            for x, y in nx.edges(self._graph)
        ]

        plt.cla()
        nx.draw_networkx_nodes(self._graph, self.layout, node_color=node_colors)
        nx.draw_networkx_labels(self._graph, self.layout)
        nx.draw_networkx_edges(
            self._graph, self.layout, edge_color=edge_colors, alpha=alphas
        )
        fig.canvas.draw()

        image_from_plot = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        image_from_plot = image_from_plot.reshape(
            fig.canvas.get_width_height()[::-1] + (3,)
        )

        if self.render_mode == "file":
            img = Image.fromarray(image_from_plot)
            print("saving", f"{self.base_filename}{self.render_iter}.png")
            img.save(f"{self.base_filename}{self.render_iter}.png")
            self.render_iter += 1
            plt.close()

        elif self.render_mode == "human":
            plt.show()

    def _get_obs(self):
        return {
            "node_features": self._node_feats,
            "col_features": self._col_feats,
            "k": self._k,
        }

    def _initialize_obs(self):
        # Features of node: (degree, conflicting edges, neighboring groups)
        node_feats = np.zeros((self._n, 3), dtype=np.float32)
        # Features of node, group pairs (i, j): (number of nodes in group j adjacent to i, total number of nodes in group j, |V|)
        col_feats = np.zeros((self._n, self._k, 3), dtype=np.float32)

        for node in self._graph.nodes():
            conflicts = 0
            groups = set()
            for adj in self._graph.neighbors(node):
                if self._solution[node] == self._solution[adj]:
                    conflicts += 1
                groups.add(self._solution[adj])
            node_feats[node, :] = [self._graph.degree(node), conflicts, len(groups)]

        nodes_with_color = np.zeros((self._k), dtype=np.int32)
        for col in self._solution:
            nodes_with_color[col] += 1

        for node in self._graph.nodes():
            for col in range(0, self._k):
                n_adjacent = 0
                for adj in self._graph.neighbors(node):
                    if self._solution[adj] == col:
                        n_adjacent += 1
                col_feats[node, col, :] = [n_adjacent, nodes_with_color[col], self._n]

        self._node_feats = node_feats
        self._col_feats = col_feats

    def _calculate_score(self):
        score = 0
        for node in self._graph.nodes():
            for neighbor in self._graph.neighbors(node):
                if self._solution[node] == self._solution[neighbor]:
                    score += 1

        score = score // 2
        return score

    def _calculate_score_from(self, sol):
        """Calculate score for any solution (used for debug logging in restart)."""
        score = 0
        for node in self._graph.nodes():
            for neighbor in self._graph.neighbors(node):
                if sol[node] == sol[neighbor]:
                    score += 1
        return score // 2

    def get_graph(self): 
        return self._graph

    def get_solution(self):
        # Return the most recently completed episode's solution.
        # Falls back to _prev_solution if no episode has completed yet.
        if getattr(self, "last_episode_solution", None) is not None:
            return self.last_episode_solution
        return self._prev_solution

    def get_action_stats(self):
        """Return action statistics for analyzing agent behavior.

        Returns dict with:
            - total_actions: int, total steps taken
            - effective_actions: int, steps that changed color
            - node_selections: list (n,), frequency per node
            - color_selections: list (k,), frequency per color
            - effective_ratio: float, effective_actions / total_actions
            - top_5_nodes: list of (node_id, count) sorted by frequency
            - top_5_colors: list of (color_id, count) sorted by frequency

        Note: Returns last_episode's stats (saved before reset), not current _action_stats.
        """
        # Use saved stats from last episode, not the current empty one
        if self.last_episode_action_stats is None:
            # First episode hasn't completed yet, return empty stats
            return {
                'total_actions': 0,
                'effective_actions': 0,
                'node_selections': [0.0] * self._n,
                'color_selections': [0.0] * self._k,
                'effective_ratio': 0.0,
                'top_5_nodes': [(i, 0.0) for i in range(min(5, self._n))],
                'top_5_colors': [(i, 0.0) for i in range(min(5, self._k))],
            }

        stats = self.last_episode_action_stats.copy()
        total = stats['total_actions']
        effective = stats['effective_actions']

        # Convert numpy arrays to lists for JSON serialization
        if isinstance(stats['node_selections'], np.ndarray):
            stats['node_selections'] = stats['node_selections'].tolist()
        if isinstance(stats['color_selections'], np.ndarray):
            stats['color_selections'] = stats['color_selections'].tolist()

        # Add derived metrics
        if total > 0:
            stats['effective_ratio'] = effective / total
        else:
            stats['effective_ratio'] = 0.0

        # Top 5 most frequently selected nodes
        node_counts = stats['node_selections']
        top_5_nodes = sorted(
            [(i, node_counts[i]) for i in range(len(node_counts))],
            key=lambda x: x[1],
            reverse=True
        )[:5]
        stats['top_5_nodes'] = top_5_nodes

        # Top 5 most frequently selected colors
        color_counts = stats['color_selections']
        top_5_colors = sorted(
            [(i, color_counts[i]) for i in range(len(color_counts))],
            key=lambda x: x[1],
            reverse=True
        )[:5]
        stats['top_5_colors'] = top_5_colors

        return stats
