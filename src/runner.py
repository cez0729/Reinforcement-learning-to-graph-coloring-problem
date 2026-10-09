import numpy as np
from gcp_env import GcpEnv
from gymnasium.envs.registration import register
import argparse
import random
import networkx as nx
import time
import json
import os
import sys
from pathlib import Path

import torch
import gymnasium as gym
from tianshou.data import Collector
from tianshou.utils.net.common import ActorCritic
from tianshou.env import DummyVectorEnv
from network import ActorNetwork, CriticNetwork, GCPPPOPolicy
from tianshou.policy import BasePolicy
from tianshou.data import Batch


class ThinGraph(nx.Graph):
    all_edge_dict = {"weight": 1}

    def single_dict(self):
        return self.all_edge_dict

    edge_attr_dict_factory = single_dict
    node_attr_dict_factory = single_dict


def read_graph_from_file(path):
    graph = nx.Graph()
    with open(path, "r") as f:
        for line in f.readlines():
            if line.startswith("c"):
                continue
            if line.startswith("p"):
                 n, m = line.split()[2:4]
                 n, m = int(n), int(m)

                 graph.add_nodes_from(range(n))
                 continue
            if line.startswith("e"):
                u, v = line.split()[1:3]
                u, v = int(u), int(v)
                graph.add_edge(u - 1, v - 1)
                continue
    return graph


def initial_solution(graph, k):
    solution = {}
    for node in graph.nodes():
        solution[node] = random.randint(0, k - 1)
    return solution


def calculate_score(graph, solution):
    score = 0
    for node in graph.nodes():
        for neighbor in graph.neighbors(node):
            if solution[node] == solution[neighbor]:
                score += 1
    score = score // 2
    return score


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


class RandomGCPPolicy(BasePolicy):
    def __init__(self, action_space):
        super().__init__()
        self.action_space = action_space

    def forward(self, batch, state=None, **kwargs):
        action = np.array([self.action_space.sample()])
        return Batch(act=action)

    def learn(self, batch: Batch, **kwargs):
        return {}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Reinforcement learning based tabu search for graph coloring"
    )
    parser.add_argument("policy", type=str, help="Path to policy to use")
    parser.add_argument("graph", type=str, help="Path to graph file")
    parser.add_argument("k", type=int, help="Number of colors")
    parser.add_argument(
        "-I",
        "--max-steps",
        type=int,
        dest="max_steps",
        default=300,
        help="Max RL steps per episode (default 300, matches training distribution in trainer.py)",
    )
    parser.add_argument(
        "-T",
        "--max-tabucol-iters",
        type=int,
        dest="max_tabucol_iters",
        default=100000,
        help="Max tabucol iterations in each episode",
    )
    parser.add_argument(
        "-E",
        "--max-episodes",
        type=int,
        dest="episodes",
        default=None,
        help="Max episodes to run",
    )
    parser.add_argument(
        "-B",
        "--beta",
        type=float,
        dest="beta",
        default=0.2,
        help="Beta parameter in RLTCol",
    )
    parser.add_argument(
        "--time-limit",
        type=int,
        dest="time_limit",
        default=18000,
        help="Time limit per run in seconds (default: 18000 = 5 hours, matching the RLTabucol testing budget)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Recorded independent-run seed (default: 0).",
    )
    parser.add_argument(
        "--result-json",
        type=Path,
        default=None,
        help="Write the final structured result to this JSON file.",
    )
    parser.add_argument(
        "--RL", type=str2bool, nargs="?", const=True, default=True, help="Use RL or not"
    )
    # ===== ACO参数 =====
    parser.add_argument(
        "--aco-ants",
        type=int,
        dest="aco_ants",
        default=40,
        help="Number of ants in ACO (default: 40)"
    )
    parser.add_argument(
        "--aco-iter",
        type=int,
        dest="aco_iter",
        default=50,
        help="Number of ACO iterations per episode (default: 50)"
    )
    parser.add_argument(
        "--use-aco",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        dest="use_aco",
        help=(
            "Use ACO at end of each episode instead of TabuCol. "
            "Combine with --RL True for RL+ACO, --RL False for pure ACO baseline."
        ),
    )
    parser.add_argument(
        "--disable-tabucol",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        dest="disable_tabucol",
        help=(
            "Disable Tabucol handoff entirely (actor-only mode). "
            "Each episode contains only RL recoloring steps."
        ),
    )
    parser.add_argument(
        "--inject-mode",
        type=str,
        choices=["hard", "soft"],
        default="hard",
        dest="inject_mode",
        help=(
            "How the RL policy's signal is injected into ACO pheromone. "
            "'hard' = boost tau at solution[v] only (argmax). "
            "'soft' = interpolate tau by the policy's per-vertex color distribution. "
            "Only meaningful when --RL True --use-aco True."
        ),
    )
    parser.add_argument(
        "--pure-aco",
        action="store_true",
        default=False,
        dest="pure_aco",
        help=(
            "Run Pure ACO baseline (no RL, no env, direct AntColony.run). "
            "Supplementary baseline with independent restarts. "
            "Overrides --RL and --use-aco."
        ),
    )
    parser.add_argument(
        "--use-conflict-prior",
        action="store_true",
        default=False,
        dest="use_conflict_prior",
        help=(
            "Run the RLACO_conf baseline: SAME trained policy and 300-step "
            "RL phase as RLACO (requires --RL True), but the ACO soft "
            "prior is replaced with an explicit per-episode conflict-count "
            "heuristic derived from the current post-RL coloring (read "
            "from col_features[:,:,0], no extra greedy solve). Isolates "
            "'learned vs explicit prior' from 'RL phase used', answering "
            "whether the actor's learned color preference beats a simple "
            "conflict-count heuristic under an identical RL phase and "
            "identical soft-injection pipeline. Forces --use-aco True "
            "--inject-mode soft; requires --RL True (use RLACO's policy "
            "checkpoint)."
        ),
    )
    parser.add_argument(
        "--keep-rl-best",
        action="store_true",
        default=False,
        dest="keep_rl_best",
        help=(
            "Before invoking ACO/TabuCol at episode end, restore the best "
            "score frame seen during the RL phase. Mitigates RL degradation "
            "in the policy's out-of-distribution regime (low-conflict states)."
        ),
    )
    parser.add_argument(
        "--polish-tabucol",
        action="store_true",
        default=False,
        dest="polish_tabucol",
        help=(
            "After ACO finishes inside each episode, run TabuCol as a polish "
            "stage seeded with the ACO best. Forms the RLACOT pipeline "
            "(RL -> ACO -> TabuCol). Only meaningful with --use-aco."
        ),
    )
    parser.add_argument(
        "--terminal-tabucol",
        action="store_true",
        default=False,
        dest="terminal_tabucol",
        help=(
            "After the RL/ACO stage, run an independent terminal TabuCol "
            "stage from the saved global-best coloring. Each block resets "
            "the tabu table and passes its returned coloring forward."
        ),
    )
    parser.add_argument(
        "--terminal-time-limit",
        type=int,
        default=None,
        dest="terminal_time_limit",
        help="Terminal TabuCol budget in seconds (default: same as --time-limit).",
    )
    parser.add_argument(
        "--terminal-tabu-iters",
        type=int,
        default=10000,
        dest="terminal_tabu_iters",
        help="Iterations in each terminal TabuCol block (default: 10000).",
    )
    parser.add_argument(
        "--allow-python-tabucol",
        action="store_true",
        default=False,
        help="Allow the Python Tabucol fallback for smoke tests only.",
    )
    parser.add_argument(
        "--restart-every",
        type=int,
        dest="restart_every",
        default=0,
        help=(
            "If > 0, perturb the carried-over solution every K env resets "
            "(K = episodes for RL+local-search pipeline). Used to escape "
            "search traps on hard cases like le450_15d. Default 0 (disabled)."
        ),
    )
    parser.add_argument(
        "--restart-perturb",
        type=float,
        dest="restart_perturb",
        default=0.2,
        help=(
            "Fraction of vertices to recolor randomly when the restart fires. "
            "1.0 = full random reset; smaller values preserve more of the "
            "prior solution. Default 0.2."
        ),
    )
    # ===== Render parameters =====
    parser.add_argument(
        "--render-mode",
        type=str,
        choices=["human", "file"],
        default=None,
        dest="render_mode",
        help="Visualization mode: 'human' for display, 'file' for saving PNG images"
    )
    parser.add_argument(
        "--render-dir",
        type=str,
        default="render_output",
        dest="render_dir",
        help="Directory to save render images (used with --render-mode file)"
    )
    # ===================

    device = "cuda" if torch.cuda.is_available() else "cpu"

    node_features = 3
    col_features = 3

    args = parser.parse_args()
    if args.polish_tabucol and args.terminal_tabucol:
        parser.error(
            "--polish-tabucol and --terminal-tabucol are different protocols "
            "and cannot be enabled together."
        )
    if args.terminal_tabucol and not args.use_aco:
        parser.error("--terminal-tabucol requires --use-aco.")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    if args.use_conflict_prior:
        # RLACO_conf must share RLACO's exact RL phase (same trained policy),
        # so --RL is NOT auto-forced here — the caller must explicitly pass
        # --RL True (with the same --policy checkpoint used by RLACO) so a
        # misconfiguration (forgetting the policy path) fails loudly instead
        # of silently reusing whatever --RL happened to default to.
        if not args.RL:
            parser.error(
                "--use-conflict-prior requires --RL True together with "
                "--policy pointing at the same checkpoint used by RLACO "
                "(RLACO_conf keeps RLACO's RL phase identical; only the "
                "ACO soft prior source differs)."
            )
        args.use_aco = True
        args.inject_mode = "soft"
    graph = read_graph_from_file(args.graph)

    # 五种实验模式：
    #   RLACO      : --RL True  --use-aco True   走 collector + GcpEnv (ACO 在 episode 末)
    #   RLACO0     : --RL False --use-aco True   走 collector + RandomGCPPolicy + GcpEnv (ACO 在 episode 末)
    #   RLACO_conf : --use-conflict-prior          走 collector + 训练好的 policy + GcpEnv，RL 阶段与 RLACO
    #                                              完全一致，ACO 注入源换成当前 episode 冲突计数启发式
    #                                              （单变量对照 RLACO：learned prior vs explicit prior）
    #   Pure ACO   : --pure-aco                  绕过 RL/env，直接 AntColony.run()（supplementary baseline）
    #   RLTabucol  : --RL True  --use-aco False  走 collector + GcpEnv (TabuCol 在 episode 末)
    #   注: --RL False --use-aco False 退化为 RandomTabucol
    pure_aco_mode = getattr(args, 'pure_aco', False)

    if pure_aco_mode:
        # ----- Pure ACO baseline：不创建 env、不加载 policy -----
        from gcp_env.ACO import AntColony

        mode_label = "Pure ACO (no RL)"
        print("-------------------------")
        print("Starting evaluation, info:")
        print(f"Mode: {mode_label}")
        print(f"Graph: {args.graph}")
        print(f"k: {args.k}")
        print(f"Time limit: {args.time_limit}")
        print(f"ACO ants: {args.aco_ants}")
        print(f"ACO iterations per episode: {args.aco_iter}")
        print(f"Max episodes: {args.episodes}")
        print("-------------------------")
        sys.stdout.flush()

        episodes = 0
        tot_start = time.time()
        best_score = float('inf')
        best_solution = None
        episode_traces = []

        # 单个连续的 ACO 长跑：pheromone (Tau) 和 best_solution 在 episode
        # 之间不重置，每个 episode 只是同一次 run() 的又一段 max_iter 迭代，
        # 这样才是真正意义上的纯 ACO（不是多次独立重启）。
        aco = AntColony(graph, args.k)
        while True:
            if args.episodes is not None and episodes >= args.episodes:
                break
            if time.time() - tot_start > args.time_limit:
                break

            episodes += 1
            start_time = time.time()

            sol = aco.run(
                m=args.aco_ants,
                max_iter=args.aco_iter,
                verbose=False,
            )
            end_time = time.time()
            current_score = calculate_score(graph, sol)

            if current_score < best_score:
                best_score = current_score
                best_solution = sol.copy()

            episode_traces.append({
                "episode": episodes,
                "t_start": start_time - tot_start,
                "t_end": end_time - tot_start,
                "rl_start_score": None,
                "rl_end_score": None,
                "aco_seed_score": None,
                "aco_end_score": current_score,
                "final_score": current_score,
                "running_best": best_score,
                "aco_iter_x": list(aco.iter_x),
                "aco_iter_y": list(aco.iter_y),
            })
            # run() 内部不会自动清空 iter_x/iter_y（只有 reset_pheromones()/
            # bias_from_solution() 才清空），必须手动清空，否则会跨 episode
            # 无限累积，导致内存和输出文件 O(episode^2) 增长。不影响 Tau/
            # best_solution 的连续性。
            aco.iter_x = []
            aco.iter_y = []

            print(
                f"Episode {episodes} done, "
                f"score: {current_score}, "
                f"best: {best_score}, "
                f"time: {end_time - start_time:.2f}s",
                flush=True,
            )

            if current_score == 0:
                print(f"[Perfect solution found in episode {episodes}!]", flush=True)
                break

        tot_end = time.time()
        solution = best_solution
        iterations = max(episodes * args.aco_ants * args.aco_iter, 1)

        print("[DONE]")
        json_result = {
            "mode": mode_label,
            "score": best_score,
            "episodes": episodes,
            "iterations": iterations,
            "time": tot_end - tot_start,
            "time_per_iteration": (tot_end - tot_start) / iterations,
            "solution": solution,
            "graph": args.graph,
            "k": args.k,
            "max_steps": args.max_steps,
            "policy": None,
            "RL": False,
            "use_aco": True,
            "inject_mode": None,
            "aco_ants": args.aco_ants,
            "aco_iter": args.aco_iter,
            "last_rl_start_score": None,
            "last_rl_end_score": None,
            "last_aco_seed_score": None,
            "last_aco_end_score": None,
            "episode_traces": episode_traces,
        }
        print(json.dumps(json_result, default=str), flush=True)
        os._exit(0)

    # ----- RL 路径 (RL+ACO 或 RLTabucol 或 RandomTabucol) -----
    register(
        id="GcpEnvMaxIters-v0",
        entry_point="gcp_env:GcpEnv",
        max_episode_steps=args.max_steps,
    )

    spec = gym.spec("GcpEnvMaxIters-v0")

    # Setup render directory if needed
    render_base_filename = None
    if args.render_mode == "file":
        import os
        os.makedirs(args.render_dir, exist_ok=True)
        render_base_filename = os.path.join(args.render_dir, f"{args.graph.split('/')[-1].replace('.col', '')}_")

    env = gym.make(
        spec,
        graph=graph,
        k=args.k,
        tabucol_iters=args.max_tabucol_iters,
        beta=args.beta,
        aco_num_ants=args.aco_ants,
        aco_max_iter=args.aco_iter,
        disable_tabucol=args.disable_tabucol or args.use_aco,
        use_aco=args.use_aco,
        inject_mode=args.inject_mode,
        keep_rl_best=args.keep_rl_best,
        aco_then_tabucol=args.polish_tabucol,
        restart_every=args.restart_every,
        restart_perturb=args.restart_perturb,
        random_seed=args.seed,
        render_mode=args.render_mode,
        base_filename=render_base_filename,
    )

    vector_env = DummyVectorEnv([lambda: env])
    policy = None

    if args.RL:
        actor = ActorNetwork(node_features, col_features, device=device).to(device)
        critic = CriticNetwork(node_features, col_features, device=device).to(device)
        actor_critic = ActorCritic(actor, critic).to(device)
        optim = torch.optim.Adam(actor_critic.parameters(), lr=0.0003)

        dist = torch.distributions.Categorical
        policy = GCPPPOPolicy(
            actor,
            critic,
            optim,
            dist,
            k=args.k,
            nodes=len(graph),
            action_space=env.action_space,
        ).to(device)

        policy.load_state_dict(torch.load(args.policy, map_location=device))
    else:
        # 仅用于 RandomTabucol 基线（--RL False --use-aco False）
        policy = RandomGCPPolicy(action_space=env.action_space)

    policy.eval()
    eval_collector = Collector(policy, vector_env)
    eval_collector.reset()

    # If soft injection is requested, install a closure on the env that
    # turns the current obs into a per-vertex color distribution from the
    # actor network. This runs once per episode end (when ACO is invoked),
    # so it doesn't affect RL action sampling at all.
    # Mutually exclusive with --use-conflict-prior: RLACO_conf shares this
    # same --RL True branch (so the trained actor gets loaded and runs the
    # RL phase identically to RLACO), but its ACO prior must come from the
    # conflict-count closure below, not from the actor.
    if args.RL and args.use_aco and args.inject_mode == "soft" and not args.use_conflict_prior:
        actor_for_prior = policy.actor

        def _soft_prior_fn(obs):
            with torch.no_grad():
                # actor expects a batched obs dict; wrap and unwrap a singleton.
                batched = {
                    "node_features": np.expand_dims(obs["node_features"], 0),
                    "col_features": np.expand_dims(obs["col_features"], 0),
                    "k": np.array([obs["k"]]),
                }
                # The actor does its own to-tensor + device handling.
                # We bypass node_probs and only need col_probs (per-vertex distribution).
                node_feats = torch.as_tensor(
                    batched["node_features"], device=device, dtype=torch.float32
                )
                col_feats = torch.as_tensor(
                    batched["col_features"], device=device, dtype=torch.float32
                )
                # col_probs: (1, n, k), softmax over colors per vertex.
                col_probs = torch.squeeze(actor_for_prior.col_model(col_feats), -1)
                # Combine with node selection probability so confident vertices
                # weigh more — but for ACO injection, we want the per-vertex
                # color choice distribution, NOT the joint (node, color) distribution.
                # So return col_probs directly.
                dist = col_probs[0].cpu().numpy()  # (n, k)
            return dist

        env.env.set_soft_prior_fn(_soft_prior_fn)

    if args.use_conflict_prior:
        # RLACO_conf: per-episode conflict-count prior derived from the
        # CURRENT post-RL coloring (obs["col_features"][:, :, 0] already
        # holds H_e(v,c) = neighbor-color counts under the live solution,
        # incrementally maintained by GcpEnv.step() — no extra greedy
        # solve, no touching self._solution directly). Recomputed fresh
        # every episode from whatever obs the RL phase just produced, so
        # it is state-adaptive like RLACO's actor prior, differing from it
        # ONLY in prior source (explicit heuristic vs learned network).
        def _conflict_prior_fn(obs):
            H = np.asarray(obs["col_features"][:, :, 0], dtype=np.float64)  # (n, k)
            weights = 1.0 / (1.0 + H)
            dist = weights / weights.sum(axis=1, keepdims=True)
            return dist

        env.env.set_soft_prior_fn(_conflict_prior_fn)

    if args.use_conflict_prior:
        mode_label = "RLACO_conf (episode-wise conflict prior, soft)"
    elif args.use_aco and args.RL:
        mode_label = f"RLACO ({args.inject_mode})"
        if args.polish_tabucol:
            mode_label = f"RLACOT ({args.inject_mode} + tabucol polish)"
    elif args.use_aco and not args.RL:
        mode_label = f"RLACO0 (random + ACO {args.inject_mode})"
        if args.polish_tabucol:
            mode_label = f"RLACOT0 (random + ACO {args.inject_mode} + tabucol polish)"
    elif not args.use_aco and args.RL:
        mode_label = "RLTabucol"
    else:
        mode_label = "RLTabucol0 (random + TabuCol)"
    if args.terminal_tabucol:
        mode_label += " + terminal Tabucol"

    print("-------------------------")
    print("Starting evaluation, info:")
    print(f"Mode: {mode_label}")
    print(f"Device: {device}")
    print(f"Policy: {args.policy if args.RL else '(random policy)'}")
    print(f"Graph: {args.graph}")
    print(f"k: {args.k}")
    print(f"Beta: {args.beta}")
    print(f"Max RL steps: {args.max_steps}")
    print(f"Max tabucol iters: {args.max_tabucol_iters}")
    print(f"Max episodes: {args.episodes}")
    print(f"Time limit: {args.time_limit}")
    if args.use_aco:
        print(f"ACO ants: {args.aco_ants}")
        print(f"ACO iterations: {args.aco_iter}")
        print(f"(TabuCol is disabled when --use-aco is on)")
    print("-------------------------")
    sys.stdout.flush()

    episodes = 0
    tot_start = time.time()

    best_score = float('inf')
    best_solution = None
    best_episode = None
    no_improvement_count = 0

    # Per-episode convergence trace.
    # For RL+ACO: we record (episode_end_time_relative, aco_iter_curve, episode_best_score).
    # The aco_iter_curve is the inner iter_y from the ACO run inside that episode.
    episode_traces = []

    while True:
        if args.episodes is not None and episodes >= args.episodes:
            break

        run_time = time.time() - tot_start
        if run_time > args.time_limit:
            break

        episodes += 1
        start_time = time.time()

        result = eval_collector.collect(n_episode=1)

        # Render if enabled
        if args.render_mode:
            env.env.render()

        end_time = time.time()
        graph = env.env.get_graph()
        solution = env.env.get_solution()
        current_score = calculate_score(graph, solution)

        # New score tracking
        rl_start_score = getattr(env.env, "last_episode_rl_start_score", None)
        rl_end_score = getattr(env.env, "last_episode_rl_end_score", None)
        rl_actual_end_score = getattr(env.env, "last_episode_rl_actual_end_score", None)
        rl_best_step = getattr(env.env, "last_episode_rl_best_step", None)
        aco_seed_score = getattr(env.env, "last_episode_aco_seed_score", None)
        aco_end_score = getattr(env.env, "last_episode_aco_end_score", None)
        polish_seed_score = getattr(env.env, "last_episode_polish_seed_score", None)
        polish_end_score = getattr(env.env, "last_episode_polish_end_score", None)

        # Action statistics for analyzing agent behavior
        action_stats = env.env.get_action_stats()

        # ACO inner convergence curve (only meaningful when use_aco=True)
        if args.use_aco:
            aco_iter_x, aco_iter_y = env.env.get_last_aco_curve()
        else:
            aco_iter_x, aco_iter_y = [], []

        if current_score < best_score:
            best_score = current_score
            best_solution = np.asarray(solution, dtype=np.int32).copy()
            best_episode = episodes
            no_improvement_count = 0
        else:
            no_improvement_count += 1

        episode_traces.append({
            "episode": episodes,
            "t_start": start_time - tot_start,
            "t_end": end_time - tot_start,
            "rl_start_score": rl_start_score,
            "rl_end_score": rl_end_score,
            "rl_actual_end_score": rl_actual_end_score,
            "rl_best_step": rl_best_step,
            "aco_seed_score": aco_seed_score,
            "aco_end_score": aco_end_score,
            "polish_seed_score": polish_seed_score,
            "polish_end_score": polish_end_score,
            "final_score": current_score,
            "running_best": best_score,
            "aco_iter_x": aco_iter_x,
            "aco_iter_y": aco_iter_y,
            "action_stats": action_stats,
        })

        print(
            f"Episode {episodes} done, "
            f"score: {current_score}, "
            f"RL_start: {rl_start_score}, "
            f"RL_end: {rl_end_score}, "
            f"RL_actual_end: {rl_actual_end_score}, "
            f"RL_best_step: {rl_best_step}, "
            f"ACO_seed: {aco_seed_score}, "
            f"ACO_end: {aco_end_score}, "
            f"Polish_seed: {polish_seed_score}, "
            f"Polish_end: {polish_end_score}, "
            f"actions: {action_stats['total_actions']}/{action_stats['effective_actions']} "
            f"(eff_ratio: {action_stats['effective_ratio']:.2%}), "
            f"time: {end_time - start_time:.2f}s, "
            f"result: {result}",
            flush=True,
        )

        if current_score == 0:
            print(f"[Perfect solution found in episode {episodes}!]", flush=True)
            break

    rlaco_elapsed = time.time() - tot_start
    rlaco_best_score = int(best_score)
    rlaco_best_solution = (
        None if best_solution is None else np.asarray(best_solution, dtype=np.int32).copy()
    )
    rlaco_best_episode = best_episode

    terminal_blocks = []
    terminal_start_score = None
    terminal_best_score = best_score
    terminal_best_solution = (
        None if best_solution is None else np.asarray(best_solution, dtype=np.int32).copy()
    )
    terminal_elapsed = 0.0
    terminal_backend = None

    # Priority 3 terminal stage. This is deliberately outside the episode
    # loop and is different from the existing per-episode RLACOT polish.
    terminal_time_to_feasibility = None
    if args.terminal_tabucol and terminal_best_score != 0:
        if terminal_best_solution is None:
            terminal_best_solution = np.asarray(solution, dtype=np.int32).copy()
            terminal_best_score = calculate_score(graph, terminal_best_solution)

        terminal_start_score = int(terminal_best_score)
        terminal_budget = (
            args.time_limit
            if args.terminal_time_limit is None
            else args.terminal_time_limit
        )
        if terminal_budget <= 0:
            raise ValueError("--terminal-time-limit must be positive.")
        if args.terminal_tabu_iters <= 0:
            raise ValueError("--terminal-tabu-iters must be positive.")

        import tabucol
        terminal_backend = getattr(tabucol, "BACKEND_ORIGIN", None)

        if not getattr(tabucol, "IS_NATIVE", False) and not args.allow_python_tabucol:
            raise RuntimeError(
                "Formal terminal experiments require the native Rust/PyO3 "
                "Tabucol extension. Build src/tabucol with maturin first."
            )
        if not hasattr(tabucol.TabuColSolver, "get_best_solution"):
            raise RuntimeError(
                "The loaded Tabucol extension is stale and does not expose "
                "get_best_solution(). Rebuild src/tabucol before running the "
                "formal experiment."
            )

        adj_list = [[] for _ in range(len(graph))]
        for node in graph:
            for neighbor in graph.neighbors(node):
                adj_list[node].append(neighbor)

        terminal_solver = tabucol.TabuColSolver(
            adj_list,
            args.k,
            max_iterations=args.terminal_tabu_iters,
            tabu_a=10,
            tabu_alpha=1.2,
            beta=args.beta,
        )
        terminal_clock = time.time()
        terminal_current_solution = terminal_best_solution.copy()
        block = 0
        while time.time() - terminal_clock < terminal_budget:
            block += 1
            block_start = time.time()
            block_start_score = int(calculate_score(graph, terminal_current_solution))
            # set_solution resets both the tabu table and the iteration counter.
            terminal_solver.set_solution(terminal_current_solution.astype(int).tolist())
            returned_solution, reward = terminal_solver.solve()
            returned_solution = np.asarray(returned_solution, dtype=np.int32)
            returned_score = calculate_score(graph, returned_solution)
            block_best_solution = np.asarray(
                terminal_solver.get_best_solution(), dtype=np.int32
            )
            block_best_score = calculate_score(graph, block_best_solution)
            solver_best_score = int(terminal_solver.get_best_score())
            if block_best_score != solver_best_score:
                raise RuntimeError(
                    "Tabucol best score/solution mismatch: "
                    f"reported={solver_best_score}, recomputed={block_best_score}."
                )
            terminal_current_solution = returned_solution.copy()
            if block_best_score < terminal_best_score:
                terminal_best_score = int(block_best_score)
                terminal_best_solution = block_best_solution.copy()
            block_end = time.time()
            terminal_blocks.append({
                "block": block,
                "start_score": block_start_score,
                "returned_score": int(returned_score),
                "block_best_score": int(block_best_score),
                "global_best_score": int(terminal_best_score),
                "reward": int(reward),
                "iterations": int(args.terminal_tabu_iters),
                "elapsed": block_end - block_start,
                "tabu_table_reset": True,
            })
            print(
                f"[Terminal TabuCol] block={block}, returned={returned_score}, "
                f"block_best={block_best_score}, "
                f"global_best={terminal_best_score}, "
                f"elapsed={block_end - block_start:.2f}s",
                flush=True,
            )
            if terminal_best_score == 0:
                terminal_time_to_feasibility = block_end - terminal_clock
                break
        terminal_elapsed = time.time() - terminal_clock
        best_score = int(terminal_best_score)
        best_solution = terminal_best_solution.copy()
        solution = best_solution
    elif args.terminal_tabucol:
        terminal_start_score = 0
        terminal_best_score = 0
        terminal_best_solution = rlaco_best_solution.copy()
        terminal_time_to_feasibility = 0.0
        best_score = 0
        solution = terminal_best_solution

    if not args.terminal_tabucol and best_solution is not None:
        solution = best_solution.copy()

    tot_end = time.time()
    if args.use_aco:
        iterations = episodes * (
            args.max_steps + args.aco_ants * args.aco_iter
        )
    else:
        iterations = episodes * (
            args.max_steps + args.max_tabucol_iters
        )
    iterations += len(terminal_blocks) * args.terminal_tabu_iters
    iterations = max(iterations, 1)

    print("[DONE]")
    json_result = {
        "mode": mode_label,
        "score": best_score,
        "episodes": episodes,
        "iterations": iterations,
        "time": tot_end - tot_start,
        "time_per_iteration": (tot_end - tot_start) / iterations,
        "solution": np.asarray(solution, dtype=np.int32).tolist(),
        "graph": args.graph,
        "k": args.k,
        "max_steps": args.max_steps,
        "max_tabucol_iters": args.max_tabucol_iters,
        "policy": args.policy,
        "RL": args.RL,
        "use_aco": args.use_aco,
        "use_conflict_prior": args.use_conflict_prior,
        "inject_mode": args.inject_mode if args.use_aco and args.RL else None,
        "polish_tabucol": args.polish_tabucol,
        "terminal_tabucol": args.terminal_tabucol,
        "terminal_time_limit": (
            args.terminal_time_limit
            if args.terminal_time_limit is not None
            else args.time_limit
        ) if args.terminal_tabucol else None,
        "terminal_tabu_iters": args.terminal_tabu_iters if args.terminal_tabucol else None,
        "terminal_start_score": terminal_start_score,
        "terminal_best_score": terminal_best_score if args.terminal_tabucol else None,
        "terminal_elapsed": terminal_elapsed,
        "terminal_time_to_feasibility": terminal_time_to_feasibility,
        "terminal_blocks": terminal_blocks,
        "rlaco_elapsed": rlaco_elapsed,
        "rlaco_best_score": rlaco_best_score,
        "rlaco_best_solution": (
            rlaco_best_solution.tolist() if rlaco_best_solution is not None else None
        ),
        "rlaco_best_episode": rlaco_best_episode,
        "seed": args.seed,
        "tabucol_backend": terminal_backend,
        "aco_ants": args.aco_ants,
        "aco_iter": args.aco_iter,
        "last_rl_start_score": getattr(env.env, "last_episode_rl_start_score", None),
        "last_rl_end_score": getattr(env.env, "last_episode_rl_end_score", None),
        "last_aco_seed_score": getattr(env.env, "last_episode_aco_seed_score", None),
        "last_aco_end_score": getattr(env.env, "last_episode_aco_end_score", None),
        "last_polish_seed_score": getattr(env.env, "last_episode_polish_seed_score", None),
        "last_polish_end_score": getattr(env.env, "last_episode_polish_end_score", None),
        "episode_traces": episode_traces,
    }

    # JSONIFY
    print(json.dumps(json_result, default=str), flush=True)

    if args.result_json is not None:
        args.result_json.parent.mkdir(parents=True, exist_ok=True)
        temporary = args.result_json.with_suffix(args.result_json.suffix + ".tmp")
        temporary.write_text(json.dumps(json_result, indent=2), encoding="utf-8")
        temporary.replace(args.result_json)

    # exit process
    os._exit(0)
