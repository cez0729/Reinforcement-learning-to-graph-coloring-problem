import argparse
import os
import subprocess
import time
from datetime import datetime
from multiprocessing.dummy import Pool


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run batch of experiments usin the runner.py script"
    )
    parser.add_argument("policy", type=str, help="Policy to use")
    parser.add_argument("graph", type=str, help="Path to graph file")
    parser.add_argument("k", type=int, help="Number of colors")
    parser.add_argument("output_dir", type=str, help="Output directory")
    parser.add_argument(
        "-I",
        "--max-steps",
        type=int,
        dest="max_steps",
        default=300,
        help="Max RL steps per episode (default 300, matches training distribution)",
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
        help="Time limit per job in seconds (default: 18000 = 5 hours, matching the RLTabucol testing budget)",
    )
    parser.add_argument(
        "--RL", type=str2bool, nargs="?", const=True, default=True, help="Use RL or not"
    )
    parser.add_argument(
        "--num-jobs",
        type=int,
        dest="num_jobs",
        default=10,
        help="Number of independent runs per instance (each run is a different random seed)",
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        dest="concurrency",
        default=2,
        help="Number of jobs to run in parallel. Keep at 1 to avoid CPU contention skewing wall-clock budgets",
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
        help="Use ACO at end of each episode instead of TabuCol (default: False)"
    )
    parser.add_argument(
        "--disable-tabucol",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        dest="disable_tabucol",
        help="Disable Tabucol handoff entirely (actor-only mode). Default: False"
    )
    parser.add_argument(
        "--inject-mode",
        type=str,
        choices=["hard", "soft"],
        default="hard",
        dest="inject_mode",
        help="Pheromone injection mode for RL+ACO: hard or soft (default: hard)"
    )
    parser.add_argument(
        "--polish-tabucol",
        action="store_true",
        default=False,
        dest="polish_tabucol",
        help=(
            "Forward --polish-tabucol to runner.py: run TabuCol as a polish "
            "stage after ACO inside each episode (RLACOT pipeline)."
        ),
    )
    parser.add_argument(
        "--terminal-tabucol",
        action="store_true",
        default=False,
        dest="terminal_tabucol",
        help="Run the Priority 3 terminal TabuCol stage after the RL/ACO stage.",
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
        help="Iterations per terminal TabuCol block (default: 10000).",
    )
    parser.add_argument(
        "--keep-rl-best",
        action="store_true",
        default=False,
        dest="keep_rl_best",
        help="Forward --keep-rl-best to runner.py (restore RL phase best frame).",
    )
    parser.add_argument(
        "--restart-every",
        type=int,
        dest="restart_every",
        default=0,
        help="Forward --restart-every to runner.py (perturbation restart period; 0 = off).",
    )
    parser.add_argument(
        "--restart-perturb",
        type=float,
        dest="restart_perturb",
        default=0.2,
        help="Forward --restart-perturb to runner.py (fraction of vertices recolored).",
    )
    parser.add_argument(
        "--pure-aco",
        action="store_true",
        default=False,
        dest="pure_aco",
        help=(
            "Forward --pure-aco to runner.py: run the Pure ACO baseline "
            "(no RL, no env, direct AntColony.run() with tau reset each "
            "episode). Overrides --RL/--use-aco in runner.py."
        ),
    )
    parser.add_argument(
        "--use-conflict-prior",
        action="store_true",
        default=False,
        dest="use_conflict_prior",
        help=(
            "Forward --use-conflict-prior to runner.py: run the RLACO_conf "
            "baseline (same trained policy and RL phase as RLACO; the ACO "
            "soft prior is an explicit per-episode conflict-count "
            "heuristic instead of the actor's learned distribution). "
            "Requires --RL True (default) with a real policy checkpoint."
        ),
    )

    args = parser.parse_args()

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    pool = Pool(args.concurrency)

    slack_ts = None
    big_time = time.time()

    def run_job(i):
        start_time = time.time()
        now = datetime.now()
        timestamp = now.strftime("%Y%m%d_%H%M%S_%f")
        print(f"Starting job {i + 1}")
        f = open(f"{args.output_dir}/{args.k}_{timestamp}.txt", "w")
        aco_args = ""
        if args.use_aco:
            aco_args = f" --use-aco --aco-ants {args.aco_ants} --aco-iter {args.aco_iter} --inject-mode {args.inject_mode}"
        polish_args = " --polish-tabucol" if args.polish_tabucol else ""
        terminal_args = ""
        if args.terminal_tabucol:
            terminal_args = " --terminal-tabucol"
            if args.terminal_time_limit is not None:
                terminal_args += f" --terminal-time-limit {args.terminal_time_limit}"
            terminal_args += f" --terminal-tabu-iters {args.terminal_tabu_iters}"
        keep_args = " --keep-rl-best" if args.keep_rl_best else ""
        disable_tabucol_args = " --disable-tabucol True" if args.disable_tabucol else ""
        restart_args = ""
        if args.restart_every > 0:
            restart_args = f" --restart-every {args.restart_every} --restart-perturb {args.restart_perturb}"
        if args.pure_aco:
            cmd = f'python runner.py "{args.policy}" "{args.graph}" {args.k} --time-limit {args.time_limit} --pure-aco --aco-ants {args.aco_ants} --aco-iter {args.aco_iter}'
        elif args.use_conflict_prior:
            # RLACO_conf: same RL phase as RLACO (--RL True with a real
            # policy checkpoint), only the ACO prior source differs.
            cmd = f'python runner.py "{args.policy}" "{args.graph}" {args.k} -I {args.max_steps} -T {args.max_tabucol_iters} -B {args.beta} --time-limit {args.time_limit} --RL {args.RL} --use-conflict-prior --aco-ants {args.aco_ants} --aco-iter {args.aco_iter}{terminal_args}'
        else:
            cmd = f'python runner.py "{args.policy}" "{args.graph}" {args.k} -I {args.max_steps} -T {args.max_tabucol_iters} -B {args.beta} --time-limit {args.time_limit} --RL {args.RL}{aco_args}{polish_args}{keep_args}{disable_tabucol_args}{restart_args}{terminal_args}'
        print("running: ", cmd)
        process = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in process.stdout:
            print(line, end='', flush=True)
            f.write(line)
            f.flush()
        process.wait()
        f.close()
        end_time = time.time()

        print(f"Finished job {i + 1}", flush=True)

    result = pool.map_async(run_job, range(args.num_jobs))
    result.wait()

    big_end = time.time()
    print("All jobs finished", flush=True)
