$ErrorActionPreference = "Stop"

$python = (Get-Command python -ErrorAction Stop).Source
$jobs = @(
    @{ graph = "le450_15c.col"; k = 15; output = "main_experiment/le450_15c_k15_rlaco_conf" },
    @{ graph = "le450_15d.col"; k = 15; output = "main_experiment/le450_15d_k15_rlaco_conf" },
    @{ graph = "le450_25c.col"; k = 25; output = "main_experiment/le450_25c_k25_rlaco_conf" },
    @{ graph = "le450_25d.col"; k = 25; output = "main_experiment/le450_25d_k25_rlaco_conf" }
)

foreach ($job in $jobs) {
    & $python batch_runner.py policy.pt $job.graph $job.k $job.output `
        --RL True --use-conflict-prior `
        --aco-ants 40 --aco-iter 50 `
        --time-limit 10800 --num-jobs 10 --concurrency 2

    if ($LASTEXITCODE -ne 0) {
        throw "Experiment failed for $($job.graph) with exit code $LASTEXITCODE"
    }
}
