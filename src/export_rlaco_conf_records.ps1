$ErrorActionPreference = "Stop"

$sourceRoot = Join-Path $PSScriptRoot "main_experiment"
$exportRoot = Join-Path $PSScriptRoot "..\..\deliverables\RLACO_conf_le450_experiment_records_20260813"
$rawRoot = Join-Path $exportRoot "raw_logs"

$experiments = @(
    @{ instance = "le450_15c"; k = 15; directory = "le450_15c_k15_rlaco_conf" },
    @{ instance = "le450_15d"; k = 15; directory = "le450_15d_k15_rlaco_conf" },
    @{ instance = "le450_25c"; k = 25; directory = "le450_25c_k25_rlaco_conf" },
    @{ instance = "le450_25d"; k = 25; directory = "le450_25d_k25_rlaco_conf" }
)

New-Item -ItemType Directory -Force -Path $rawRoot | Out-Null
$runRows = @()

foreach ($experiment in $experiments) {
    $sourceDir = Join-Path $sourceRoot $experiment.directory
    $targetDir = Join-Path $rawRoot $experiment.directory
    Copy-Item -LiteralPath $sourceDir -Destination $targetDir -Recurse -Force

    $runIndex = 0
    foreach ($file in Get-ChildItem -LiteralPath $sourceDir -File -Filter "*.txt" | Sort-Object CreationTime) {
        $runIndex++
        $jsonLine = $null
        foreach ($line in [System.IO.File]::ReadLines($file.FullName)) {
            if ($line.StartsWith('{"mode"')) {
                $jsonLine = $line
            }
        }
        if ($null -eq $jsonLine) {
            throw "Missing final JSON result in $($file.FullName)"
        }

        $result = $jsonLine | ConvertFrom-Json
        $runRows += [pscustomobject]@{
            instance = $experiment.instance
            k = $experiment.k
            run = $runIndex
            score = [double]$result.score
            runtime_seconds = [math]::Round([double]$result.time, 3)
            runtime_hours = [math]::Round([double]$result.time / 3600, 4)
            episodes = [int]$result.episodes
            policy = [string]$result.policy
            rl_steps_per_episode = [int]$result.max_steps
            use_conflict_prior = [bool]$result.use_conflict_prior
            injection_mode = [string]$result.inject_mode
            aco_ants = [int]$result.aco_ants
            aco_iterations = [int]$result.aco_iter
            source_file = $file.Name
        }
    }
}

$runRows | Export-Csv -LiteralPath (Join-Path $exportRoot "per_run_results.csv") -NoTypeInformation -Encoding UTF8

$summaryRows = foreach ($group in $runRows | Group-Object instance) {
    $rows = @($group.Group)
    $scores = @($rows | ForEach-Object { [double]$_.score })
    $mean = ($scores | Measure-Object -Average).Average
    $sumSquares = 0.0
    foreach ($score in $scores) {
        $sumSquares += [math]::Pow($score - $mean, 2)
    }
    $sampleSd = if ($scores.Count -gt 1) {
        [math]::Sqrt($sumSquares / ($scores.Count - 1))
    } else {
        0.0
    }

    [pscustomobject]@{
        instance = $group.Name
        k = $rows[0].k
        runs = $scores.Count
        scores = $scores -join "; "
        mean_score = [math]::Round($mean, 2)
        sample_sd = [math]::Round($sampleSd, 2)
        best_score = ($scores | Measure-Object -Minimum).Minimum
        worst_score = ($scores | Measure-Object -Maximum).Maximum
        zero_score_runs = @($scores | Where-Object { $_ -eq 0 }).Count
        success_rate = [math]::Round(@($scores | Where-Object { $_ -eq 0 }).Count / $scores.Count, 4)
        mean_runtime_hours = [math]::Round((($rows.runtime_hours | Measure-Object -Average).Average), 4)
    }
}

$summaryRows | Export-Csv -LiteralPath (Join-Path $exportRoot "summary_results.csv") -NoTypeInformation -Encoding UTF8

Copy-Item -LiteralPath (Join-Path $sourceRoot "rlaco_conf_le450_campaign_run2.stdout.log") -Destination $exportRoot -Force
Copy-Item -LiteralPath (Join-Path $sourceRoot "rlaco_conf_le450_campaign_run2.stderr.log") -Destination $exportRoot -Force
Copy-Item -LiteralPath (Join-Path $PSScriptRoot "run_rlaco_conf_le450.ps1") -Destination $exportRoot -Force

$readme = @"
RLACO_conf le450 experiment records

Configuration:
- policy: policy.pt
- RL steps per episode: 300
- structured prior: per-episode conflict-count prior from post-RL observation
- injection: soft
- ACO ants: 40
- ACO iterations: 50
- time limit: 10,800 seconds per run
- runs: 10 per instance
- concurrency: 2
- keep_rl_best: false
- TabuCol polish: disabled

Contents:
- per_run_results.csv: one row per completed run
- summary_results.csv: aggregate statistics per instance
- raw_logs/: original 40 result logs, unchanged
- campaign stdout/stderr logs
- run_rlaco_conf_le450.ps1: campaign command script

Sample standard deviation is reported in summary_results.csv.
"@
Set-Content -LiteralPath (Join-Path $exportRoot "README.txt") -Value $readme -Encoding UTF8

$zipPath = "$exportRoot.zip"
if (Test-Path -LiteralPath $zipPath) {
    Remove-Item -LiteralPath $zipPath -Force
}
Compress-Archive -LiteralPath $exportRoot -DestinationPath $zipPath -CompressionLevel Optimal

Write-Output $exportRoot
Write-Output $zipPath
