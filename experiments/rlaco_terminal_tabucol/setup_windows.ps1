$ErrorActionPreference = "Stop"

$ProjectRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
$Requirements = Join-Path $PSScriptRoot "requirements-experiment.txt"
$TabucolRoot = Join-Path $ProjectRoot "src\tabucol"
$VenvRoot = Join-Path $ProjectRoot ".venv"
$Python = Join-Path $VenvRoot "Scripts\python.exe"

function Import-VisualStudioEnvironment {
    $VsWhere = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
    if (-not (Test-Path $VsWhere)) {
        return
    }
    $Installation = & $VsWhere -latest -products * `
        -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 `
        -property installationPath
    if (-not $Installation) {
        return
    }
    $DevCmd = Join-Path $Installation "Common7\Tools\VsDevCmd.bat"
    if (-not (Test-Path $DevCmd)) {
        return
    }
    $EnvironmentLines = & cmd.exe /s /c "`"$DevCmd`" -arch=x64 -host_arch=x64 >nul && set"
    foreach ($Line in $EnvironmentLines) {
        $Parts = $Line -split "=", 2
        if ($Parts.Count -eq 2) {
            Set-Item -Path "Env:$($Parts[0])" -Value $Parts[1]
        }
    }
}

if (-not (Test-Path $Python)) {
    python -m venv --system-site-packages $VenvRoot
}

$env:VIRTUAL_ENV = $VenvRoot
$env:PATH = "$(Join-Path $VenvRoot 'Scripts');$env:PATH"
& $Python -m pip install --upgrade pip
& $Python -m pip install -r $Requirements

if (-not (Get-Command cargo -ErrorAction SilentlyContinue)) {
    throw "Rust/Cargo is missing. Install rustup from https://rustup.rs, reopen PowerShell, and rerun this script."
}
if (-not (Get-Command link.exe -ErrorAction SilentlyContinue)) {
    Import-VisualStudioEnvironment
}
if (-not (Get-Command link.exe -ErrorAction SilentlyContinue)) {
    throw "MSVC linker is missing. Install Visual Studio 2022 Build Tools with 'Desktop development with C++', then rerun this script."
}

$env:PYO3_USE_ABI3_FORWARD_COMPATIBILITY = "1"
Push-Location $TabucolRoot
try {
    & $Python -m maturin develop --release
}
finally {
    Pop-Location
}

& $Python (Join-Path $PSScriptRoot "verify_install.py")

Write-Host "Setup complete. Use: $Python"
