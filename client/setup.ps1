# Client setup for Windows: Python 3.12, git, network time, then a check run.
#   powershell -ExecutionPolicy Bypass -File client\setup.ps1
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

function Has($name) { [bool](Get-Command $name -ErrorAction SilentlyContinue) }

$installed = $false
if (-not (Has "py")) {
    winget install -e --id Python.Python.3.12 --accept-source-agreements --accept-package-agreements
    $installed = $true
}
if (-not (Has "git")) {
    winget install -e --id Git.Git --accept-source-agreements --accept-package-agreements
    $installed = $true
}
if ($installed) {
    Write-Host "Installed Python/git. Close this window, open a new PowerShell and rerun setup.ps1."
    exit 0
}
py -3 -c "import sys; sys.exit(sys.version_info < (3, 10))"
if ($LASTEXITCODE -ne 0) { Write-Host "Python 3.10+ needed: winget install Python.Python.3.12"; exit 1 }
Write-Host "python: $(py -3 --version)   git: $(git --version)"

Write-Host "== network time (resync needs an Administrator PowerShell)"
try { w32tm /resync | Out-Host } catch { Write-Host "resync skipped: run as Administrator to force it" }
w32tm /query /status | Out-Host

if (-not (Test-Path ".env")) {
    Copy-Item ".env.example" ".env"
    Write-Host "Created client\.env: fill in SERVER_URL and the server details, then run:"
    Write-Host "  py -3 client\run.py --check"
    exit 0
}
py -3 run.py --check
