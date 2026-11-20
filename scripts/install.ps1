# VibeSound installer for Windows. Run from the repository folder:
#   powershell -ExecutionPolicy Bypass -File scripts\install.ps1
$ErrorActionPreference = "Stop"
$repo = Split-Path -Parent $PSScriptRoot
Set-Location $repo

$py = if (Get-Command py -ErrorAction SilentlyContinue) { "py" } elseif (Get-Command python -ErrorAction SilentlyContinue) { "python" } else { $null }
if (-not $py) { throw "Python was not found. Install Python 3.10+ from https://www.python.org/downloads/windows/ and tick 'Add python.exe to PATH'." }

Write-Host "Creating Python environment in $repo\.venv ..."
if (-not (Test-Path ".venv\Scripts\python.exe")) { & $py -m venv .venv }
$venvPy = Join-Path $repo ".venv\Scripts\python.exe"
& $venvPy -m pip install --upgrade pip --quiet
& $venvPy -m pip install -e . --quiet
& $venvPy -c "import vibesound.server; print('VibeSound', vibesound.__version__, 'installed')"
if ($LASTEXITCODE -ne 0) { throw "VibeSound did not install correctly." }

# Claude Desktop keeps its config in %APPDATA%\Claude (or under Packages for the Microsoft Store build).
$candidates = @("$env:APPDATA\Claude")
$store = Get-ChildItem "$env:LOCALAPPDATA\Packages" -Filter "Claude_*" -Directory -ErrorAction SilentlyContinue | Select-Object -First 1
if ($store) { $candidates = @("$($store.FullName)\LocalCache\Roaming\Claude") + $candidates }
$dir = $candidates | Where-Object { Test-Path $_ } | Select-Object -First 1
if (-not $dir) { $dir = "$env:APPDATA\Claude"; New-Item -ItemType Directory -Force -Path $dir | Out-Null }
$cfgPath = Join-Path $dir "claude_desktop_config.json"

$cfg = [pscustomobject]@{}
if (Test-Path $cfgPath) {
    $raw = Get-Content $cfgPath -Raw
    if ($raw -and $raw.Trim()) {
        Copy-Item $cfgPath "$cfgPath.bak" -Force
        $cfg = $raw | ConvertFrom-Json
    }
}
if (-not $cfg.PSObject.Properties["mcpServers"]) { $cfg | Add-Member -NotePropertyName mcpServers -NotePropertyValue ([pscustomobject]@{}) }
$entry = [pscustomobject]@{ command = $venvPy; args = @("-m", "vibesound.server") }
$cfg.mcpServers | Add-Member -NotePropertyName vibesound -NotePropertyValue $entry -Force
$json = $cfg | ConvertTo-Json -Depth 20
[System.IO.File]::WriteAllText($cfgPath, $json, (New-Object System.Text.UTF8Encoding($false)))

Write-Host ""
Write-Host "Done. Added 'vibesound' to $cfgPath"
Write-Host "Now quit Claude Desktop fully (system tray icon, Quit) and reopen it."
