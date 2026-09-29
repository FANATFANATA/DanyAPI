$ErrorActionPreference = "Stop"

$RepoUrl = "https://github.com/FANATFANATA/DanyAPI"
$Branch = "prod"
$ZipUrl = "$RepoUrl/archive/refs/heads/$Branch.zip"
$Target = $env:DANYAPI_DIR
if (-not $Target) { $Target = Join-Path $HOME "DanyAPI" }
$MinPython = [version]"3.10"
$script:EnvBackup = $null

function Get-PythonVersion {
    param([string]$Exe, [string[]]$Prefix)
    try {
        $output = & $Exe @Prefix -c "import sys; print('%d.%d' % sys.version_info[:2])" 2>$null
    }
    catch {
        return $null
    }
    if ($LASTEXITCODE -ne 0) { return $null }
    $line = @($output)[-1]
    if (-not $line) { return $null }
    return $line.ToString().Trim()
}

function Find-Python {
    foreach ($name in @("py", "python", "python3")) {
        $cmd = Get-Command $name -ErrorAction SilentlyContinue
        if (-not $cmd) { continue }
        $prefix = if ($name -eq "py") { @("-3") } else { @() }
        $version = $null
        $reported = Get-PythonVersion -Exe $cmd.Source -Prefix $prefix
        if (-not $reported) { continue }
        if ([version]::TryParse($reported, [ref]$version) -and $version -ge $MinPython) {
            return [pscustomobject]@{ Exe = $cmd.Source; Prefix = $prefix }
        }
    }
    return $null
}

function Save-Env {
    $script:EnvBackup = $null
    $envFile = Join-Path $Target ".env"
    if (Test-Path $envFile) {
        $script:EnvBackup = (Join-Path (Split-Path -Parent $Target) ((Split-Path -Leaf $Target) + ".env.danyapi-backup"))
        Copy-Item $envFile $script:EnvBackup -Force
    }
}

function Restore-Env {
    if (-not $script:EnvBackup) { return }
    $envFile = Join-Path $Target ".env"
    $backup = $script:EnvBackup
    $script:EnvBackup = $null
    if (Test-Path $envFile) {
        Remove-Item -Force $backup -ErrorAction SilentlyContinue
        return
    }
    try {
        New-Item -ItemType Directory -Force -Path $Target | Out-Null
        Move-Item $backup $envFile -Force
        Write-Host "Restored your existing $envFile"
    }
    catch {
        Write-Host "Could not restore $envFile, your settings are still in $backup" -ForegroundColor Red
    }
}

# Refuse to delete a target that is neither empty nor recognisable as a DanyAPI
# install, so a mistyped DANYAPI_DIR cannot destroy an unrelated directory.
function Test-TargetRemovable {
    param([string]$Dest)
    if (-not (Test-Path $Dest)) { return $true }
    if ((Test-Path (Join-Path $Dest ".git")) -or (Test-Path (Join-Path $Dest "app.py")) -or (Test-Path (Join-Path $Dest "docs\setup.py"))) {
        return $true
    }
    $items = @(Get-ChildItem -Force -LiteralPath $Dest -ErrorAction SilentlyContinue)
    if ($items.Count -eq 0) { return $true }
    return $false
}

function Remove-Target {
    param([string]$Dest)
    if (-not (Test-Path $Dest)) { return }
    if (Test-TargetRemovable -Dest $Dest) {
        Remove-Item -Recurse -Force $Dest
        return
    }
    throw "$Dest is not empty and does not look like a DanyAPI install, refusing to delete it."
}

function Install-FromZip {
    param([string]$Dest)
    $work = Join-Path ([System.IO.Path]::GetTempPath()) ("danyapi-" + [System.Guid]::NewGuid().ToString("N"))
    New-Item -ItemType Directory -Force -Path $work | Out-Null
    $zip = Join-Path $work "danyapi.zip"
    $extract = Join-Path $work "extracted"
    try {
        Write-Host "Downloading $ZipUrl"
        Write-Host "No checksum is published for this archive, so it is applied unverified."
        Invoke-WebRequest -Uri $ZipUrl -OutFile $zip -UseBasicParsing
        Expand-Archive -Path $zip -DestinationPath $extract -Force
        $src = Join-Path $extract "DanyAPI-$Branch"
        if (-not (Test-Path (Join-Path $src "app.py")) -or -not (Test-Path (Join-Path $src "docs\setup.py"))) {
            throw "The downloaded archive does not contain a DanyAPI checkout."
        }
        Remove-Target -Dest $Dest
        Move-Item $src $Dest
    }
    finally {
        Remove-Item -Recurse -Force $work -ErrorAction SilentlyContinue
    }
}

function Install-Venv {
    param([pscustomobject]$Python, [string]$Dest)
    $venv = Join-Path $Dest ".venv"
    $exe = Join-Path $venv "Scripts\python.exe"
    if (Test-Path $exe) { return $exe }
    Write-Host "Creating a virtualenv in $venv"
    & $Python.Exe @($Python.Prefix) -m venv $venv | Out-Null
    if ($LASTEXITCODE -ne 0 -or -not (Test-Path $exe)) {
        throw "Could not create a virtualenv."
    }
    return $exe
}

$python = Find-Python
if (-not $python) {
    Write-Host ""
    Write-Host "Python 3.10+ was not found. Install it from https://www.python.org/downloads/ and run the command again."
    exit 1
}

Write-Host "DanyAPI will be installed into: $Target"

try {
    Save-Env

    if (Test-Path (Join-Path $Target ".git")) {
        Write-Host "Updating existing checkout..."
        Push-Location $Target
        try {
            & git pull --ff-only
            if ($LASTEXITCODE -ne 0) { throw "git pull failed" }
        }
        finally {
            Pop-Location
        }
    }
    elseif (Get-Command git -ErrorAction SilentlyContinue) {
        Remove-Target -Dest $Target
        Write-Host "Cloning $RepoUrl ..."
        & git clone $RepoUrl $Target
        if ($LASTEXITCODE -ne 0) {
            Write-Host "git clone failed, trying the source archive."
            Remove-Target -Dest $Target
            Install-FromZip -Dest $Target
        }
    }
    else {
        Write-Host "git not found, downloading the source archive instead."
        Install-FromZip -Dest $Target
    }

    Restore-Env

    $setup = Join-Path $Target "docs\setup.py"
    if (-not (Test-Path $setup)) {
        throw "Could not find $setup in the checkout."
    }

    $venvPython = Install-Venv -Python $python -Dest $Target
}
catch {
    Write-Host ""
    Write-Host $_.Exception.Message -ForegroundColor Red
    Restore-Env
    exit 1
}
finally {
    Restore-Env
}

& $venvPython $setup
exit $LASTEXITCODE
