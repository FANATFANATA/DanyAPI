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
        $script:EnvBackup = "$Target.env.danyapi-backup"
        Copy-Item $envFile $script:EnvBackup -Force
    }
}

function Restore-Env {
    if (-not $script:EnvBackup) { return }
    $envFile = Join-Path $Target ".env"
    $backup = $script:EnvBackup
    $script:EnvBackup = $null
    if (Test-Path $envFile) {
        Remove-Item -Force $backup
        return
    }
    try {
        Move-Item $backup $envFile -Force
        Write-Host "Restored your existing $envFile"
    }
    catch {
        Write-Host "Could not restore $envFile, your settings are still in $backup" -ForegroundColor Red
    }
}

function Install-FromZip {
    param([string]$Dest)
    $zip = Join-Path $env:TEMP "danyapi.zip"
    $extract = Join-Path $env:TEMP "danyapi-extract"
    Write-Host "Downloading $ZipUrl"
    Invoke-WebRequest -Uri $ZipUrl -OutFile $zip -UseBasicParsing
    if (Test-Path $extract) { Remove-Item -Recurse -Force $extract }
    try {
        Expand-Archive -Path $zip -DestinationPath $extract
        $src = Join-Path $extract "DanyAPI-$Branch"
        if (-not (Test-Path $src)) { throw "Unexpected archive layout" }
        if (Test-Path $Dest) { Remove-Item -Recurse -Force $Dest }
        Move-Item $src $Dest
    }
    finally {
        Remove-Item -Recurse -Force $extract -ErrorAction SilentlyContinue
        Remove-Item -Force $zip -ErrorAction SilentlyContinue
    }
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
        if (Test-Path $Target) { Remove-Item -Recurse -Force $Target }
        Write-Host "Cloning $RepoUrl ..."
        & git clone $RepoUrl $Target
        if ($LASTEXITCODE -ne 0) {
            Write-Host "git clone failed, trying the source archive."
            if (Test-Path $Target) { Remove-Item -Recurse -Force $Target }
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
        Write-Host "Could not find $setup in the checkout."
        exit 1
    }
}
finally {
    if ($script:EnvBackup -and (Test-Path $script:EnvBackup)) {
        Remove-Item -Force $script:EnvBackup -ErrorAction SilentlyContinue
    }
}

& $python.Exe @($python.Prefix) $setup
exit $LASTEXITCODE
