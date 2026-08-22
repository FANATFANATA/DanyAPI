$ErrorActionPreference = "Stop"

$RepoUrl = "https://github.com/FANATFANATA/DanyAPI"
$Branch = "main"
$ZipUrl = "$RepoUrl/archive/refs/heads/$Branch.zip"
$Target = $env:DANYAPI_DIR
if (-not $Target) { $Target = Join-Path $HOME "DanyAPI" }

function Find-Python {
    foreach ($name in @("py", "python")) {
        $cmd = Get-Command $name -ErrorAction SilentlyContinue
        if ($cmd) { return $cmd.Source }
    }
    return $null
}

function Install-FromZip {
    param([string]$Dest)
    $zip = Join-Path $env:TEMP "danyapi.zip"
    $extract = Join-Path $env:TEMP "danyapi-extract"
    Write-Host "Downloading $ZipUrl"
    Invoke-WebRequest -Uri $ZipUrl -OutFile $zip
    if (Test-Path $extract) { Remove-Item -Recurse -Force $extract }
    Expand-Archive -Path $zip -DestinationPath $extract
    $src = Join-Path $extract "DanyAPI-$Branch"
    if (-not (Test-Path $src)) { throw "Unexpected archive layout" }
    if (Test-Path $Dest) { Remove-Item -Recurse -Force $Dest }
    Move-Item $src $Dest
    Remove-Item -Force $zip
}

$python = Find-Python
if (-not $python) {
    Write-Host ""
    Write-Host "Python was not found. Install Python 3.10+ from https://www.python.org/downloads/ and run the command again."
    exit 1
}

Write-Host "DanyAPI will be installed into: $Target"

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

$setup = Join-Path $Target "docs\setup.py"
if (-not (Test-Path $setup)) {
    Write-Host "Could not find $setup in the checkout."
    exit 1
}

& $python $setup
exit $LASTEXITCODE
