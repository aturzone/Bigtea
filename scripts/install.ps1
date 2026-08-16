<#
.SYNOPSIS
    Install or upgrade Chaos on Windows. CLI only -- there is no GUI and no
    .exe installer, on purpose.

.DESCRIPTION
    Copies the binaries next to this script into a prefix, puts that prefix on
    the *user* PATH, and creates a models directory for .gguf files.

    Re-running it over an older version upgrades in place: binaries the new
    version no longer ships are removed, the PATH entry is not duplicated, and
    the models directory is never touched.

    Nothing here needs administrator rights. It writes to two places only --
    the prefix, and the per-user Environment key in the registry.

.PARAMETER Prefix
    Where the binaries go. Default: %LOCALAPPDATA%\Chaos

.PARAMETER ModelsDir
    Where you will put .gguf files. Default: %USERPROFILE%\.chaos\models
    Created if missing, never written to or cleaned up.

.PARAMETER NoPath
    Copy the binaries but leave PATH alone.

.PARAMETER Uninstall
    Remove the prefix and the PATH entry. Models are left where they are.

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File .\install.ps1

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File .\install.ps1 -Prefix D:\tools\chaos

.EXAMPLE
    powershell -ExecutionPolicy Bypass -File .\install.ps1 -Uninstall
#>
[CmdletBinding()]
param(
    [string] $Prefix,
    [string] $ModelsDir,
    [switch] $NoPath,
    [switch] $Uninstall
)

$ErrorActionPreference = 'Stop'

# -- where things go ---------------------------------------------------------

if (-not $Prefix) { $Prefix = Join-Path $env:LOCALAPPDATA 'Chaos' }
if (-not $ModelsDir) { $ModelsDir = Join-Path $env:USERPROFILE '.chaos\models' }
$BinDir = Join-Path $Prefix 'bin'
# What the last run installed, so an upgrade can remove binaries this version
# no longer ships instead of leaving a stale one on PATH shadowing nothing.
$Manifest = Join-Path $Prefix 'installed-files.txt'
# Copied in beside the binaries when the archive carries them, and removed again
# on uninstall so the prefix can be cleaned up rather than left as a stub.
$DOCS = @('README.md', 'STATUS.md', 'CHANGELOG.md', 'LICENSE', 'NOTICE')

function Write-Step($msg) { Write-Host "  $msg" }

# -- PATH, per user, without duplicating it ----------------------------------
#
# Read from the registry rather than $env:PATH: the process variable is the
# merge of machine and user PATH plus whatever the parent shell exported, so
# testing against it would refuse to add an entry that is not actually
# persisted, and the install would work until the next reboot.

function Get-UserPath {
    $v = [Environment]::GetEnvironmentVariable('Path', 'User')
    if ($null -eq $v) { return '' }
    return $v
}

function Test-OnUserPath($dir) {
    $entries = (Get-UserPath) -split ';' | Where-Object { $_ -ne '' }
    foreach ($e in $entries) {
        if ($e.TrimEnd('\') -ieq $dir.TrimEnd('\')) { return $true }
    }
    return $false
}

function Add-ToUserPath($dir) {
    if (Test-OnUserPath $dir) {
        Write-Step "PATH already contains $dir"
        return
    }
    $current = Get-UserPath
    if ($current -eq '') { $new = $dir } else { $new = "$($current.TrimEnd(';'));$dir" }
    [Environment]::SetEnvironmentVariable('Path', $new, 'User')
    Write-Step "added $dir to your user PATH"
    Write-Host ''
    Write-Host '  Open a NEW terminal for that to take effect -- a running shell'
    Write-Host '  keeps the PATH it started with.'
}

function Remove-FromUserPath($dir) {
    $entries = (Get-UserPath) -split ';' | Where-Object { $_ -ne '' }
    $kept = $entries | Where-Object { $_.TrimEnd('\') -ine $dir.TrimEnd('\') }
    [Environment]::SetEnvironmentVariable('Path', ($kept -join ';'), 'User')
    Write-Step "removed $dir from your user PATH"
}

# -- uninstall ---------------------------------------------------------------

if ($Uninstall) {
    Write-Host 'Uninstalling Chaos.'
    if (Test-Path $BinDir) {
        Remove-Item -Recurse -Force $BinDir
        Write-Step "removed $BinDir"
    }
    if (Test-Path $Manifest) { Remove-Item -Force $Manifest }
    # The docs this script copied in. Named explicitly rather than wildcarded:
    # the prefix may be a directory the user chose and already had things in.
    foreach ($doc in $DOCS) {
        $p = Join-Path $Prefix $doc
        if (Test-Path $p) { Remove-Item -Force $p }
    }
    # Only if it is empty. A prefix the user pointed at something else of their
    # own must not be swept away because it happened to hold our bin directory.
    if ((Test-Path $Prefix) -and -not (Get-ChildItem -Force $Prefix)) {
        Remove-Item -Force $Prefix
        Write-Step "removed $Prefix"
    }
    if (-not $NoPath) { Remove-FromUserPath $BinDir }
    Write-Host ''
    Write-Host "Your models in $ModelsDir were left alone."
    exit 0
}

# -- find the binaries -------------------------------------------------------

$here = Split-Path -Parent $MyInvocation.MyCommand.Path
$binaries = @(Get-ChildItem -Path $here -Filter '*.exe' -File -ErrorAction SilentlyContinue |
    Where-Object { $_.Name -like 'chaos-*' -or $_.Name -eq 'gguf-info.exe' })

if ($binaries.Count -eq 0) {
    Write-Host 'No Chaos binaries found next to this script.'
    Write-Host ''
    Write-Host "  Looked in: $here"
    Write-Host '  Run this from the unpacked release archive, which contains'
    Write-Host '  install.ps1 and the chaos-*.exe files side by side.'
    exit 1
}

# -- a running binary is the one failure worth catching early ----------------
#
# Windows locks a running .exe, so Copy-Item fails halfway through and leaves a
# half-upgraded install. Checking first turns that into one sentence.

$locked = @()
foreach ($b in $binaries) {
    $dest = Join-Path $BinDir $b.Name
    if (-not (Test-Path $dest)) { continue }
    try {
        $fs = [IO.File]::Open($dest, 'Open', 'Write', 'None')
        $fs.Close()
    } catch {
        $locked += $b.Name
    }
}
if ($locked.Count -gt 0) {
    Write-Host 'Cannot upgrade: these are running right now.'
    Write-Host ''
    foreach ($n in $locked) { Write-Host "  $n" }
    Write-Host ''
    Write-Host '  Close them and run this again. Nothing has been changed.'
    exit 1
}

# -- install -----------------------------------------------------------------

$version = 'unknown'
$runner = $binaries | Where-Object { $_.Name -eq 'chaos-run.exe' } | Select-Object -First 1
if ($runner) {
    try { $version = (& $runner.FullName --version 2>&1 | Select-Object -First 1) } catch { }
}

$upgrading = Test-Path $BinDir
if ($upgrading) { Write-Host "Upgrading Chaos in $Prefix" } else { Write-Host "Installing Chaos into $Prefix" }
Write-Host ''

if (-not (Test-Path $BinDir)) { New-Item -ItemType Directory -Force -Path $BinDir | Out-Null }

# Remove what the previous version installed and this one does not, so an
# upgrade cannot leave an old binary on PATH. Only names from our own manifest
# are considered -- anything else the user put in this directory is theirs.
if ($upgrading -and (Test-Path $Manifest)) {
    $previous = Get-Content $Manifest | Where-Object { $_ -ne '' }
    $incoming = $binaries | ForEach-Object { $_.Name }
    foreach ($old in $previous) {
        if ($incoming -notcontains $old) {
            $p = Join-Path $BinDir $old
            if (Test-Path $p) {
                Remove-Item -Force $p
                Write-Step "removed $old (not in this version)"
            }
        }
    }
}

foreach ($b in $binaries) {
    Copy-Item -Path $b.FullName -Destination (Join-Path $BinDir $b.Name) -Force
}
Write-Step "installed $($binaries.Count) binaries into $BinDir"

# Docs, when the archive carries them. Not fatal if it does not.
foreach ($doc in $DOCS) {
    $src = Join-Path $here $doc
    if (Test-Path $src) { Copy-Item -Path $src -Destination (Join-Path $Prefix $doc) -Force }
}

Set-Content -Path $Manifest -Value ($binaries | ForEach-Object { $_.Name }) -Encoding utf8

if (-not (Test-Path $ModelsDir)) {
    New-Item -ItemType Directory -Force -Path $ModelsDir | Out-Null
    Write-Step "created $ModelsDir"
} else {
    Write-Step "models directory already exists: $ModelsDir"
}

if (-not $NoPath) { Add-ToUserPath $BinDir }

# -- what to do next ---------------------------------------------------------

Write-Host ''
Write-Host "Installed $version"
Write-Host ''
Write-Host '  Put your .gguf files here:'
Write-Host "    $ModelsDir"
Write-Host ''
Write-Host '  Then, in a new terminal:'
Write-Host '    chaos-probe                        what this machine can run'
Write-Host "    chaos-run `"$ModelsDir\<model>.gguf`" `"hello`" -n 32 --auto"
Write-Host ''
Write-Host '  Chaos downloads nothing on its own. Models are yours to obtain,'
Write-Host '  under their own licences.'
