<#
.SYNOPSIS
    Take N MiB of physical RAM away from the machine and hold it.

.DESCRIPTION
    The tok/s-versus-RAM frontier cannot be swept upward on a laptop that has
    all the memory it will ever have, but it can be swept *downward*: remove
    memory, and Chaos sizes its resident block from the free RAM it sees at
    start. That turns an unmeasurable axis into a measurable one.

    The allocation must be **touched**, not merely requested. .NET commits
    lazily, so an untouched byte[] is an imaginary balloon that moves free
    physical memory not at all -- every page is written here, one byte per 4 KiB
    page, which is what forces the commit.

    Writes a marker file when the balloon is fully inflated, because the sweep
    has to know the machine is actually short before it starts a run. Holds
    until killed.

.PARAMETER MiB
    How much to take.

.PARAMETER Ready
    Path to create once every page has been touched.

.EXAMPLE
    powershell -NoProfile -File scripts/ram-balloon.ps1 -MiB 2048 -Ready /tmp/b.ok
#>
param(
    [Parameter(Mandatory = $true)][int]$MiB,
    [string]$Ready = ""
)

$ErrorActionPreference = 'Stop'

# 512 MiB at a time: .NET caps a single object at 2 GiB, and smaller chunks are
# likelier to be satisfied on a fragmented heap than one huge contiguous one.
$chunkMiB = 512
$chunks = New-Object System.Collections.ArrayList
$remaining = $MiB
$sw = [System.Diagnostics.Stopwatch]::StartNew()

while ($remaining -gt 0) {
    $thisMiB = [Math]::Min($chunkMiB, $remaining)
    $bytes = $thisMiB * 1MB
    $buf = New-Object byte[] $bytes
    # One write per 4 KiB page. Fewer writes would leave pages uncommitted and
    # the balloon would be a no-op that looks like a result.
    for ($o = 0; $o -lt $bytes; $o += 4096) { $buf[$o] = 1 }
    [void]$chunks.Add($buf)
    $remaining -= $thisMiB
}

$sw.Stop()
$free = [math]::Round((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory / 1MB, 2)
Write-Host ("balloon {0} MiB touched in {1:N1}s; free now {2} GiB" -f $MiB, $sw.Elapsed.TotalSeconds, $free)

if ($Ready -ne "") { Set-Content -Path $Ready -Value $free -Encoding utf8 }

# Hold. The sweep kills this process when the run is done.
while ($true) { Start-Sleep -Seconds 60 }
