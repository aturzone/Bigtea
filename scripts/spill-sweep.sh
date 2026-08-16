#!/usr/bin/env bash
#
# Sweep the cost of a residency shortfall by removing RAM.
#
# WHAT IT ANSWERS
#   `chaos-run` warns that N GiB of always-read weights will be re-read on every
#   token, and says what that costs. The cost is a rate, and the rate has to come
#   from somewhere. This measures the only thing that settles it: how much slower
#   a token gets per additional GiB spilled.
#
#   The regression slope IS the per-GiB cost. Everything the runner prints at
#   load time is an estimate of this number, and this is the number to check it
#   against.
#
# METHOD
#   A balloon process takes RAM away; Chaos sizes its resident block from the
#   free RAM it sees at start, so a bigger balloon means a bigger spill. Passes
#   are INTERLEAVED -- every balloon size once, then again -- because three runs
#   at one point followed by three at the next returns a slope that is really a
#   clock.
#
#   Bash, not PowerShell: PowerShell 5.1 wraps a native executable's stderr in an
#   ErrorRecord even with plain `2> file`, and Chaos writes its status lines to
#   stderr, so a PowerShell driver dies before producing a row.
#
# USAGE
#   scripts/spill-sweep.sh <model.gguf> [passes] [balloon-MiB ...]
set -uo pipefail

MODEL="${1:?usage: spill-sweep.sh <model.gguf> [passes] [balloon-MiB ...]}"
PASSES="${2:-3}"
shift 2 2>/dev/null || shift 1
BALLOONS=("${@:-0 1536 3072 4608}")
# shellcheck disable=SC2206
BALLOONS=(${BALLOONS[@]})

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN="${CHAOS_RUN:-$HERE/../target/release/chaos-run.exe}"
OUT="${OUT_DIR:-${TMPDIR:-/tmp}/spill-sweep}"
mkdir -p "$OUT"
CSV="$OUT/spill-sweep.csv"

[ -x "$RUN" ] || { echo "no chaos-run at $RUN" >&2; exit 1; }

free_gib() {
    powershell.exe -NoProfile -Command \
        '[math]::Round((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory/1048576,2)' \
        2>/dev/null | tr -d '\r'
}

kill_balloons() {
    powershell.exe -NoProfile -Command \
        "Get-CimInstance Win32_Process -Filter \"Name='powershell.exe'\" |
         Where-Object { \$_.CommandLine -like '*ram-balloon*' } |
         ForEach-Object { Stop-Process -Id \$_.ProcessId -Force -ErrorAction SilentlyContinue }" \
        >/dev/null 2>&1
}
trap kill_balloons EXIT

echo "pass,balloon_mib,free_before,resident_gib,spill_gib,est_s,rate_gibs,prefill_s,tok_s,s_per_token" > "$CSV"
echo "sweep -> $CSV"

for pass in $(seq 1 "$PASSES"); do
    for mib in "${BALLOONS[@]}"; do
        kill_balloons
        sleep 2
        ready="$OUT/balloon-$pass-$mib.ok"
        rm -f "$ready"
        if [ "$mib" -gt 0 ]; then
            # -ExecutionPolicy Bypass: this box refuses unsigned .ps1 by default,
            # and the failure is a security error on stderr rather than a
            # non-zero exit, so without it the sweep runs happily against a
            # balloon that was never inflated.
            powershell.exe -NoProfile -ExecutionPolicy Bypass \
                -File "$(cygpath -w "$HERE/ram-balloon.ps1")" \
                -MiB "$mib" -Ready "$(cygpath -w "$ready")" >"$OUT/balloon-$pass-$mib.log" 2>&1 &
            # Wait for every page to be touched. Starting the run against a
            # half-inflated balloon measures a machine that no longer exists by
            # the time the tokens are generated.
            for _ in $(seq 1 120); do [ -f "$ready" ] && break; sleep 1; done
            [ -f "$ready" ] || { echo "balloon $mib MiB never inflated" >&2; continue; }
        fi

        before="$(free_gib)"
        log="$OUT/run-p$pass-b$mib.log"
        "$RUN" -m "$MODEL" -p "The capital of France is" -n 5 --temp 0 >"$log" 2>&1

        # `resident   loaded N tensors, X GiB of Y GiB budget in Zs (W GB/s); S GiB did not fit`
        resident=$(grep -oP 'loaded \d+ tensors, \K[0-9.]+' "$log" | head -1)
        spill=$(grep -oP '\K[0-9.]+(?= GiB did not fit)' "$log" | head -1)
        [ -z "$spill" ] && spill=0
        # `~As of each, at a measured B GiB/s on these tensors`
        est=$(grep -oP '~\K[0-9.]+(?=s of each)' "$log" | head -1)
        rate=$(grep -oP 'at a measured \K[0-9.]+(?= GiB/s)' "$log" | head -1)
        prefill=$(grep -oP 'prefill\s+\d+ tokens in \K[0-9.]+' "$log" | head -1)
        toks=$(grep -oP 'generate\s+\d+ tokens in [0-9.]+s \(\K[0-9.]+' "$log" | head -1)
        spt=$(grep -oP 'tok/s, \K[0-9.]+(?=s per token)' "$log" | head -1)

        echo "$pass,$mib,$before,${resident:-},${spill},${est:-},${rate:-},${prefill:-},${toks:-},${spt:-}" >> "$CSV"
        printf 'pass %s  balloon %5s MiB  free %5s  spill %6s GiB  %6s tok/s  %5s s/token\n' \
            "$pass" "$mib" "$before" "$spill" "${toks:-?}" "${spt:-?}"
        kill_balloons
        sleep 2
    done
done

echo
echo "=== least squares: s/token against spilled GiB ==="
awk -F, 'NR>1 && $5!="" && $10!="" {n++; x=$5; y=$10; sx+=x; sy+=y; sxx+=x*x; sxy+=x*y; syy+=y*y}
END {
  if (n<3) { print "not enough rows"; exit }
  m=(n*sxy-sx*sy)/(n*sxx-sx*sx); b=(sy-m*sx)/n;
  r=(n*sxy-sx*sy)/sqrt((n*sxx-sx*sx)*(n*syy-sy*sy));
  printf "  t = %.3f s/GiB * spill + %.3f s   R^2 = %.3f   (n=%d)\n", m, b, r*r, n;
  printf "  implied re-read rate: %.2f GiB/s\n", 1/m;
}' "$CSV"
