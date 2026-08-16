#!/bin/bash
# Capture one routing histogram per prompt, for the R0 hot-set analysis.
#
#   GGUF=/path/model-00001-of-00005.gguf ./capture.sh [outdir]
#
# `-n 1` is load-bearing: exactly one prefill and no generation. chaos-run
# regenerates statelessly, so each generated token re-runs prefill over the whole
# sequence and the histogram counts the same prompt again -- which inflates
# chi-square by the number of passes while leaving top-k coverage untouched.
# That is how v0.0.2 came to publish a 97.8% coverage next to a chi-square of
# 7805, two figures that no single-pass run produces together.
#
# Then: python analyse.py <outdir>/csv
set -eu

HERE=$(cd "$(dirname "$0")" && pwd)
OUT=${1:-"$HERE/captures"}
GGUF=${GGUF:?set GGUF to the first shard of the model}
BIN=${BIN:-"$HERE/../../target/release/chaos-run"}

mkdir -p "$OUT/csv" "$OUT/logs"

for f in "$HERE"/prompts/*.txt; do
  name=$(basename "$f" .txt)
  if [ -s "$OUT/csv/$name.csv" ]; then
    echo "SKIP $name (already captured)"
    continue
  fi
  echo "RUN  $name"
  start=$(date +%s)
  CHAOS_ROUTING=1 CHAOS_ROUTING_DUMP="$OUT/csv/$name.csv" \
    "$BIN" "$GGUF" "$(cat "$f")" -n 1 > "$OUT/logs/$name.log" 2>&1
  echo "DONE $name in $(( $(date +%s) - start ))s  $(grep -m1 '^prompt' "$OUT/logs/$name.log")"
done
echo "captures in $OUT/csv — now: python $HERE/analyse.py $OUT/csv"
