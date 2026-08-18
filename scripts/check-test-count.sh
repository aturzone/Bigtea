#!/usr/bin/env bash
# The test count lives in four documents and has gone stale three times:
# 566 -> 570, 570 -> 575, 575 -> 583. Every time it was caught by a person
# noticing, which is not a mechanism.
#
# Run the suite, count what actually ran, and fail if any document disagrees.
set -uo pipefail

log="${1:-}"
if [ -z "$log" ]; then
  log=$(mktemp)
  cargo test --release --workspace > "$log" 2>&1
fi

actual=$(awk '/^test result/ { gsub(/[^0-9 ]/, " "); split($0, a, " "); p += a[1] } END { print p+0 }' "$log")
if [ "$actual" -eq 0 ]; then
  echo "::error::no test results found in $log -- did the suite build?"
  exit 1
fi
echo "measured: $actual tests"

fail=0
check() { # file, extracted value
  if [ "$2" != "$actual" ]; then
    echo "::error::$1 says $2 tests, the suite runs $actual"
    fail=1
  else
    echo "ok  $1"
  fi
}

check STATUS.md  "$(grep -oE '\*\*[0-9]+ tests\*\*' STATUS.md | head -1 | grep -oE '[0-9]+')"
check README.md  "$(grep -oE 'tests-[0-9]+' README.md | grep -oE '[0-9]+')"
check CLAUDE.md  "$(grep -oE '# [0-9]+ tests' CLAUDE.md | grep -oE '[0-9]+')"

exit $fail
