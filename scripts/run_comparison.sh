#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$project_dir"
mode="${1:-full}"
if (($#)); then shift; fi
case "$mode" in
  smoke)
    config="configs/comparison.yaml"
    budget="2m"
    extra=(--smoke)
    ;;
  pilot)
    config="configs/comparison_pilot.yaml"
    budget="30m"
    extra=(--checkpoints 62.5 125 250 --check-replicates 3)
    ;;
  full)
    config="configs/comparison.yaml"
    budget="115m"
    extra=()
    ;;
  *)
    echo "Usage: bash scripts/run_comparison.sh [smoke|pilot|full] [--resume]" >&2
    exit 2
    ;;
esac
for option in "$@"; do
  if [[ "$option" != "--resume" ]]; then
    echo "The fixed launcher accepts only --resume; use the Python module for custom experiments." >&2
    exit 2
  fi
done
if [[ ! -x .venv/bin/python ]]; then
  echo "Create .venv and install the project first; see docs/comparison.md." >&2
  exit 2
fi
command -v timeout >/dev/null
command -v flock >/dev/null
# Parallelize records, while each NumPy/SciPy worker uses one BLAS thread.
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
workers="${COMPARISON_WORKERS:-24}"
output="runs/comparison-${mode}"
mkdir -p runs
log_status() {
  printf '%s %s\n' "$(date --iso-8601=seconds)" "$*" | tee -a "${output}.log"
}
log_status "Launch: profile=${mode}, workers=${workers}, time_limit=${budget}"
# Keep logs outside the output directory, which must initially be empty.
if flock --nonblock "${output}.lock" \
    timeout --signal=TERM --kill-after=20s "$budget" \
    .venv/bin/python -u -m likelihood_filtering.comparison \
    --config "$config" --output "$output" --workers "$workers" \
    "${extra[@]}" "$@" 2>&1 | tee -a "${output}.log"; then
  log_status "Completed (exit 0). Results: ${output}/summary.csv"
else
  result=$?
  if [[ "$result" == 124 || "$result" == 137 ]]; then
    log_status "Time limit reached (exit ${result}). Saved records are resumable; this run is incomplete."
  else
    log_status "Comparison failed (exit ${result}); see ${output}.log."
  fi
  exit "$result"
fi
