#!/usr/bin/env bash
# Sealed local overnight: prefetch → determinism → local baseline arms.
# Detach with setsid/nohup; do not read raw.jsonl until analysis stage.
set -uo pipefail
cd /home/gautamtalksdev/projects/jevbench
export PYTHONUNBUFFERED=1
export HF_HUB_DISABLE_PROGRESS_BARS=1

LOG=logs/local_overnight.log
STATUS=runs/SEALED_LOCAL_STATUS.txt
mkdir -p logs runs
exec >>"$LOG" 2>&1

_status() { echo "$1" >"$STATUS"; echo "=== $(date -u -Iseconds) $1 ==="; }

_status "overnight PID=$$ start"

_status "prefetch models"
set +e
.venv/bin/python - <<'PY'
from huggingface_hub import snapshot_download
for mid in [
    "Qwen/Qwen2.5-1.5B-Instruct",
    "knowledgator/gliclass-base-v1.0",
    "facebook/bart-large-mnli",
]:
    print(f"prefetch: {mid}", flush=True)
    print(f"  ok -> {snapshot_download(mid)}", flush=True)
PY
rc=$?
set -e
if [[ $rc -ne 0 ]]; then
  _status "prefetch FAILED exit=$rc"
  exit "$rc"
fi
_status "prefetch done"

# Determinism: 1 thread for CPU float16 bit-stability; pass_idx fixed in script.
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 TORCH_NUM_THREADS=1
_status "determinism start (threads=1)"
set +e
.venv/bin/python scripts/run_local_determinism.py --n-items 50 --repeats 3
rc=$?
set -e
_status "determinism end exit=$rc"
if [[ $rc -eq 137 ]] || [[ $rc -eq 9 ]]; then
  echo "HINT: exit 137/9 usually means OOM-killer (SIGKILL)."
  exit "$rc"
fi
if [[ $rc -ne 0 ]]; then
  echo "determinism non-zero; continuing to local arms (FAIL is recorded in results/)."
fi

# Full arms: 2 threads — modest speedup without the RSS thrash of 4.
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 TORCH_NUM_THREADS=2
_status "local baselines start (threads=2)"
set +e
.venv/bin/python -m jevbench run experiments/exp1_local_baselines.yaml --confirm --concurrency 1
rc=$?
set -e
_status "local baselines end exit=$rc"
_status "overnight complete"
exit "$rc"
