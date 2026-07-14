#!/bin/bash
# Runs v7d2 (Large backbone + D2 identity gating) on the given sequence split,
# with auto-resume across crashes. Intended to run ON the cloud instance.
# usage: bash scripts/cloud/cloud_run_ld2.sh <seqs_file>
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
SEQFILE="$1"
SEQS=$(cat "$SEQFILE")
OUT=dam4sam_l_d2
PY=dam4sam/.venv/bin/python3
for i in $(seq 1 50); do
  echo "===== ATTEMPT $i $(date '+%m-%d %H:%M:%S') ====="
  $PY scripts/run_eval_170.py --tracker v7d2 --backbone L --out $OUT --seqs "$SEQS" 2>&1 \
    | grep -vE "Warning|warn|Triggered|kernel not used|attention|_C.so|Skipping the post|fill_holes|forward_call|_methods|ret = ret"
  rc=$?
  rem=$($PY -c "import os; w=set(open('$SEQFILE').read().strip().split(',')); d=set(os.listdir('workspace2026/results/$OUT/baseline')) if os.path.isdir('workspace2026/results/$OUT/baseline') else set(); print(len(w-d))")
  echo "===== exit rc=$rc, remaining=$rem/$(echo $SEQS | tr ',' '\n' | wc -l) ====="
  [ "$rem" = "0" ] && echo "BOX DONE" && break
  sleep 10
done
echo "===== WRAPPER DONE ====="
