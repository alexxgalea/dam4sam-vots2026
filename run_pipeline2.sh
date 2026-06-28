#!/bin/bash
# Pipeline curat dupa fix-ul MEM_WINDOW=300: v3 --all -> v4 --all.
set -u
cd /home/alexx/Projects/VOTSTracker
PY=dam4sam/.venv/bin/python3

echo "[pipe2] $(date +%H:%M) RULEZ v3 --all (MEM_WINDOW=300)..."
$PY run_eval.py --tracker v3 --out v3 --all > /tmp/run_v3b.log 2>&1
echo "[pipe2] $(date +%H:%M) v3 GATA: $(ls workspace_hard/results/dam4sam_v3/baseline/ 2>/dev/null | wc -l) seqs"

echo "[pipe2] $(date +%H:%M) RULEZ v4 --all (MEM_WINDOW=300 + absence)..."
$PY run_eval.py --tracker v4 --out v4 --all > /tmp/run_v4b.log 2>&1
echo "[pipe2] $(date +%H:%M) v4 GATA: $(ls workspace_hard/results/dam4sam_v4/baseline/ 2>/dev/null | wc -l) seqs"

echo "[pipe2] $(date +%H:%M) PIPELINE COMPLET."
