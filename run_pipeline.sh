#!/bin/bash
# Pipeline secvential pe GPU (un singur GPU -> rulari una dupa alta):
#   1. asteapta v2 full (PID dat) sa termine
#   2. smoke test v4 pe colorfish (284 cadre, rapid)
#   3. v3 pe tot setul greu
#   4. v4 pe tot setul greu
# Loguri separate in /tmp. v2/v3/v4 sar peste secventele deja existente.
set -u
cd /home/alexx/Projects/VOTSTracker
PY=dam4sam/.venv/bin/python3
V2_PID="${1:-449283}"

echo "[pipe] $(date +%H:%M) astept v2 (PID $V2_PID)..."
while kill -0 "$V2_PID" 2>/dev/null; do sleep 30; done
echo "[pipe] $(date +%H:%M) v2 TERMINAT. v2 seqs: $(ls workspace_hard/results/dam4sam_v2/baseline/ | wc -l)"

echo "[pipe] $(date +%H:%M) SMOKE v4 pe colorfish..."
$PY run_eval.py --tracker v4 --out v4_smoke --seqs colorfish > /tmp/v4_smoke.log 2>&1
if ls workspace_hard/results/dam4sam_v4_smoke/baseline/colorfish/*.bin >/dev/null 2>&1; then
  echo "[pipe] $(date +%H:%M) SMOKE v4 OK"
else
  echo "[pipe] $(date +%H:%M) SMOKE v4 ESUAT - opresc pipeline (vezi /tmp/v4_smoke.log)"
  exit 1
fi

echo "[pipe] $(date +%H:%M) RULEZ v3 --all..."
$PY run_eval.py --tracker v3 --out v3 --all > /tmp/run_v3.log 2>&1
echo "[pipe] $(date +%H:%M) v3 GATA: $(ls workspace_hard/results/dam4sam_v3/baseline/ 2>/dev/null | wc -l) seqs"

echo "[pipe] $(date +%H:%M) RULEZ v4 --all..."
$PY run_eval.py --tracker v4 --out v4 --all > /tmp/run_v4.log 2>&1
echo "[pipe] $(date +%H:%M) v4 GATA: $(ls workspace_hard/results/dam4sam_v4/baseline/ 2>/dev/null | wc -l) seqs"

echo "[pipe] $(date +%H:%M) PIPELINE COMPLET."
