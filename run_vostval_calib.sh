#!/bin/bash
# Calibrare VOST-val: ruleaza secvential baseline -> v3 -> v6 pe toate 70 secventele
# via calea OFICIALA vot evaluate (format garantat + _time.value), apoi vot analysis -> Q REAL.
# Resumabil (vot evaluate sare secventele deja facute). Un singur model extra incarcat la un moment dat.
set -u
PROJ=/home/alexx/Projects/VOTSTracker
WS=$PROJ/workspace_vost_val
source $PROJ/dam4sam/.venv/bin/activate
cd $WS

echo "=== [calib] START $(date) ==="

for TR in dam4sam_b dam4sam_v3 dam4sam_v6; do
    echo "=== [calib] vot evaluate $TR $(date) ==="
    vot evaluate --persist --workspace $WS $TR >> /tmp/calib_${TR}.log 2>&1
    echo "=== [calib] $TR exit=$? $(date) ==="
done

echo "=== [calib] vot analysis $(date) ==="
vot analysis --workspace $WS >> /tmp/calib_analysis.log 2>&1
echo "=== [calib] analysis exit=$? $(date) ==="

echo "=== [calib] DONE $(date) ==="
echo "ALL_DONE" > /tmp/calib_done.flag
