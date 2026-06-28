#!/bin/bash
# Pilot automat noapte: asteapta v6 remaining (14 seq), compara cu v3 pe 22,
# decide automat GO/NOGO pentru 170 secvente, scrie raport complet.
set -u
cd /home/alexx/Projects/VOTSTracker
PY=dam4sam/.venv/bin/python3
R=/tmp/morning_v6_report.txt
ALL_SEQS_DIR=workspace2026/sequences
THRESHOLD_VS_V3="-0.01"   # v6 trebuie sa fie cu cel putin 0.01 mai bun decat v3 per seq

echo "[overnight] $(date) pornit" > "$R"

# ── 1. Asteapta terminarea celor 14 secvente ─────────────────────────────────
REMAINING="tank-16 bicycle-7 motorcycle-1 person-12 motorcycle racing-16 helicopter deer diabolo colorfish bus-2 crab-6 carchase birds1"
echo "[overnight] Astept terminarea v6 pe 14 secvente ramase..." >> "$R"

while true; do
    all_done=true
    for s in $REMAINING; do
        # numara obiectele necesare (din baseline)
        need=$(ls workspace2026/results/dam4sam_b/baseline/$s/*.bin 2>/dev/null | grep -v time | wc -l)
        have=$(ls workspace_hard/results/dam4sam_v6/baseline/$s/*.bin 2>/dev/null | wc -l)
        if [ "$have" -lt "$need" ] && [ "$need" -gt 0 ]; then
            all_done=false
            break
        fi
    done
    if $all_done; then break; fi
    sleep 120
done

echo "[overnight] $(date) v6 remaining GATA." >> "$R"

# ── 2. Compara v6 vs v3 pe toate 22 ──────────────────────────────────────────
echo "" >> "$R"
echo "========== V6 vs V3 — TOATE 22 SECVENTE ==========" >> "$R"
$PY compare_robustness.py \
    workspace_hard/results/dam4sam_v3/baseline \
    workspace_hard/results/dam4sam_v6/baseline \
    --nameA v3 --nameB v6 2>/dev/null >> "$R"

echo "" >> "$R"
echo "========== V6 vs BASELINE — TOATE 22 SECVENTE ==========" >> "$R"
$PY compare_robustness.py \
    workspace2026/results/dam4sam_b/baseline \
    workspace_hard/results/dam4sam_v6/baseline \
    --nameA base --nameB v6 2>/dev/null >> "$R"

# ── 3. Decizie GO/NOGO pentru 170 secvente ───────────────────────────────────
MEAN_DR=$($PY - <<'PYEOF' 2>/dev/null
import numpy as np, sys
from pathlib import Path
from vot.region.io import read_trajectory

def r_seq(d, s):
    bins = list(Path(d, s).glob("*.bin"))
    if not bins: return None
    vals = []
    for b in bins:
        t = read_trajectory(str(b)); a = []
        for r in t:
            try: m = r.mask if hasattr(r,'mask') else None; a.append(int(m.sum()) if m is not None else 0)
            except: a.append(0)
        a = np.array(a, float); n = len(a)
        if n == 0: continue
        vals.append(float((a==0).mean()) + float((a[int(n*0.9):]==0).mean()))
    return float(np.mean(vals)) if vals else None

HARD = ["uav1","tank-9","fish2","tank-16","bicycle-7","bull","motorcycle-1",
        "person-12","wubble_bubble","seagull","rabbit","motorcycle","motorcycle-18",
        "racing-16","helicopter","deer","diabolo","colorfish","bus-2","crab-6","carchase","birds1"]
V3  = "workspace_hard/results/dam4sam_v3/baseline"
V6  = "workspace_hard/results/dam4sam_v6/baseline"

drs = []
for s in HARD:
    r3 = r_seq(V3, s); r6 = r_seq(V6, s)
    if r3 is not None and r6 is not None:
        drs.append(r6 - r3)
print(f"{np.mean(drs):.4f}")
PYEOF
)

echo "" >> "$R"
echo "MEDIE dR (v6 vs v3, toate 22): $MEAN_DR" >> "$R"

# GO daca v6 e mai bun decat v3 cu cel putin 0.01 per secventa
GO=$(python3 -c "print('GO' if float('$MEAN_DR') < float('$THRESHOLD_VS_V3') else 'NOGO')" 2>/dev/null)
echo "DECIZIE: $GO (prag: v6 trebuie sa aiba dR_mean < $THRESHOLD_VS_V3 vs v3)" >> "$R"
echo "" >> "$R"

if [ "$GO" = "GO" ]; then
    echo "========== GO: PORNESC V6 PE TOATE 170 SECVENTE ==========" >> "$R"
    echo "[overnight] $(date) GO -> pornesc v6 --all-170" >> "$R"
    nohup $PY run_eval_170.py --tracker v6 --out v6_170 \
        > /tmp/run_v6_170.log 2>&1 &
    echo "PID run_v6_170: $!" >> "$R"
    echo "Log: /tmp/run_v6_170.log" >> "$R"
else
    echo "========== NOGO: V6 NU BATE V3 PE HARD-22 ==========" >> "$R"
    echo "Cel mai bun candidat pentru submisie ramane V3." >> "$R"
    echo "De analizat dimineata: care secvente au ramas pierderi in v6?" >> "$R"
fi

echo "" >> "$R"
echo "[overnight] $(date) GATA. Raport complet in $R" >> "$R"
