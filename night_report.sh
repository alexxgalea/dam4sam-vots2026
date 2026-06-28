#!/bin/bash
# Ruleaza autonom dupa ce pipeline2 (v3->v4) termina. Genereaza raportul de dimineata.
set -u
cd /home/alexx/Projects/VOTSTracker
PY=dam4sam/.venv/bin/python3
R=/tmp/night_report.txt

echo "[night] $(date +%H:%M) astept pipeline2 sa termine (v3+v4)..."
until grep -q "PIPELINE COMPLET" /tmp/pipeline2.log 2>/dev/null; do sleep 120; done
echo "[night] $(date +%H:%M) pipeline2 GATA. Generez raport."

{
echo "================ RAPORT NOAPTE $(date) ================"
echo "v3 seqs: $(ls workspace_hard/results/dam4sam_v3/baseline/ 2>/dev/null | wc -l)/22"
echo "v4 seqs: $(ls workspace_hard/results/dam4sam_v4/baseline/ 2>/dev/null | wc -l)/22"
echo ""
echo "########## ROBUSTETE (compare_robustness.py, perm GRADAT, dR<0 = mai bun ca base) ##########"
echo "----- v2 vs base -----"
$PY compare_robustness.py workspace2026/results/dam4sam_b/baseline workspace_hard/results/dam4sam_v2/baseline --nameA base --nameB v2 2>/dev/null | tail -30
echo "----- v3 vs base -----"
$PY compare_robustness.py workspace2026/results/dam4sam_b/baseline workspace_hard/results/dam4sam_v3/baseline --nameA base --nameB v3 2>/dev/null | tail -30
echo "----- v4 vs base -----"
$PY compare_robustness.py workspace2026/results/dam4sam_b/baseline workspace_hard/results/dam4sam_v4/baseline --nameA base --nameB v4 2>/dev/null | tail -30
echo "----- v4 vs v3 (ajuta automatul de absenta?) -----"
$PY compare_robustness.py workspace_hard/results/dam4sam_v3/baseline workspace_hard/results/dam4sam_v4/baseline --nameA v3 --nameB v4 2>/dev/null | tail -30
echo ""
echo "########## CALITATE PREZENTA / ANTI-HALUCINATIE (validate_appearance.py, PQ mai mare = mai bine) ##########"
$PY validate_appearance.py --dirs base:workspace2026/results/dam4sam_b/baseline v3:workspace_hard/results/dam4sam_v3/baseline v4:workspace_hard/results/dam4sam_v4/baseline 2>/dev/null
echo ""
echo "########## VERIFICARE TINTA: rabbit & seagull (re-detectie dupa fix MEM_WINDOW=300) ##########"
$PY - <<'PYEOF' 2>/dev/null
import numpy as np
from pathlib import Path
from vot.region.io import read_trajectory
def emp(d,s):
    b=list(Path(d,s).glob("*.bin"))
    if not b: return None
    vals=[]
    for bb in b:
        t=read_trajectory(str(bb)); a=[]
        for r in t:
            try: m=r.mask if hasattr(r,'mask') else None; a.append(int(m.sum()) if m is not None else 0)
            except: a.append(0)
        vals.append(float((np.array(a)==0).mean()))
    return np.mean(vals)
for s in ["rabbit","seagull","motorcycle","fish2","colorfish"]:
    eb=emp("workspace2026/results/dam4sam_b/baseline",s)
    e3=emp("workspace_hard/results/dam4sam_v3/baseline",s)
    e4=emp("workspace_hard/results/dam4sam_v4/baseline",s)
    print(f"  {s:<14} empty: base={eb:.3f}  v3={e3:.3f}  v4={e4:.3f}")
PYEOF
echo "================ SFARSIT RAPORT ================"
} > "$R" 2>&1

echo "[night] $(date +%H:%M) RAPORT SCRIS in $R"
