#!/bin/bash
# Dupa ce pipeline2 (v3->v4) termina: testeaza v5 (memorie adaptiva) pe secventele
# decisive. Daca recupereaza fish2 SI tine rabbit reparat -> ruleaza v5 pe tot setul.
set -u
cd /home/alexx/Projects/VOTSTracker
PY=dam4sam/.venv/bin/python3
OUT=/tmp/night_v5.txt

echo "[v5] $(date +%H:%M) astept pipeline2 (v3+v4)..." > "$OUT"
until grep -q "PIPELINE COMPLET" /tmp/pipeline2.log 2>/dev/null; do sleep 120; done
echo "[v5] $(date +%H:%M) pipeline2 gata. Test decisiv v5 pe fish2/rabbit/seagull." >> "$OUT"

$PY run_eval.py --tracker v5 --out v5 --seqs fish2 rabbit seagull >> /tmp/run_v5_test.log 2>&1

# Decizie GO/NOGO
DECISION=$($PY - <<'PYEOF' 2>/dev/null
import numpy as np
from pathlib import Path
from vot.region.io import read_trajectory
def emp(d,s):
    b=list(Path(d,s).glob("*.bin"))
    if not b: return None
    v=[]
    for bb in b:
        t=read_trajectory(str(bb)); a=[]
        for r in t:
            try: m=r.mask if hasattr(r,'mask') else None; a.append(int(m.sum()) if m is not None else 0)
            except: a.append(0)
        v.append(float((np.array(a)==0).mean()))
    return np.mean(v)
B="workspace2026/results/dam4sam_b/baseline"
V5="workspace_hard/results/dam4sam_v5/baseline"
V3="workspace_hard/results/dam4sam_v3/baseline"
res={}
for s in ["fish2","rabbit","seagull"]:
    res[s]=(emp(B,s),emp(V3,s),emp(V5,s))
print("SUBSET empty: secventa base / v3 / v5")
for s,(b,v3,v5) in res.items():
    print(f"  {s:<10} {b:.3f} / {v3:.3f} / {v5:.3f}")
# criteriu: fish2 v5 recupereaza (< 0.45, vs v3 ~0.83) SI rabbit v5 nu regreseaza (< 0.65)
f=res["fish2"][2]; r=res["rabbit"][2]
go = (f is not None and r is not None and f < 0.45 and r < 0.65)
print("GO" if go else "NOGO")
PYEOF
)
echo "$DECISION" >> "$OUT"

if echo "$DECISION" | grep -q "^GO$"; then
  echo "[v5] $(date +%H:%M) GO -> rulez v5 --all" >> "$OUT"
  $PY run_eval.py --tracker v5 --out v5 --all >> /tmp/run_v5_full.log 2>&1
  echo "[v5] $(date +%H:%M) v5 --all GATA: $(ls workspace_hard/results/dam4sam_v5/baseline/ 2>/dev/null | wc -l) seqs" >> "$OUT"
  echo "----- v5 vs base (graded) -----" >> "$OUT"
  $PY compare_robustness.py workspace2026/results/dam4sam_b/baseline workspace_hard/results/dam4sam_v5/baseline --nameA base --nameB v5 2>/dev/null | tail -28 >> "$OUT"
else
  echo "[v5] $(date +%H:%M) NOGO - memoria adaptiva nu recupereaza fish2 sau strica rabbit. Nu rulez v5 full; de tunat MEM_WINDOW_SHORT dimineata." >> "$OUT"
fi
echo "[v5] $(date +%H:%M) GATA." >> "$OUT"
