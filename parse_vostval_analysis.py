"""
Parseaza ultimul raport vot analysis dintr-un workspace si afiseaza, per tracker:
  - Quality (AverageAccuracy)  -- metrica principala VOTS
  - AR: accuracy, robustness
  - Auxiliary (NRE / DRE / AD etc daca exista)
Foloseste pentru calibrare: compara Q real intre baseline/v3/v6.

Uz: python parse_vostval_analysis.py [workspace_dir]
"""
import sys, json, glob
from pathlib import Path

ws = sys.argv[1] if len(sys.argv) > 1 else "/home/alexx/Projects/VOTSTracker/workspace_vost_val"
files = sorted(glob.glob(f"{ws}/analysis/*.json"))
if not files:
    print(f"Niciun raport in {ws}/analysis/"); sys.exit(1)
d = json.load(open(files[-1]))
print(f"Raport: {files[-1]}\n")

# trackeri (ordinea conteaza pt indexare in results)
trackers = d.get("trackers", [])
def tname(t):
    if isinstance(t, dict):
        return t.get("identifier") or t.get("label") or t.get("name") or str(t)
    return str(t)
tnames = [tname(t) for t in trackers]
print("Trackeri:", tnames, "\n")

# results: dict pe experiment ('baseline')
for exp, blk in d.get("results", {}).items():
    analyses = blk.get("parameters", {}).get("analyses", [])
    res = blk.get("results", [])
    print(f"### Experiment: {exp}")
    for ai, an in enumerate(analyses):
        name = an.get("name", f"analysis{ai}")
        if ai >= len(res):
            continue
        per_tracker = res[ai]
        # Quality / AR / Auxiliary -> valori scalare per tracker
        if name in ("Quality", "AR", "Auxiliary"):
            print(f"  -- {name} ({an.get('type','')}) --")
            for ti, val in enumerate(per_tracker):
                nm = tnames[ti] if ti < len(tnames) else f"tracker{ti}"
                # val poate fi [scalar] sau [acc, rob] sau lista
                print(f"     {nm:14s}: {val}")
    print()
