"""
Comparatie robustete intre doi trackeri, GT-free, metrica DEFENSABILA.

Spre deosebire de diagnose_results.py (care normaliza volatilitatea la arie si
penaliza nedrept urmarirea stransa a obiectelor mici), aici folosim doar semnale
aliniate cu robustetea VOTS:

  - empty_rate     : fractia de cadre fara masca (obiect pierdut)         [0..1]
  - perm_loss      : 1 daca ultimele 10 cadre sunt goale (pierdere definitiva)
  - reacq_fail     : pierderi lungi (>=10 cadre goale consecutive) din care
                     NU se mai recupereaza in <30 cadre

Scor robustete (mai mic = mai bun):
  R = empty_rate + perm_loss

Utilizare:
  python compare_robustness.py <dirA> <dirB> [--seqs s1 s2 ...]
  (implicit: dam4sam_b/baseline vs dam4sam_v2/baseline pe secventele din B)
"""
import sys
import argparse
import numpy as np
from pathlib import Path
from vot.region.io import read_trajectory

ROOT = Path("/home/alexx/Projects/VOTSTracker")
DEF_A = ROOT / "workspace2026/results/dam4sam_b/baseline"
DEF_B = ROOT / "workspace_hard/results/dam4sam_v2/baseline"


def areas(traj_path):
    traj = read_trajectory(str(traj_path))
    out = []
    for r in traj:
        try:
            m = r.mask if hasattr(r, "mask") else None
            out.append(int(m.sum()) if m is not None else 0)
        except Exception:
            out.append(0)
    return np.array(out, dtype=float)


def robustness(a):
    n = len(a)
    if n == 0:
        return dict(empty=1.0, perm=1.0, R=2.0)
    empty = float((a == 0).mean())
    # perm GRADAT: fractia goala pe ultimele 10% din secventa (nu binar pe ultimele 10
    # cadre). Un binar last-10 pedepsea nedrept un tracker excelent care pierde tinta
    # doar in finalul secventei (ex. seagull: empty 0.18 vs 0.53 dar +1.0 penalizare).
    tail = a[int(n * 0.9):]
    perm = float((tail == 0).mean())
    return dict(empty=empty, perm=perm, R=empty + perm)


def seq_R(seq_dir):
    """Media R peste toate obiectele (.bin) dintr-o secventa."""
    bins = sorted(seq_dir.glob("*.bin"))
    if not bins:
        return None
    rs = [robustness(areas(b)) for b in bins]
    return dict(
        empty=np.mean([r["empty"] for r in rs]),
        perm=np.mean([r["perm"] for r in rs]),
        R=np.mean([r["R"] for r in rs]),
        nobj=len(bins),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dirA", nargs="?", default=str(DEF_A))
    ap.add_argument("dirB", nargs="?", default=str(DEF_B))
    ap.add_argument("--seqs", nargs="*", default=None)
    ap.add_argument("--nameA", default="A")
    ap.add_argument("--nameB", default="B")
    args = ap.parse_args()

    dirA, dirB = Path(args.dirA), Path(args.dirB)
    if args.seqs:
        seqs = args.seqs
    else:
        seqs = sorted(d.name for d in dirB.iterdir() if d.is_dir())

    print(f"\n{args.nameA} = {dirA}")
    print(f"{args.nameB} = {dirB}\n")
    print(f"{'Secventa':<16}{'R_'+args.nameA:>8}{'R_'+args.nameB:>8}{'dR':>8}  {'verdict'}")
    print("-" * 56)

    dR = []
    win = lose = 0
    for s in seqs:
        ra = seq_R(dirA / s)
        rb = seq_R(dirB / s)
        if ra is None or rb is None:
            continue
        d = rb["R"] - ra["R"]
        dR.append(d)
        if d < -0.02:
            v = f"{args.nameB} ✓"; win += 1
        elif d > 0.02:
            v = f"{args.nameA} ✓"; lose += 1
        else:
            v = "~"
        print(f"{s:<16}{ra['R']:>8.3f}{rb['R']:>8.3f}{d:>+8.3f}  {v}")

    print("-" * 56)
    if dR:
        m = float(np.mean(dR))
        print(f"{'MEDIE dR':<16}{'':>8}{'':>8}{m:>+8.3f}")
        verdict = f"{args.nameB} MAI BUN" if m < -0.005 else (
            f"{args.nameA} mai bun" if m > 0.005 else "EGALITATE")
        print(f"\n{verdict}  (dR<0 inseamna {args.nameB} mai robust)")
        print(f"{args.nameB} castiga: {win}/{len(dR)} | {args.nameA} castiga: {lose}/{len(dR)} | "
              f"egal: {len(dR)-win-lose}/{len(dR)}")


if __name__ == "__main__":
    main()
