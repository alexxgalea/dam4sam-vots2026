"""
Validare GT-free pentru DISPARITIE/REAPARITIE prin consistenta de ASPECT.

Idee: in VOTS long-term, comportamentul corect inseamna (a) sa emiti masca DOAR
cand obiectul e prezent si (b) masca sa fie chiar obiectul, nu un distractor.
Fara GT, masuram aspectul: histograma HS a regiunii mascate vs template-ul de init.

Metrici per secventa (peste cadrele cu masca non-goala):
  - nonempty      : fractia de cadre cu masca
  - mean_sim      : corelatie medie aspect vs init (mai mare = masti mai de incredere)
  - halluc_rate   : fractia de masti cu sim < APP_THR (probabil distractor) -> MAI MIC e mai bine

Scor sintetic de calitate prezenta (mai mare = mai bine):
  PQ = mean_sim * (1 - halluc_rate)

Compara doi/trei trackeri:
  python validate_appearance.py --dirs dam4sam_b:workspace2026/results/dam4sam_b/baseline \
        v3:workspace_hard/results/dam4sam_v3/baseline \
        v4:workspace_hard/results/dam4sam_v4/baseline
"""
import argparse
import numpy as np
import cv2
from pathlib import Path
from vot.region.io import read_trajectory

ROOT = Path("/home/alexx/Projects/VOTSTracker")
SEQ_DIR = ROOT / "workspace2026/sequences"
APP_THR = 0.45
HS_BINS = [16, 16]


def hs_hist(rgb, mask):
    ys, xs = np.where(mask > 0)
    if len(xs) < 10:
        return None
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    h = cv2.calcHist([hsv], [0, 1], (mask > 0).astype(np.uint8), HS_BINS, [0, 180, 0, 256])
    cv2.normalize(h, h, 0, 1, cv2.NORM_MINMAX)
    return h.flatten().astype(np.float32)


def frames_of(seq):
    d = SEQ_DIR / seq / "color"
    return sorted(d.glob("*.jpg")) or sorted(d.glob("*.png"))


def mask_areas_and_hists(bin_path, frames, init_hist):
    """Returneaza (nonempty_frac, mean_sim, halluc_rate) pentru un .bin."""
    traj = read_trajectory(str(bin_path))
    n = min(len(traj), len(frames))
    sims = []
    nonempty = 0
    for i in range(n):
        r = traj[i]
        try:
            m = r.mask if hasattr(r, "mask") else None
        except Exception:
            m = None
        if m is None or int(m.sum()) < 10:
            continue
        nonempty += 1
        rgb = cv2.cvtColor(cv2.imread(str(frames[i])), cv2.COLOR_BGR2RGB)
        # masca poate avea offset/dim diferit; rasterizam la dim cadru
        H, W = rgb.shape[:2]
        mm = np.zeros((H, W), dtype=np.uint8)
        mh, mw = m.shape
        mm[:min(H, mh), :min(W, mw)] = m[:min(H, mh), :min(W, mw)]
        h = hs_hist(rgb, mm)
        if h is None or init_hist is None:
            continue
        s = max(0.0, cv2.compareHist(init_hist.reshape(-1, 1), h.reshape(-1, 1), cv2.HISTCMP_CORREL))
        sims.append(s)
    if not sims:
        return nonempty / max(n, 1), 0.0, 0.0
    sims = np.array(sims)
    return nonempty / max(n, 1), float(sims.mean()), float((sims < APP_THR).mean())


def init_hist_for(seq):
    sd = SEQ_DIR / seq
    gts = sorted(sd.glob("groundtruth_obj*.txt")) or [sd / "groundtruth.txt"]
    frames = frames_of(seq)
    if not frames or not gts[0].exists():
        return None, None
    rgb = cv2.cvtColor(cv2.imread(str(frames[0])), cv2.COLOR_BGR2RGB)
    H, W = rgb.shape[:2]
    reg = read_trajectory(str(gts[0]))[0]
    m = reg.rasterize((0, 0, W - 1, H - 1)).astype(np.uint8)
    return hs_hist(rgb, m), frames


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", nargs="+", required=True, help="name:path ...")
    ap.add_argument("--seqs", nargs="*", default=None)
    args = ap.parse_args()

    trackers = []
    for d in args.dirs:
        name, path = d.split(":", 1)
        trackers.append((name, Path(path) if Path(path).is_absolute() else ROOT / path))

    # secvente = cele prezente in TOATE dirurile
    if args.seqs:
        seqs = args.seqs
    else:
        sets = [set(p.name for p in tp.iterdir() if p.is_dir()) for _, tp in trackers]
        seqs = sorted(set.intersection(*sets))

    hdr = f"{'Secventa':<15}"
    for name, _ in trackers:
        hdr += f"{'PQ_'+name:>10}"
    print(hdr); print("-" * len(hdr))

    agg = {name: [] for name, _ in trackers}
    for seq in seqs:
        ih, frames = init_hist_for(seq)
        if ih is None:
            continue
        row = f"{seq:<15}"
        for name, tp in trackers:
            bins = sorted((tp / seq).glob("*.bin"))
            if not bins:
                row += f"{'-':>10}"; continue
            pqs = []
            for b in bins:
                ne, ms, hr = mask_areas_and_hists(b, frames, ih)
                pqs.append(ms * (1 - hr))
            pq = float(np.mean(pqs))
            agg[name].append(pq)
            row += f"{pq:>10.3f}"
        print(row)

    print("-" * len(hdr))
    foot = f"{'MEDIE PQ':<15}"
    for name, _ in trackers:
        foot += f"{np.mean(agg[name]):>10.3f}" if agg[name] else f"{'-':>10}"
    print(foot)
    print("\nPQ = mean_appearance_sim * (1 - halluc_rate). Mai MARE = mai bine.")


if __name__ == "__main__":
    main()
