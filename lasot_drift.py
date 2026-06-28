"""
LaSOT-drift harness (PLAN_WIN_VOTS2026 §4.2) — masoara DRIFT cu GT real (box).

De ce: VOST-val e bucatarie lenta; clasele LaSOT (bird, monkey, zebra, motorcycle,
drone) = exact domeniul fast-motion + distractori al VOTS2026. Box GT e suficient
pentru drift: centroidul prezis in afara box-ului GT = obiect gresit.

IMPORTANT: secventele cu acelasi nume in workspace2026/sequences sunt EXCLUSE
automat (sunt in setul de test al competitiei — nu tunam pe test).

Eveniment de drift: centroid prezis in afara box GT >= PERSIST cadre consecutive,
pe cadre valide (GT prezent, fara full_occlusion / out_of_view).

Run:     dam4sam/.venv/bin/python lasot_drift.py --tracker v3g --out v3g
Compare: dam4sam/.venv/bin/python lasot_drift.py --compare baseline,v3g,v7
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/home/alexx/Projects/VOTSTracker")
sys.path.insert(0, "/home/alexx/Projects/VOTSTracker/dam4sam")

LASOT_DIR = Path("/home/alexx/Projects/VOTSTracker/lasot")
VOTS_SEQS = Path("/home/alexx/Projects/VOTSTracker/workspace2026/sequences")
OUT_ROOT = Path("/home/alexx/Projects/VOTSTracker/lasot_drift_out")
PERSIST = 5  # cadre consecutive in afara box-ului ca sa declaram drift

# split-ul de test LaSOT pt clasele descarcate (GT public pt toate, dar testul
# e curat + comparabil cu literatura)
TEST_SPLIT = [
    "bird-2", "bird-3", "bird-15", "bird-17",
    "monkey-3", "monkey-4", "monkey-9", "monkey-17",
    "zebra-10", "zebra-14", "zebra-16", "zebra-17",
    "motorcycle-1", "motorcycle-3", "motorcycle-9", "motorcycle-18",
    "drone-2", "drone-7", "drone-13", "drone-15",
]


def eligible_sequences():
    vots_names = {d.name for d in VOTS_SEQS.iterdir()} if VOTS_SEQS.exists() else set()
    seqs = []
    for name in TEST_SPLIT:
        cls = name[: name.rfind("-")]
        d = LASOT_DIR / cls / name
        if not d.is_dir():
            continue
        if name in vots_names:
            print(f"[exclus] {name}: e in cele 170 VOTS2026 (nu tunam pe test)")
            continue
        seqs.append((name, d))
    return seqs


def load_seq(d):
    gt = np.loadtxt(d / "groundtruth.txt", delimiter=",")          # (N,4) xywh
    frames = sorted((d / "img").glob("*.jpg"))
    invalid = np.zeros(len(gt), dtype=bool)
    for tag in ("full_occlusion.txt", "out_of_view.txt"):
        p = d / tag
        if p.exists():
            v = np.loadtxt(p, delimiter=",").flatten()[: len(gt)]
            invalid |= v.astype(bool)
    invalid |= (gt[:, 2] <= 0) | (gt[:, 3] <= 0)
    return gt, frames, invalid


def drift_metrics(centroids, gt, invalid):
    """centroids: (N,2) sau NaN unde masca goala."""
    n = len(gt)
    outside = np.zeros(n, dtype=bool)
    for i in range(1, n):
        if invalid[i] or np.isnan(centroids[i]).any():
            continue
        x, y = centroids[i]
        bx, by, bw, bh = gt[i]
        outside[i] = not (bx <= x <= bx + bw and by <= y <= by + bh)
    # evenimente: >= PERSIST cadre valide consecutive in afara
    events, first_drift, run = 0, -1, 0
    for i in range(1, n):
        if invalid[i]:
            continue
        if outside[i]:
            run += 1
            if run == PERSIST:
                events += 1
                if first_drift < 0:
                    first_drift = i - PERSIST + 1
        else:
            run = 0
    valid = ~invalid
    valid[0] = False
    return {
        "events": events,
        "first_drift": int(first_drift),
        "drift_frac": float(outside[valid].mean()) if valid.sum() else 0.0,
        "n_valid": int(valid.sum()),
    }


def get_tracker_class(name):
    if name == "baseline":
        from dam4sam_tracker import DAM4SAMTracker as TC
    elif name == "v3":
        from dam4sam_tracker_v3 import DAM4SAMTrackerV3 as TC
    elif name == "v3g":
        from dam4sam_tracker_v3g import DAM4SAMTrackerV3g as TC
    elif name == "v7":
        from dam4sam_tracker_v7 import DAM4SAMTrackerV7 as TC
    elif name == "v7d2":
        from dam4sam_tracker_v7 import DAM4SAMTrackerV7 as TC
        TC.USE_D2 = True
    else:
        raise SystemExit(f"tracker necunoscut: {name}")
    return TC


def run(args):
    import torch
    from PIL import Image
    from run_eval_170 import make_shared, TRACKER_NAME
    from utils.utils import determine_tracker
    from utils.mask_utils import mask2box
    from sam2.build_sam import build_sam2_video_predictor
    import run_eval_170 as _re

    if getattr(args, "backbone", "B") == "L":
        TRACKER_NAME = "sam21pp-L"
        _re.TRACKER_NAME = "sam21pp-L"
        print("[lasot_drift] backbone=L -> sam21pp-L")

    TC = get_tracker_class(args.tracker)
    for kv in args.set:
        k, v = kv.split("=", 1)
        setattr(TC, k, float(v))
        print(f"[override] {k}={v}")
    probe = args.tracker in ("v7", "v7d2")
    if probe:
        TC.PROBE = True   # log-only: comportament identic V3g, dar logheaza semnale
    out_dir = OUT_ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)

    seqs = eligible_sequences()
    if args.seqs:
        wanted = set(args.seqs.split(","))
        seqs = [(n, d) for n, d in seqs if n in wanted]
    print(f"[lasot_drift] tracker={args.tracker} -> {len(seqs)} secvente")

    ckpt, cfg = determine_tracker(TRACKER_NAME)
    shared = build_sam2_video_predictor(cfg, ckpt, device="cuda:0")

    for name, d in seqs:
        out_json = out_dir / f"{name}.json"
        if out_json.exists():
            print(f"[skip] {name}"); continue
        gt, frames, invalid = load_seq(d)
        n = min(len(gt), len(frames))
        print(f"[seq] {name}: {n} cadre")

        tracker = make_shared(TC, shared)
        img0 = Image.open(frames[0])
        tracker.initialize(img0, None, bbox=list(gt[0]))

        cents = np.full((n, 2), np.nan)
        areas = np.zeros(n)
        bx, by, bw, bh = gt[0]
        cents[0] = (bx + bw / 2, by + bh / 2)
        areas[0] = bw * bh
        for i in range(1, n):
            out = tracker.track(Image.open(frames[i]))
            bb = mask2box(out["pred_mask"])
            if bb is not None:
                cents[i] = (bb[0] + bb[2] / 2, bb[1] + bb[3] / 2)
                areas[i] = bb[2] * bb[3]
            torch.cuda.empty_cache()

        m = drift_metrics(cents, gt[:n], invalid[:n])
        m["seq"] = name
        out_json.write_text(json.dumps(m))
        # seriile brute — materia prima pt switch-detector (§4.3)
        extra = {}
        if probe and getattr(tracker, "_probe_records", None):
            recs = tracker._probe_records
            for k in ("frame", "ious", "cand_boxes", "tie", "chosen", "obj_logit",
                      "cos_ptr_p0", "cos_ptr_last", "cos_hiera_p0", "cos_hiera_last"):
                extra["probe_" + k] = np.array([r[k] for r in recs], dtype=np.float64)
        np.savez(out_dir / f"{name}.npz", centroids=cents, areas=areas,
                 gt=gt[:n], invalid=invalid[:n], **extra)
        print(f"  events={m['events']} first={m['first_drift']} "
              f"drift_frac={m['drift_frac']:.3f}")
        if getattr(tracker, "_d2_stats", None) and tracker.USE_D2:
            print(f"  d2: {tracker._d2_stats}")


def compare(names):
    rows = {}
    for t in names:
        d = OUT_ROOT / t
        for f in sorted(d.glob("*.json")):
            m = json.loads(f.read_text())
            rows.setdefault(m["seq"], {})[t] = m
    print(f"{'secventa':<16} | " + " | ".join(f"{t:>26}" for t in names))
    print(f"{'':<16} | " + " | ".join(f"{'ev / first / frac':>26}" for _ in names))
    agg = {t: [0, 0.0, 0] for t in names}
    for seq, by_t in sorted(rows.items()):
        row = f"{seq:<16} | "
        for t in names:
            m = by_t.get(t)
            if m:
                row += f"{m['events']:>6} {m['first_drift']:>7} {m['drift_frac']:>10.3f} | "
                agg[t][0] += m["events"]; agg[t][1] += m["drift_frac"]; agg[t][2] += 1
            else:
                row += f"{'—':>26} | "
        print(row)
    print("-" * (18 + 29 * len(names)))
    row = f"{'TOTAL':<16} | "
    for t in names:
        ev, fr, k = agg[t]
        row += f"{ev:>6} {'':>7} {fr / max(k, 1):>10.3f} | "
    print(row)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tracker", default="baseline")
    ap.add_argument("--out", default=None)
    ap.add_argument("--seqs", default=None)
    ap.add_argument("--backbone", choices=["B", "L"], default="B")
    ap.add_argument("--compare", default=None, help="lista out-uri, ex: baseline,v3g")
    ap.add_argument("--set", action="append", default=[],
                    help="override atribut clasa tracker, ex: --set TAU_DROP_MEM=-1")
    args = ap.parse_args()
    if args.compare:
        compare(args.compare.split(","))
    else:
        args.out = args.out or args.tracker
        run(args)
