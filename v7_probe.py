"""
Proba de discriminabilitate a semnalelor de identitate V7 (PLAN_WIN_VOTS2026 §4.1).

Ruleaza V7 (PROBE=True, comportament identic V3g) pe secvente unde baseline-ul
dam4sam_b e cunoscut-bun si exista distractori (babychimp, bull, colorfish).
Traiectoria baseline = pseudo-GT: la fiecare cadru, candidatul cu IoU maxim fata
de masca baseline = candidatul "corect".

Intrebarea (gate-ul §4.1): pe cadrele cu distractor real (un candidat se potriveste
cu referinta, altul nu), semnalele de identitate aleg candidatul corect mai des
decat insusi IoU-ul SAM? Daca da -> D1/D2 au combustibil. Daca nu -> escaladare §4.6.

Run:    dam4sam/.venv/bin/python v7_probe.py --seqs babychimp,bull,colorfish
Analiza: dam4sam/.venv/bin/python v7_probe.py --analyze
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/home/alexx/Projects/VOTSTracker")
sys.path.insert(0, "/home/alexx/Projects/VOTSTracker/dam4sam")

OUT_DIR = Path("/home/alexx/Projects/VOTSTracker/v7_probe_out")
DEFAULT_SEQS = "babychimp,bull,colorfish"


def run(seqs, ref_tracker):
    import torch
    from PIL import Image
    from vot.region.io import read_trajectory
    from run_eval_170 import (make_shared, get_objects, get_frames,
                              SEQUENCES_DIR, RESULTS_ROOT, TRACKER_NAME)
    from utils.utils import determine_tracker
    from sam2.build_sam import build_sam2_video_predictor
    from dam4sam_tracker_v7 import DAM4SAMTrackerV7

    DAM4SAMTrackerV7.PROBE = True
    ckpt, cfg = determine_tracker(TRACKER_NAME)
    shared = build_sam2_video_predictor(cfg, ckpt, device="cuda:0")
    OUT_DIR.mkdir(exist_ok=True)

    for seq_name in seqs:
        seq_dir = SEQUENCES_DIR / seq_name
        objects = get_objects(seq_dir)
        frames = get_frames(seq_dir)
        img0 = Image.open(frames[0])
        w, h = img0.width, img0.height
        n_objects = len(objects)

        for obj_id, gt_path in objects:
            out_npz = OUT_DIR / f"{seq_name}_obj{obj_id}.npz"
            if out_npz.exists():
                print(f"[skip] {out_npz.name}")
                continue
            fn = (f"{seq_name}_001.bin" if n_objects == 1
                  else f"{seq_name}_obj{obj_id}_001.bin")
            ref_bin = RESULTS_ROOT / ref_tracker / "baseline" / seq_name / fn
            ref_traj = read_trajectory(str(ref_bin))

            init_region = read_trajectory(str(gt_path))[0]
            init_mask = init_region.rasterize((0, 0, w - 1, h - 1)).astype(np.uint8)

            tracker = make_shared(DAM4SAMTrackerV7, shared)
            tracker.initialize(img0, init_mask)

            print(f"[probe] {seq_name} obj{obj_id}: {len(frames)} cadre")
            for fi, fp in enumerate(frames[1:], start=1):
                try:
                    r = ref_traj[fi]
                    tracker._probe_ref_mask = (
                        r.rasterize((0, 0, w - 1, h - 1)).astype(np.uint8)
                        if hasattr(r, "rasterize") else None)
                except Exception:
                    tracker._probe_ref_mask = None
                tracker.track(Image.open(fp))
                torch.cuda.empty_cache()

            recs = tracker._probe_records
            keys = ["ious", "cos_ptr_p0", "cos_ptr_last", "cos_hiera_p0",
                    "cos_hiera_last", "iou_ref"]
            data = {k: np.array([r[k] for r in recs], dtype=np.float64) for k in keys}
            for k in ["frame", "obj_logit", "tie", "chosen", "sam_argmax"]:
                data[k] = np.array([r[k] for r in recs], dtype=np.float64)
            np.savez(out_npz, **data)
            print(f"  -> {out_npz.name} ({len(recs)} cadre)")


def analyze():
    files = sorted(OUT_DIR.glob("*.npz"))
    if not files:
        print("nimic in v7_probe_out/"); return
    signals = ["ious", "cos_ptr_p0", "cos_ptr_last", "cos_hiera_p0", "cos_hiera_last"]
    print(f"{'secventa':<22} {'cadre':>6} {'distr':>6} | " +
          " ".join(f"{s:>14}" for s in signals))
    total = {s: [0, 0] for s in signals}
    total_tie = {s: [0, 0] for s in signals}
    n_tie_all = 0
    for f in files:
        d = np.load(f)
        iou_ref = d["iou_ref"]
        valid = ~np.isnan(iou_ref).any(axis=1)
        # cadru cu distractor real: un candidat se potriveste clar cu referinta,
        # iar urmatorul e clar mai slab
        srt = np.sort(iou_ref, axis=1)[:, ::-1]
        distractor = valid & (srt[:, 0] > 0.3) & ((srt[:, 0] - srt[:, 1]) > 0.2)
        tie = d["tie"].astype(bool) & distractor
        n_tie_all += int(tie.sum())
        correct = np.argmax(np.where(np.isnan(iou_ref), -1, iou_ref), axis=1)
        row = f"{f.stem:<22} {len(iou_ref):>6} {int(distractor.sum()):>6} | "
        for s in signals:
            v = d[s]
            pick = np.argmax(np.where(np.isnan(v), -np.inf, v), axis=1)
            agree = (pick == correct)
            n, k = int(distractor.sum()), int((agree & distractor).sum())
            total[s][0] += k; total[s][1] += n
            total_tie[s][0] += int((agree & tie).sum()); total_tie[s][1] += int(tie.sum())
            row += f"{(k / n if n else float('nan')):>13.2%} "
        print(row)
    print("-" * 120)
    line = f"{'TOTAL distractor':<22} {'':>6} {total[signals[0]][1]:>6} | "
    for s in signals:
        k, n = total[s]
        line += f"{(k / n if n else float('nan')):>13.2%} "
    print(line)
    line = f"{'TOTAL doar tie-frames':<22} {'':>6} {n_tie_all:>6} | "
    for s in signals:
        k, n = total_tie[s]
        line += f"{(k / n if n else float('nan')):>13.2%} "
    print(line)
    print("\nGate §4.1: un semnal cos_* trebuie sa bata coloana 'ious' (SAM insusi) "
          "pe cadrele distractor, in special pe tie-frames.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seqs", default=DEFAULT_SEQS)
    ap.add_argument("--ref", default="dam4sam_b")
    ap.add_argument("--analyze", action="store_true")
    args = ap.parse_args()
    if args.analyze:
        analyze()
    else:
        run([s for s in args.seqs.split(",") if s], args.ref)
        analyze()
