"""
Runner standalone parametrizat pentru evaluare DAM4SAM (v2/v3 + ablatii).

Exemple:
  # v3 complet pe tot setul greu
  python run_eval.py --tracker v3 --out v3 --all

  # ablatie: doar motion (v2), pe subset decisiv
  python run_eval.py --tracker v2 --out v2_motion --motion 1 --postproc 0 \
      --seqs seagull fish2 tank-16 rabbit bull uav1 motorcycle-18

  # ablatie: doar postproc (v2)
  python run_eval.py --tracker v2 --out v2_postproc --motion 0 --postproc 1 --seqs ...

Output: workspace_hard/results/dam4sam_<out>/baseline/<seq>/<seq>[_obj<N>]_001.bin
"""
import sys
import os
import argparse
sys.path.insert(0, "/home/alexx/Projects/VOTSTracker")
sys.path.insert(0, "/home/alexx/Projects/VOTSTracker/dam4sam")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import re
import random
import numpy as np
import torch
from pathlib import Path
from PIL import Image

from vot.region.io import read_trajectory, write_trajectory
from vot.region.shapes import Mask

from utils.utils import determine_tracker
from sam2.build_sam import build_sam2_video_predictor

SEQUENCES_DIR  = Path("/home/alexx/Projects/VOTSTracker/workspace2026/sequences")
BASELINE_DIR   = Path("/home/alexx/Projects/VOTSTracker/workspace2026/results/dam4sam_b/baseline")
RESULTS_ROOT   = Path("/home/alexx/Projects/VOTSTracker/workspace_hard/results")
TRACKER_NAME  = "sam21pp-B"

HARD_SEQS = [
    "uav1", "tank-9", "fish2", "tank-16", "bicycle-7", "bull", "motorcycle-1",
    "person-12", "wubble_bubble", "seagull", "rabbit", "motorcycle", "motorcycle-18",
    "racing-16", "helicopter", "deer", "diabolo", "colorfish", "bus-2", "crab-6",
    "carchase", "birds1",
]

seed = 0
random.seed(seed); os.environ["PYTHONHASHSEED"] = str(seed)
np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed(seed)


def quick_r(result_dir, seq_name):
    """R = empty_rate + perm_graded pentru o singura secventa. None daca lipsesc rezultatele."""
    bins = list(Path(result_dir, seq_name).glob("*.bin"))
    if not bins:
        return None
    vals = []
    for b in bins:
        t = read_trajectory(str(b))
        a = []
        for r in t:
            try:
                m = r.mask if hasattr(r, "mask") else None
                a.append(int(m.sum()) if m is not None else 0)
            except Exception:
                a.append(0)
        a = np.array(a, dtype=float)
        n = len(a)
        if n == 0:
            continue
        empty = float((a == 0).mean())
        tail = a[int(n * 0.9):]
        perm = float((tail == 0).mean())
        vals.append(empty + perm)
    return float(np.mean(vals)) if vals else None


def get_objects(seq_dir):
    gt_files = sorted(seq_dir.glob("groundtruth_obj*.txt"))
    if gt_files:
        objs = []
        for f in gt_files:
            mm = re.search(r"groundtruth_obj(\d+)\.txt$", f.name)
            if mm:
                objs.append((int(mm.group(1)), f))
        return objs
    gt = seq_dir / "groundtruth.txt"
    return [(1, gt)] if gt.exists() else []


def get_frames(seq_dir):
    return sorted((seq_dir / "color").glob("*.jpg")) or sorted((seq_dir / "color").glob("*.png"))


def make_shared(TrackerCls, shared_predictor):
    class _Shared(TrackerCls):
        def __init__(self, predictor):
            self.checkpoint, self.model_cfg = determine_tracker(TRACKER_NAME)
            self.input_image_size = 1024
            self.img_mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
            self.img_std  = torch.tensor([0.229, 0.224, 0.225])[:, None, None]
            self.predictor = predictor
            self.tracking_times = []
            self.kf = None
        def init_state_tw(self):
            return TrackerCls.init_state_tw(self)
    return _Shared(shared_predictor)


@torch.inference_mode()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tracker", choices=["v2", "v3", "v3b", "v3g", "v4", "v5", "v6"], default="v3")
    ap.add_argument("--vel-ratio", type=float, default=None, help="override VEL_RATIO_THR (v3g)")
    ap.add_argument("--absence", type=int, default=None, help="override USE_ABSENCE (v4)")
    ap.add_argument("--appearance", type=int, default=None, help="override USE_APPEARANCE (v4)")
    ap.add_argument("--out", required=True, help="subdir rezultat: dam4sam_<out>")
    ap.add_argument("--motion", type=int, default=None, help="override USE_MOTION (0/1)")
    ap.add_argument("--postproc", type=int, default=None, help="override USE_POSTPROC (0/1)")
    ap.add_argument("--seqs", nargs="*", default=None)
    ap.add_argument("--all", action="store_true")
    # Opreste daca dR cumulativ depaseste pragul dupa minim 5 secvente comparate.
    # Principiu greedy: nu risipi ore GPU pe o directie dovedita gresita.
    ap.add_argument("--max-regression", type=float, default=None,
                    help="Stop daca dR cumulativ vs baseline > acest prag (ex: 0.05)")
    ap.add_argument("--mem-short", type=int, default=None, help="override MEM_WINDOW_SHORT (v6)")
    ap.add_argument("--mem-long",  type=int, default=None, help="override MEM_WINDOW_LONG (v6)")
    args = ap.parse_args()

    if args.tracker == "v6":
        from dam4sam_tracker_v6 import DAM4SAMTrackerV6 as TC
    elif args.tracker == "v3b":
        from dam4sam_tracker_v3b import DAM4SAMTrackerV3b as TC
    elif args.tracker == "v3g":
        from dam4sam_tracker_v3g import DAM4SAMTrackerV3g as TC
    elif args.tracker == "v5":
        from dam4sam_tracker_v5 import DAM4SAMTrackerV5 as TC
    elif args.tracker == "v4":
        from dam4sam_tracker_v4 import DAM4SAMTrackerV4 as TC
    elif args.tracker == "v3":
        from dam4sam_tracker_v3 import DAM4SAMTrackerV3 as TC
    else:
        from dam4sam_tracker_v2 import DAM4SAMTrackerV2 as TC

    if args.motion is not None:
        TC.USE_MOTION = bool(args.motion)
    if args.postproc is not None:
        TC.USE_POSTPROC = bool(args.postproc)
    if args.absence is not None and hasattr(TC, "USE_ABSENCE"):
        TC.USE_ABSENCE = bool(args.absence)
    if args.appearance is not None and hasattr(TC, "USE_APPEARANCE"):
        TC.USE_APPEARANCE = bool(args.appearance)
    if args.mem_short is not None and hasattr(TC, "MEM_WINDOW_SHORT"):
        TC.MEM_WINDOW_SHORT = args.mem_short
    if args.mem_long is not None and hasattr(TC, "MEM_WINDOW_LONG"):
        TC.MEM_WINDOW_LONG = args.mem_long
    if args.vel_ratio is not None and hasattr(TC, "VEL_RATIO_THR"):
        TC.VEL_RATIO_THR = args.vel_ratio

    seqs = HARD_SEQS if (args.all or not args.seqs) else args.seqs
    out_dir = RESULTS_ROOT / f"dam4sam_{args.out}" / "baseline"

    print(f"[run_eval] tracker={args.tracker} out=dam4sam_{args.out} "
          f"USE_MOTION={TC.USE_MOTION} USE_POSTPROC={TC.USE_POSTPROC}")
    print(f"[run_eval] {len(seqs)} secvente")
    if args.max_regression is not None:
        print(f"[run_eval] GREEDY MODE: stop daca dR_cumul > {args.max_regression:+.3f} dupa min 5 seq")
    ckpt, cfg = determine_tracker(TRACKER_NAME)
    shared = build_sam2_video_predictor(cfg, ckpt, device="cuda:0")
    print(f"[run_eval] model incarcat, VRAM {torch.cuda.memory_allocated()/1e9:.2f} GB\n")

    cumul_dr = 0.0
    n_compared = 0
    for seq_name in seqs:
        seq_dir = SEQUENCES_DIR / seq_name
        if not seq_dir.exists():
            print(f"[SKIP] {seq_name}: lipseste"); continue
        objects = get_objects(seq_dir)
        frames = get_frames(seq_dir)
        if not objects or not frames:
            print(f"[SKIP] {seq_name}: fara obiecte/cadre"); continue
        n_objects = len(objects)

        def rpath(oid):
            fn = f"{seq_name}_001.bin" if n_objects == 1 else f"{seq_name}_obj{oid}_001.bin"
            return out_dir / seq_name / fn

        if all(rpath(oid).exists() for oid, _ in objects):
            print(f"[SKIP] {seq_name}: deja gata"); continue

        print(f"[SEQ] {seq_name} — {len(frames)} cadre, {n_objects} obiect(e)")
        img0 = Image.open(frames[0]); img_w, img_h = img0.width, img0.height

        for obj_id, gt_path in objects:
            op = rpath(obj_id)
            if op.exists():
                print(f"  obj{obj_id}: deja existent"); continue
            try:
                init_region = read_trajectory(str(gt_path))[0]
                init_mask = init_region.rasterize((0, 0, img_w - 1, img_h - 1)).astype(np.uint8)
            except Exception as e:
                print(f"  obj{obj_id}: eroare GT: {e}"); continue

            tracker = make_shared(TC, shared)
            try:
                tracker.initialize(img0, init_mask)
            except Exception as e:
                print(f"  obj{obj_id}: eroare initialize: {e}"); continue

            pred_masks = [Mask(init_mask)]
            for fi, fp in enumerate(frames[1:], start=1):
                try:
                    out = tracker.track(Image.open(fp))
                    pred_masks.append(Mask(out["pred_mask"]))
                except Exception as e:
                    print(f"  obj{obj_id} frame {fi}: eroare track: {e}")
                    pred_masks.append(Mask(np.zeros((img_h, img_w), dtype=np.uint8)))
                torch.cuda.empty_cache()

            op.parent.mkdir(parents=True, exist_ok=True)
            write_trajectory(str(op), pred_masks)
            print(f"  obj{obj_id}: {len(pred_masks)} cadre -> {op.name}")

        torch.cuda.empty_cache()
        print(f"  VRAM: {torch.cuda.memory_allocated()/1e9:.2f} GB")

        # ── Monitorizare greedy: dR vs baseline dupa fiecare secventa ────────
        r_base = quick_r(BASELINE_DIR, seq_name)
        r_new  = quick_r(out_dir, seq_name)
        if r_base is not None and r_new is not None:
            dr = r_new - r_base
            cumul_dr += dr
            n_compared += 1
            flag = "WIN" if dr < -0.01 else ("LOSS" if dr > 0.01 else "TIE")
            print(f"  [dR] {seq_name}: base={r_base:.3f} nou={r_new:.3f} "
                  f"dR={dr:+.3f} [{flag}] | cumul({n_compared}seq)={cumul_dr:+.3f}")
            if (args.max_regression is not None
                    and n_compared >= 5
                    and cumul_dr > args.max_regression):
                print(f"\n[STOP-GREEDY] dR_cumul={cumul_dr:+.3f} > limita {args.max_regression:.3f} "
                      f"dupa {n_compared} secvente. Oprim — diagnosticheaza inainte de a continua.")
                break
        print()

    print("[run_eval] Gata!")


if __name__ == "__main__":
    main()
