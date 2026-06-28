"""
Runner pentru toate 170 de secvente VOTS2026. Identic cu run_eval.py dar:
  - SEQUENCES_DIR si RESULTS_ROOT pointeaza la workspace2026
  - Lista de secvente = tot ce exista in workspace2026/sequences/
  - Sare automat peste secventele deja calculate (resumabil)
  - Suporta --max-regression pentru early-stop greedy
"""
import sys, os, re, random
import numpy as np
import torch
from pathlib import Path
from PIL import Image
import argparse

sys.path.insert(0, "/home/alexx/Projects/VOTSTracker")
sys.path.insert(0, "/home/alexx/Projects/VOTSTracker/dam4sam")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

from vot.region.io import read_trajectory, write_trajectory
from vot.region.shapes import Mask
from utils.utils import determine_tracker
from sam2.build_sam import build_sam2_video_predictor

SEQUENCES_DIR = Path("/home/alexx/Projects/VOTSTracker/workspace2026/sequences")
RESULTS_ROOT  = Path("/home/alexx/Projects/VOTSTracker/workspace2026/results")
BASELINE_DIR  = Path("/home/alexx/Projects/VOTSTracker/workspace2026/results/dam4sam_b/baseline")
TRACKER_NAME  = "sam21pp-B"

seed = 0
random.seed(seed); os.environ["PYTHONHASHSEED"] = str(seed)
np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed(seed)


def get_all_seqs():
    seqs = sorted([d.name for d in SEQUENCES_DIR.iterdir() if d.is_dir()])
    # exclude directorul 'list.txt' daca exista ca fisier
    return [s for s in seqs if (SEQUENCES_DIR / s).is_dir()]


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


def quick_r(result_dir, seq_name):
    bins = list(Path(result_dir, seq_name).glob("*.bin"))
    if not bins:
        return None
    vals = []
    for b in bins:
        t = read_trajectory(str(b)); a = []
        for r in t:
            try:
                m = r.mask if hasattr(r, "mask") else None
                a.append(int(m.sum()) if m is not None else 0)
            except Exception:
                a.append(0)
        a = np.array(a, float); n = len(a)
        if n == 0: continue
        empty = float((a == 0).mean())
        perm  = float((a[int(n * 0.9):] == 0).mean())
        vals.append(empty + perm)
    return float(np.mean(vals)) if vals else None


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
    ap.add_argument("--tracker", choices=["baseline", "v2", "v3", "v3b", "v3g", "v4", "v5", "v6", "v7", "v7d2"], default="v3")
    ap.add_argument("--backbone", choices=["B", "L"], default="B",
                    help="B = SAM2.1-B+ (default), L = SAM2.1-Large (sam21pp-L)")
    ap.add_argument("--seqs", default=None,
                    help="lista de secvente separate prin virgula (default: toate)")
    ap.add_argument("--vel-ratio", type=float, default=None, help="override VEL_RATIO_THR (v3g)")
    ap.add_argument("--tau-admit", type=float, default=None, help="override TAU_ADMIT_DRM (v7d2 recalibrare L)")
    ap.add_argument("--out", required=True, help="subdir rezultat: <out>/baseline/")
    ap.add_argument("--max-regression", type=float, default=None,
                    help="Stop daca dR cumulativ vs baseline > prag")
    # Catastrofe per-secventa: daca N secvente individuale au dR > prag, oprim.
    # Mai util decat cumulativ: o singura secventa de +0.40 nu ne trezeste, 5 da.
    ap.add_argument("--max-catastrophes", type=int, default=None,
                    help="Stop daca N secvente au dR individual > --catastrophe-thr")
    ap.add_argument("--catastrophe-thr", type=float, default=0.25,
                    help="Prag dR per secventa pentru a fi considerata catastrofa (default 0.25)")
    ap.add_argument("--mem-short", type=int, default=None)
    ap.add_argument("--mem-long",  type=int, default=None)
    args = ap.parse_args()

    global TRACKER_NAME
    if args.backbone == "L":
        TRACKER_NAME = "sam21pp-L"

    if args.tracker == "baseline":
        from dam4sam_tracker import DAM4SAMTracker as TC
    elif args.tracker in ("v7", "v7d2"):
        from dam4sam_tracker_v7 import DAM4SAMTrackerV7 as TC
        if args.tracker == "v7d2":
            TC.USE_D2 = True
    elif args.tracker == "v6":
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

    if args.mem_short is not None and hasattr(TC, "MEM_WINDOW_SHORT"):
        TC.MEM_WINDOW_SHORT = args.mem_short
    if args.mem_long is not None and hasattr(TC, "MEM_WINDOW_LONG"):
        TC.MEM_WINDOW_LONG = args.mem_long
    if args.vel_ratio is not None and hasattr(TC, "VEL_RATIO_THR"):
        TC.VEL_RATIO_THR = args.vel_ratio
    if args.tau_admit is not None and hasattr(TC, "TAU_ADMIT_DRM"):
        TC.TAU_ADMIT_DRM = args.tau_admit
        print(f"[run_eval_170] TAU_ADMIT_DRM override -> {args.tau_admit}")

    seqs   = get_all_seqs()
    if args.seqs:
        wanted = [s.strip() for s in args.seqs.split(",") if s.strip()]
        missing = [s for s in wanted if s not in seqs]
        if missing:
            print(f"[run_eval_170] ATENTIE: secvente inexistente: {missing}")
        seqs = [s for s in seqs if s in wanted]
    out_dir = RESULTS_ROOT / args.out / "baseline"

    print(f"[run_eval_170] tracker={args.tracker} backbone={args.backbone} out={args.out}")
    print(f"[run_eval_170] {len(seqs)} secvente totale")
    if args.max_regression is not None:
        print(f"[run_eval_170] GREEDY: stop daca dR_cumul > {args.max_regression:+.3f} dupa min 10 seq")
    if args.max_catastrophes is not None:
        print(f"[run_eval_170] GREEDY: stop daca {args.max_catastrophes} seq individuale au dR > {args.catastrophe_thr:+.2f}")

    ckpt, cfg = determine_tracker(TRACKER_NAME)
    shared = build_sam2_video_predictor(cfg, ckpt, device="cuda:0")
    print(f"[run_eval_170] model incarcat, VRAM {torch.cuda.memory_allocated()/1e9:.2f} GB\n")

    cumul_dr    = 0.0
    n_compared  = 0
    n_catastrophes = 0

    for seq_name in seqs:
        seq_dir = SEQUENCES_DIR / seq_name
        if not seq_dir.exists():
            print(f"[SKIP] {seq_name}: lipseste"); continue
        objects = get_objects(seq_dir)
        frames  = get_frames(seq_dir)
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
                init_mask   = init_region.rasterize((0, 0, img_w - 1, img_h - 1)).astype(np.uint8)
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

        # ── Monitorizare greedy ──────────────────────────────────────────────
        r_base = quick_r(BASELINE_DIR, seq_name)
        r_new  = quick_r(out_dir, seq_name)
        if r_base is not None and r_new is not None:
            dr = r_new - r_base
            cumul_dr  += dr
            n_compared += 1
            is_cat = dr > args.catastrophe_thr
            if is_cat:
                n_catastrophes += 1
            flag = "WIN" if dr < -0.01 else ("CATASTROFA" if is_cat else ("LOSS" if dr > 0.01 else "TIE"))
            print(f"  [dR] {seq_name}: base={r_base:.3f} nou={r_new:.3f} "
                  f"dR={dr:+.3f} [{flag}] | cumul({n_compared}seq)={cumul_dr:+.3f} cat={n_catastrophes}")
            stop = False
            if (args.max_regression is not None and n_compared >= 10
                    and cumul_dr > args.max_regression):
                print(f"\n[STOP-GREEDY] dR_cumul={cumul_dr:+.3f} > limita {args.max_regression:.3f} "
                      f"dupa {n_compared} seq. Oprim.")
                stop = True
            if (args.max_catastrophes is not None
                    and n_catastrophes >= args.max_catastrophes):
                print(f"\n[STOP-CATASTROFE] {n_catastrophes} secvente cu dR > {args.catastrophe_thr:.2f}. "
                      f"Modelul cauzeaza pierderi masive sistematic. Oprim.")
                stop = True
            if stop:
                break
        print()

    print("[run_eval_170] Gata!")


if __name__ == "__main__":
    main()
