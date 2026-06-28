"""
Evaluator pentru VOST-val (vots2024/votstval).
Produce .bin files in workspace_vost_val/results/<tracker>/baseline/
compatibile cu `vot analysis --workspace workspace_vost_val`.

Dupa rulare: vot analysis --workspace workspace_vost_val
"""
import sys, os, re, numpy as np
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

WORKSPACE   = Path("/home/alexx/Projects/VOTSTracker/workspace_vost_val")
SEQUENCES   = WORKSPACE / "sequences"
RESULTS     = WORKSPACE / "results"
TRACKER_NAME = "sam21pp-B"

seed = 0
import random
random.seed(seed); os.environ["PYTHONHASHSEED"] = str(seed)
np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed(seed)


def get_all_seqs():
    return sorted([d.name for d in SEQUENCES.iterdir() if d.is_dir()])


def get_objects(seq_dir):
    # VOST-val foloseste 'groundtruth_object<N>.txt' (NU 'groundtruth_obj<N>').
    gt_files = sorted(seq_dir.glob("groundtruth_object*.txt"))
    if gt_files:
        objs = []
        for f in gt_files:
            mm = re.search(r"groundtruth_object(\d+)\.txt$", f.name)
            if mm:
                objs.append((int(mm.group(1)), f))
        return objs
    # fallback: schema VOTS2026 'groundtruth_obj<N>.txt' / 'groundtruth.txt'
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
    ap.add_argument("--tracker", choices=["baseline", "v3", "v3b", "v4", "v6"],
                    default="baseline")
    args = ap.parse_args()

    tracker_label = {
        "baseline": "dam4sam_b",
        "v3":  "dam4sam_v3",
        "v3b": "dam4sam_v3b",
        "v4":  "dam4sam_v4",
        "v6":  "dam4sam_v6",
    }[args.tracker]

    if args.tracker == "baseline":
        from dam4sam_tracker import DAM4SAMTracker as TC
    elif args.tracker == "v3":
        from dam4sam_tracker_v3 import DAM4SAMTrackerV3 as TC
    elif args.tracker == "v3b":
        from dam4sam_tracker_v3b import DAM4SAMTrackerV3b as TC
    elif args.tracker == "v4":
        from dam4sam_tracker_v4 import DAM4SAMTrackerV4 as TC
    elif args.tracker == "v6":
        from dam4sam_tracker_v6 import DAM4SAMTrackerV6 as TC

    out_dir = RESULTS / tracker_label / "baseline"
    seqs = get_all_seqs()

    print(f"[vostval] tracker={tracker_label} output={out_dir}")
    print(f"[vostval] {len(seqs)} secvente in {SEQUENCES}")

    ckpt, cfg = determine_tracker(TRACKER_NAME)
    shared = build_sam2_video_predictor(cfg, ckpt, device="cuda:0")
    print(f"[vostval] model incarcat, VRAM {torch.cuda.memory_allocated()/1e9:.2f} GB\n")

    for seq_name in seqs:
        seq_dir = SEQUENCES / seq_name
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
                    print(f"  obj{obj_id} frame {fi}: eroare: {e}")
                    pred_masks.append(Mask(np.zeros((img_h, img_w), dtype=np.uint8)))
                torch.cuda.empty_cache()

            op.parent.mkdir(parents=True, exist_ok=True)
            write_trajectory(str(op), pred_masks)
            print(f"  obj{obj_id}: {len(pred_masks)} cadre -> {op.name}")

        torch.cuda.empty_cache()
        print(f"  VRAM: {torch.cuda.memory_allocated()/1e9:.2f} GB\n")

    print(f"[vostval] Gata! Ruleaza: vot analysis --workspace {WORKSPACE}")


if __name__ == "__main__":
    main()
