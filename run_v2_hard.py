"""
Evaluare standalone DAM4SAMTrackerV2 pe setul de secvente grele.

Evita TRAX/VOT evaluate (care cauza OOM in subprocese dupa crash-uri repetate).
Ruleaza tracker-ul direct in Python, fara subprocess overhead.

Output: workspace_hard/results/dam4sam_v2/baseline/<seq>/<seq>[_obj<N>]_001.bin
        (acelasi format ca workspace2026/results/dam4sam_b/baseline/)
"""
import sys
import os
sys.path.insert(0, "/home/alexx/Projects/VOTSTracker")
sys.path.insert(0, "/home/alexx/Projects/VOTSTracker/dam4sam")
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import random
import numpy as np
import torch
import yaml
from pathlib import Path
from PIL import Image

from vot.region.io import read_trajectory, write_trajectory
from vot.region.shapes import Mask

from dam4sam_tracker_v2 import DAM4SAMTrackerV2
from utils.utils import determine_tracker
from sam2.build_sam import build_sam2_video_predictor

# ── Config ────────────────────────────────────────────────────────────────────
SEQUENCES_DIR = Path("/home/alexx/Projects/VOTSTracker/workspace2026/sequences")
OUTPUT_DIR    = Path("/home/alexx/Projects/VOTSTracker/workspace_hard/results/dam4sam_v2/baseline")
TRACKER_NAME  = "sam21pp-B"

HARD_SEQS = [
    "uav1", "tank-9", "fish2", "tank-16", "bicycle-7", "bull", "motorcycle-1",
    "person-12", "wubble_bubble", "seagull", "rabbit", "motorcycle", "motorcycle-18",
    "racing-16", "helicopter", "deer", "diabolo", "colorfish", "bus-2", "crab-6",
    "carchase", "birds1",
]

seed = 0
random.seed(seed)
os.environ["PYTHONHASHSEED"] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)


# ── Helpers ───────────────────────────────────────────────────────────────────

def get_objects(seq_dir):
    """Returneaza lista de (obj_id, groundtruth_path) pentru o secventa."""
    gt_files = sorted(seq_dir.glob("groundtruth_obj*.txt"))
    if gt_files:
        objs = []
        for f in gt_files:
            import re
            m = re.search(r"groundtruth_obj(\d+)\.txt$", f.name)
            if m:
                objs.append((int(m.group(1)), f))
        return objs
    else:
        gt = seq_dir / "groundtruth.txt"
        if gt.exists():
            return [(1, gt)]
    return []


def get_frames(seq_dir):
    """Returneaza cadrele sortate din directorul color."""
    return sorted((seq_dir / "color").glob("*.jpg")) or sorted((seq_dir / "color").glob("*.png"))


def result_path(seq_name, obj_id, n_objects):
    """Calea fisierului de rezultat .bin."""
    if n_objects == 1:
        fname = f"{seq_name}_001.bin"
    else:
        fname = f"{seq_name}_obj{obj_id}_001.bin"
    return OUTPUT_DIR / seq_name / fname


class _SharedTracker(DAM4SAMTrackerV2):
    """DAM4SAMTrackerV2 cu predictor partajat, fara init propriu."""
    def __init__(self, shared_predictor):
        self.checkpoint, self.model_cfg = determine_tracker(TRACKER_NAME)
        self.input_image_size = 1024
        self.img_mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
        self.img_std  = torch.tensor([0.229, 0.224, 0.225])[:, None, None]
        self.predictor = shared_predictor
        self.tracking_times = []
        self.kf = None

    def init_state_tw(self):
        return DAM4SAMTrackerV2.init_state_tw(self)


# ── Main ──────────────────────────────────────────────────────────────────────

@torch.inference_mode()
def main():
    print("[run_v2_hard] Incarcare SAM2.1-B+...")
    ckpt, cfg = determine_tracker(TRACKER_NAME)
    shared_predictor = build_sam2_video_predictor(cfg, ckpt, device="cuda:0")
    print(f"[run_v2_hard] Model incarcat. VRAM: {torch.cuda.memory_allocated()/1e9:.2f} GB")
    print(f"[run_v2_hard] USE_MOTION={DAM4SAMTrackerV2.USE_MOTION}, USE_POSTPROC={DAM4SAMTrackerV2.USE_POSTPROC}")
    print()

    for seq_name in HARD_SEQS:
        seq_dir = SEQUENCES_DIR / seq_name
        if not seq_dir.exists():
            print(f"[SKIP] {seq_name}: secventa lipseste")
            continue

        objects = get_objects(seq_dir)
        frames  = get_frames(seq_dir)
        if not objects or not frames:
            print(f"[SKIP] {seq_name}: lipsesc obiecte sau cadre")
            continue

        n_objects = len(objects)

        # Verifica daca toate obiectele sunt deja procesate
        all_done = all(result_path(seq_name, oid, n_objects).exists() for oid, _ in objects)
        if all_done:
            print(f"[SKIP] {seq_name}: rezultate deja existente")
            continue

        print(f"[SEQ] {seq_name} — {len(frames)} cadre, {n_objects} obiect(e)")

        img0 = Image.open(frames[0])
        img_w, img_h = img0.width, img0.height

        for obj_id, gt_path in objects:
            out_path = result_path(seq_name, obj_id, n_objects)
            if out_path.exists():
                print(f"  obj{obj_id}: deja existent, skip")
                continue

            # Init mask din groundtruth (primul rand = masca de init)
            try:
                gt_regions = read_trajectory(str(gt_path))
                init_region = gt_regions[0]
                init_mask = init_region.rasterize((0, 0, img_w - 1, img_h - 1)).astype(np.uint8)
            except Exception as e:
                print(f"  obj{obj_id}: eroare citire GT: {e}")
                continue

            tracker = _SharedTracker(shared_predictor)
            try:
                tracker.initialize(img0, init_mask)
            except Exception as e:
                print(f"  obj{obj_id}: eroare initialize: {e}")
                continue

            pred_masks = [Mask(init_mask)]

            for frame_idx, frame_path in enumerate(frames[1:], start=1):
                try:
                    img = Image.open(frame_path)
                    out = tracker.track(img)
                    pred_masks.append(Mask(out["pred_mask"]))
                except Exception as e:
                    print(f"  obj{obj_id} frame {frame_idx}: eroare track: {e}")
                    pred_masks.append(Mask(np.zeros((img_h, img_w), dtype=np.uint8)))

                torch.cuda.empty_cache()

            out_path.parent.mkdir(parents=True, exist_ok=True)
            write_trajectory(str(out_path), pred_masks)
            n_perm = int((pred_masks[-1].mask == 1).sum()) if hasattr(pred_masks[-1], 'mask') else -1
            print(f"  obj{obj_id}: {len(pred_masks)} cadre salvate → {out_path.name}")

        torch.cuda.empty_cache()
        print(f"  VRAM: {torch.cuda.memory_allocated()/1e9:.2f} GB")
        print()

    print("[run_v2_hard] Gata!")


if __name__ == "__main__":
    main()
