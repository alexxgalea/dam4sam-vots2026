"""
VOT2026 wrapper for DAM4SAM-SAM3.

Multi-object: one DAM4SAMTrackerSAM3 instance per object, sharing a single
loaded SAM3 predictor so model weights only occupy GPU memory once.

NOTE: All diagnostic output uses sys.stderr (via log()). The vot-toolkit
starts the tracker with stderr=subprocess.STDOUT so both streams merge into
the TRAX pipe. tqdm is disabled in propagate_in_video to prevent carriage-
return sequences from corrupting the TRAX protocol stream.
"""
import sys
import os
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# See dam4sam_tracker_sam3.py: SAM3_ROOT / SAM3_CHECKPOINT are overridable via env.
SAM3_ROOT = os.environ.get(
    "SAM3_ROOT",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "sam3_upstream"),
)
SAM3_CHECKPOINT = os.environ.get("SAM3_CHECKPOINT",
                                 os.path.join(SAM3_ROOT, "checkpoints", "sam3.pt"))
sys.path.insert(0, SAM3_ROOT)

import random
import numpy as np
from PIL import Image
import torch
import yaml

from dam4sam_tracker_sam3 import DAM4SAMTrackerSAM3
import utils.vot_helper as vot

with open(os.path.join(os.path.dirname(__file__), "dam4sam_config.yaml")) as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

seed = config["seed"]
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)


def log(msg):
    """Diagnostic messages to stderr (never stdout — TRAX uses stdout)."""
    print(msg, file=sys.stderr, flush=True)


def make_full_size(x, output_sz):
    """Zero-pad mask x (right and down) to match output_sz (width, height)."""
    if x.shape[0] == output_sz[1] and x.shape[1] == output_sz[0]:
        return x
    pad_x = output_sz[0] - x.shape[1]
    if pad_x < 0:
        x = x[:, :x.shape[1] + pad_x]
        pad_x = 0
    pad_y = output_sz[1] - x.shape[0]
    if pad_y < 0:
        x = x[:x.shape[0] + pad_y, :]
        pad_y = 0
    return np.pad(x, ((0, pad_y), (0, pad_x)), 'constant', constant_values=0)


@torch.inference_mode()
def main():
    handle = vot.VOT("mask", multiobject=True)
    objects = handle.objects()

    imagefile = handle.frame()
    if not imagefile:
        sys.exit(0)

    image = Image.open(imagefile)
    init_masks = [make_full_size(m, (image.width, image.height)) for m in objects]

    # ── Build SAM3 predictor once and share it across all objects ─────────────
    from sam3.model_builder import build_tracker
    from dam4sam_tracker_sam3 import _load_sam3_tracker_checkpoint
    ckpt_path = SAM3_CHECKPOINT
    log("[vot_wrapper_sam3] Loading SAM3 model...")
    shared_predictor = build_tracker(
        apply_temporal_disambiguation=True, with_backbone=True
    ).cuda().eval()
    _load_sam3_tracker_checkpoint(shared_predictor, ckpt_path)
    log("[vot_wrapper_sam3] SAM3 loaded OK")
    # ─────────────────────────────────────────────────────────────────────────

    trackers = []
    for i, mask in enumerate(init_masks):
        log(f"[vot_wrapper_sam3] Initializing object {i}...")
        try:
            t = DAM4SAMTrackerSAM3(shared_predictor=shared_predictor)
            t.initialize(image, mask)
            trackers.append(t)
            log(f"[vot_wrapper_sam3] Object {i} initialized OK")
        except Exception as e:
            import traceback
            log(f"[vot_wrapper_sam3] ERROR init obj {i}: {e}")
            traceback.print_exc(file=sys.stderr)
            raise

    frame_num = 0
    log("[vot_wrapper_sam3] Entering tracking loop...")
    while True:
        try:
            imagefile = handle.frame()
        except Exception as e:
            import traceback
            log(f"[vot_wrapper_sam3] ERROR handle.frame(): {type(e).__name__}: {e}")
            traceback.print_exc(file=sys.stderr)
            raise
        if not imagefile:
            break

        frame_num += 1
        image = Image.open(imagefile)

        status = []
        for i, t in enumerate(trackers):
            try:
                outputs = t.track(image)
                status.append(outputs['pred_mask'])
                torch.cuda.empty_cache()  # release fragmented VRAM between objects
            except Exception as e:
                import traceback
                log(f"[vot_wrapper_sam3] ERROR track frame={frame_num} obj={i}: {type(e).__name__}: {e}")
                traceback.print_exc(file=sys.stderr)
                raise

        try:
            handle.report(status)
        except Exception as e:
            import traceback
            log(f"[vot_wrapper_sam3] ERROR handle.report() frame={frame_num}: {type(e).__name__}: {e}")
            traceback.print_exc(file=sys.stderr)
            raise


if __name__ == "__main__":
    main()
