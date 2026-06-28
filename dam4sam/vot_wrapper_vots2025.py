"""
VOTS2025 wrapper for DAM4SAM.

DAM4SAM is single-object; VOTS2025 is multi-object. This wrapper runs one
DAM4SAMTracker instance per object, sharing a single loaded predictor so
model weights only occupy GPU memory once.
"""
import sys
import random
import os

import numpy as np
from PIL import Image
import torch
import yaml

from dam4sam_tracker import DAM4SAMTracker
from utils.utils import determine_tracker
from sam2.build_sam import build_sam2_video_predictor
import utils.vot_helper as vot

TRACKER_NAME = "sam21pp-B"

with open(os.path.join(os.path.dirname(__file__), "dam4sam_config.yaml")) as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

seed = config["seed"]
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)


class _ObjectTracker(DAM4SAMTracker):
    """Single-object DAM4SAM tracker that reuses an already-loaded predictor."""
    def __init__(self, shared_predictor, tracker_name=TRACKER_NAME):
        # Replicate __init__ but skip build_sam2_video_predictor
        self.checkpoint, self.model_cfg = determine_tracker(tracker_name)
        self.input_image_size = 1024
        self.img_mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)[:, None, None]
        self.img_std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)[:, None, None]
        self.predictor = shared_predictor
        self.tracking_times = []


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
@torch.cuda.amp.autocast()
def main():
    handle = vot.VOT("mask", multiobject=True)
    objects = handle.objects()

    imagefile = handle.frame()
    if not imagefile:
        sys.exit(0)

    image = Image.open(imagefile)
    init_masks = [make_full_size(m, (image.width, image.height)) for m in objects]

    # Load predictor once, share across all per-object trackers
    checkpoint, model_cfg = determine_tracker(TRACKER_NAME)
    shared_predictor = build_sam2_video_predictor(model_cfg, checkpoint, device="cuda:0")

    trackers = []
    for i, mask in enumerate(init_masks):
        t = _ObjectTracker(shared_predictor, tracker_name=TRACKER_NAME)
        t.initialize(image, mask)
        trackers.append(t)

    while True:
        imagefile = handle.frame()
        if not imagefile:
            break

        image = Image.open(imagefile)

        status = []
        for t in trackers:
            outputs = t.track(image)
            status.append(outputs['pred_mask'])

        handle.report(status)


if __name__ == "__main__":
    main()
