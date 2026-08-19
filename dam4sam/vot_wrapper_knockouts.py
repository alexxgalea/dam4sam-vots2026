"""
VOT wrapper for the knock-out variants (see docs/ABLATION.md).

Identical to vot_wrapper_v3g.py, except that the tracker class is selected
through the KO_VARIANT environment variable (ko_tie | ko_speed | ko_lifecycle |
ko_postproc | ko_blend | ko_motion | ko_chassis), set from trackers.ini via
`env_KO_VARIANT = ...` -- one wrapper serves all knock-out entries.

Note: the official runs are executed standalone through
scripts/run_eval_170.py (which avoids TRAX); this wrapper exists so that the
trackers.ini entries are complete for `vot pack` and for cross-checks through
the toolkit.
"""
import sys, random, os
import numpy as np
from PIL import Image
import torch
import yaml

from dam4sam_tracker_knockouts import KNOCKOUTS
from utils.utils import determine_tracker
from sam2.build_sam import build_sam2_video_predictor
import utils.vot_helper as vot

TRACKER_NAME = "sam21pp-B"

KO_VARIANT = os.environ.get("KO_VARIANT", "")
if KO_VARIANT not in KNOCKOUTS:
    sys.stderr.write(f"[vot_wrapper_knockouts] KO_VARIANT invalid: {KO_VARIANT!r} "
                     f"(expected one of {sorted(KNOCKOUTS)})\n")
    sys.exit(1)
TrackerCls = KNOCKOUTS[KO_VARIANT]

with open(os.path.join(os.path.dirname(__file__), "dam4sam_config.yaml")) as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

seed = config["seed"]
random.seed(seed); os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed(seed)


def _make_object_tracker(shared_predictor):
    class _ObjectTracker(TrackerCls):
        def __init__(self, predictor):
            self.checkpoint, self.model_cfg = determine_tracker(TRACKER_NAME)
            self.input_image_size = 1024
            self.img_mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)[:, None, None]
            self.img_std  = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)[:, None, None]
            self.predictor = predictor
            self.tracking_times = []
            self.kf = None
    return _ObjectTracker(shared_predictor)


def make_full_size(x, output_sz):
    if x.shape[0] == output_sz[1] and x.shape[1] == output_sz[0]:
        return x
    pad_x = output_sz[0] - x.shape[1]
    if pad_x < 0: x = x[:, :x.shape[1] + pad_x]; pad_x = 0
    pad_y = output_sz[1] - x.shape[0]
    if pad_y < 0: x = x[:x.shape[0] + pad_y, :]; pad_y = 0
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

    checkpoint, model_cfg = determine_tracker(TRACKER_NAME)
    shared_predictor = build_sam2_video_predictor(model_cfg, checkpoint, device="cuda:0")

    trackers = []
    for mask in init_masks:
        t = _make_object_tracker(shared_predictor)
        t.initialize(image, mask)
        trackers.append(t)

    while True:
        imagefile = handle.frame()
        if not imagefile:
            break
        image = Image.open(imagefile)
        status = [t.track(image)['pred_mask'] for t in trackers]
        handle.report(status)


if __name__ == "__main__":
    main()
