"""
DAM4SAM tracker backed by SAM3 instead of SAM2.

Differences from dam4sam_tracker.py:
  - Uses Sam3TrackerPredictor (build_tracker) instead of SAM2VideoPredictor
  - propagate_in_video yields 5 values: (frame_idx, obj_ids, low_res_masks, video_res_masks, obj_scores)
  - DRM trigger: obj_score sigmoid + size consistency (replaces SAM2 alternative-mask IoU check)
  - No return_all_masks / alternative_masks pathway (SAM3 handles multimask internally)
  - init_state_tw_sam3() builds the frame-by-frame inference state for SAM3
"""

import sys
import os
sys.path.insert(0, "/home/alexx/Projects/VOTSTracker/sam3_upstream")

import numpy as np
import yaml
import torch
import torchvision.transforms.functional as F

from pathlib import Path

config_path = Path(__file__).parent / "dam4sam_config.yaml"
with open(config_path) as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

import random
seed = config["seed"]
random.seed(seed)
os.environ['PYTHONHASHSEED'] = str(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)

def _load_sam3_tracker_checkpoint(tracker_model, ckpt_path):
    """
    Load weights from the full sam3.pt checkpoint into a standalone tracker model.

    The checkpoint stores:
      tracker.*                            → tracker weights
      detector.backbone.vision_backbone.*  → backbone weights (strip 'detector.' prefix)

    Use this with build_tracker(with_backbone=True) for standalone inference.
    """
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    full_state = ckpt.get("model", ckpt)
    merged = {}
    for k, v in full_state.items():
        if k.startswith("tracker."):
            merged[k[len("tracker."):]] = v
        elif k.startswith("detector.backbone.vision_backbone."):
            # maps to backbone.vision_backbone.* in standalone tracker
            merged[k[len("detector."):]] = v
    missing, unexpected = tracker_model.load_state_dict(merged, strict=True)
    if missing:
        print(f"[SAM3] WARNING missing keys: {missing[:5]}")
    if unexpected:
        print(f"[SAM3] WARNING unexpected keys: {unexpected[:5]}")
    print(f"[SAM3] Loaded {len(merged)} weights from {ckpt_path}")


# ── DRM trigger thresholds ────────────────────────────────────────────────────
# SAM3 replaces the SAM2 alternative-mask IoU check with object presence score.
# add_to_drm when: score > DRM_SCORE_THRESH AND size ratio in window AND spacing > DRM_MIN_INTERVAL
DRM_SCORE_THRESH = 0.85    # sigmoid(obj_score) > this → confident detection
DRM_SIZE_LOW     = 0.80    # min allowed ratio vs recent median size
DRM_SIZE_HIGH    = 1.20    # max allowed ratio vs recent median size
DRM_MIN_INTERVAL = 5       # minimum frames between two DRM adds
# ─────────────────────────────────────────────────────────────────────────────


class DAM4SAMTrackerSAM3:
    """
    Single-object DAM4SAM tracker using SAM3 as the backbone.

    The public interface mirrors DAM4SAMTracker:
      - initialize(image, init_mask)  → {'pred_mask': np.uint8}
      - track(image)                  → {'pred_mask': np.uint8}
    """

    def __init__(self, shared_predictor=None):
        self.input_image_size = 1008  # SAM3 native resolution
        self.img_mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)[:, None, None]
        self.img_std  = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)[:, None, None]

        if shared_predictor is not None:
            self.predictor = shared_predictor
        else:
            from sam3.model_builder import build_tracker
            ckpt_path = "/home/alexx/Projects/VOTSTracker/sam3_upstream/checkpoints/sam3.pt"
            # with_backbone=True: tracker runs standalone (computes its own image features)
            self.predictor = build_tracker(
                apply_temporal_disambiguation=True, with_backbone=True
            ).cuda().eval()
            _load_sam3_tracker_checkpoint(self.predictor, ckpt_path)

        self.tracking_times = []

    # ── image prep ───────────────────────────────────────────────────────────

    def _prepare_image(self, img_pil):
        """Convert PIL image to normalised [C, H, W] tensor at SAM3 resolution."""
        device = self.inference_state["device"]
        img = torch.from_numpy(np.array(img_pil)).to(device)
        img = img.permute(2, 0, 1).float() / 255.0
        img = F.resize(img, (self.input_image_size, self.input_image_size))
        img = (img - self.img_mean.to(device)) / self.img_std.to(device)
        return img

    # ── inference state ───────────────────────────────────────────────────────

    def init_state_tw_sam3(self):
        """
        Build a frame-by-frame inference state for SAM3 (no video_path needed).
        Mirrors the SAM2 init_state_tw() in dam4sam_tracker.py.
        """
        from collections import OrderedDict
        compute_device = torch.device("cuda")

        # Start from SAM3's own init_state with dummy dimensions; we'll override images.
        inference_state = self.predictor.init_state(
            video_height=None,
            video_width=None,
            num_frames=0,
        )
        # Frame dict populated on-the-fly (same pattern as SAM2 wrapper)
        inference_state["images"] = {}
        inference_state["num_frames"] = 0
        inference_state["offload_video_to_cpu"] = True
        inference_state["offload_state_to_cpu"] = True
        inference_state["device"] = compute_device
        # Store per-frame outputs on CPU to prevent GPU OOM on long sequences.
        # With 5+ objects each accumulating ~5 MB/frame on GPU, frame 393 × 5 objects
        # exhausts the RTX 3080's 12 GB VRAM → "CUDA driver error: unknown error".
        # CPU storage means only model weights + current frame live on GPU (~4 GB),
        # keeping VRAM constant regardless of sequence length.
        inference_state["storage_device"] = torch.device("cpu")
        return inference_state

    # ── public API ────────────────────────────────────────────────────────────

    @torch.inference_mode()
    def initialize(self, image, init_mask):
        """Initialise tracker on the first frame."""
        if isinstance(init_mask, list):
            init_mask = init_mask[0]

        self.frame_index  = 0
        self.object_sizes = []
        self.last_added   = -1
        self.img_width    = image.width
        self.img_height   = image.height

        self.inference_state = self.init_state_tw_sam3()
        video_width, video_height = image.size
        self.inference_state["video_height"] = video_height
        self.inference_state["video_width"]  = video_width

        # SAM3's _get_image_feature adds .unsqueeze(0) → store [C, H, W]
        prepared_img = self._prepare_image(image)
        self.inference_state["images"][0] = prepared_img
        self.inference_state["num_frames"] = 1

        # Warm up image features (no reset_state needed — fresh inference_state each init)
        self.predictor._get_image_feature(self.inference_state, frame_idx=0, batch_size=1)

        # Register the init mask
        _, _, _, video_res_masks = self.predictor.add_new_mask(
            inference_state=self.inference_state,
            frame_idx=0,
            obj_id=0,
            mask=torch.tensor(init_mask, dtype=torch.bool),
        )

        m = (video_res_masks[0, 0] > 0).float().cpu().numpy().astype(np.uint8)

        # Consolidate the init frame's temp output into output_dict["cond_frame_outputs"]
        # and run the memory encoder so future frames have a proper conditioning anchor.
        # (SAM2 does this automatically at the start of every propagate_in_video;
        #  SAM3 requires an explicit call when propagate_preflight=False.)
        self.predictor.propagate_in_video_preflight(
            self.inference_state, run_mem_encoder=True
        )

        self.inference_state["images"].pop(0, None)
        return {"pred_mask": m}

    @torch.inference_mode()
    def track(self, image):
        """Track the object in the next frame."""
        self.frame_index += 1
        self.inference_state["num_frames"] += 1

        # SAM3's _get_image_feature adds .unsqueeze(0) itself → store [C, H, W] not [1, C, H, W]
        prepared_img = self._prepare_image(image)
        self.inference_state["images"][self.frame_index] = prepared_img

        m = None
        for out in self.predictor.propagate_in_video(
            self.inference_state,
            start_frame_idx=self.frame_index,
            max_frame_num_to_track=0,
            reverse=False,
            tqdm_disable=True,
        ):
            frame_idx, obj_ids, low_res_masks, video_res_masks, obj_scores = out

            m = (video_res_masks[0, 0] > 0.0).float().cpu().numpy().astype(np.uint8)

            # ── DRM trigger (SAM3 version) ──────────────────────────────────
            # Use object presence score as proxy for detection confidence.
            # SAM3 doesn't expose alternative masks; we rely on sigmoid(obj_score).
            obj_prob = float(torch.sigmoid(obj_scores[0, 0]))

            n_pixels = int((m == 1).sum())
            self.object_sizes.append(n_pixels)

            if len(self.object_sizes) > 1 and n_pixels >= 1:
                recent = [s for s in self.object_sizes[-300:] if s >= 1][-10:]
                obj_sizes_ratio = n_pixels / float(np.median(recent)) if recent else -1.0
            else:
                obj_sizes_ratio = -1.0

            enough_gap = (self.frame_index - self.last_added > DRM_MIN_INTERVAL
                         or self.last_added == -1)

            if (obj_prob > DRM_SCORE_THRESH
                    and DRM_SIZE_LOW <= obj_sizes_ratio <= DRM_SIZE_HIGH
                    and n_pixels >= 1
                    and enough_gap):
                self.last_added = self.frame_index
                self.predictor.add_to_drm(
                    inference_state=self.inference_state,
                    frame_idx=frame_idx,
                    obj_id=0,
                )
            # ── end DRM trigger ─────────────────────────────────────────────

        self.inference_state["images"].pop(self.frame_index, None)

        # Prune non-cond frame outputs for all frames before current to free memory.
        # These are never needed again: add_to_drm already moved any DRM-triggered frames
        # to cond_frame_outputs. Without pruning, a 5797-frame × 5-object sequence
        # accumulates tens of GB even on CPU.
        output_dict = self.inference_state["output_dict"]
        for f in list(output_dict["non_cond_frame_outputs"].keys()):
            if f < self.frame_index:
                del output_dict["non_cond_frame_outputs"][f]
                for obj_state in self.inference_state["output_dict_per_obj"].values():
                    obj_state["non_cond_frame_outputs"].pop(f, None)
                self.inference_state["consolidated_frame_inds"]["non_cond_frame_outputs"].discard(f)

        return {"pred_mask": m}
