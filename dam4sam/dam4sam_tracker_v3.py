"""
DAM4SAMTrackerV3 -- the surgical correction of V2.

V2 diagnosis (hard development subset, mask-emptiness proxy): a large
intervention with a net effect of ~0. Large wins (seagull -0.33, fish2 -0.20,
tank-16 -0.11) paid for by large losses (rabbit +0.36, bull +0.30, uav1 +0.16).
The signature of the losses is a Kalman filter that keeps a stale velocity after
a loss and locks onto a distractor.

V3 keeps the wins and removes the losses with three changes:

  1. MOTION AS A TIEBREAKER, NOT A GLOBAL BLEND.
     The default stays argmax(sam_iou), identical to Base+. The filter is
     consulted ONLY when the top two SAM IoUs are close (<= TIE_MARGIN) and the
     filter is "healthy" => on unambiguous frames the behaviour is EXACTLY Base+,
     so a large loss is impossible by construction.

  2. A CORRECT FILTER LIFECYCLE.
     - the filter is updated only when the chosen mask is confident
       (IoU > UPDATE_IOU, stable size, >= KF_GATE_PIXELS), so the velocity
       estimate is never poisoned with garbage.
     - after MAX_COAST consecutive lost frames the filter is considered unsafe and
       is re-initialised at the next good re-acquisition instead of coasting.

  3. CONDITIONAL POST-PROCESSING.
     keep_largest_component ONLY when the second component is smaller than
     SECOND_FRAC * the first (a clear spurious blob). Otherwise the mask is kept
     whole -- objects that legitimately split (a gull in flight, a partly occluded
     rabbit) must not be truncated.

Subclasses DAM4SAMTracker -- dam4sam_tracker.py is NOT modified; DRM logic identical.

Post-challenge ablation (docs/ABLATION.md): of the changes above, the bounded
memory window introduced here (MEM_WINDOW, below) is the one that moves the
metrics; in V3g the motion tiebreaker turns out to be unreachable.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import torch
from scipy import ndimage

from dam4sam_tracker import DAM4SAMTracker, DRM_MAX_FRAMES
from vot.region.shapes import Mask, Rectangle
from vot.region.raster import calculate_overlaps
from utils.utils import keep_largest_component
from motion import KalmanBoxTracker, mask_to_bbox, bbox_iou


class DAM4SAMTrackerV3(DAM4SAMTracker):
    # -- Main toggles ----------------------------------------------------
    USE_MOTION   = True
    USE_POSTPROC = True

    # -- Motion as a tiebreaker ------------------------------------------
    TIE_MARGIN   = 0.15   # consult the filter only if the top1-top2 SAM IoU gap <= this
    MOTION_WEIGHT = 0.50  # in the ambiguous zone: (1-w)*sam + w*motion

    # -- Filter lifecycle ------------------------------------------------
    UPDATE_IOU     = 0.70  # update the filter only above this predicted IoU
    KF_GATE_PIXELS = 10    # ... and above this many mask pixels
    SIZE_LO, SIZE_HI = 0.5, 2.0  # ... and if the area stays within [0.5x, 2x] of the median
    MAX_COAST      = 5     # after this many consecutive lost frames -> re-init the filter

    # -- Conditional post-processing --------------------------------------
    SECOND_FRAC = 0.30     # cut only if component #2 < 0.30 * component #1

    # -- Bounded non-conditioning memory window (re-detection fix + OOM fix)
    # Bisected on rabbit: 60 is too small (re-detection fails after a long
    # absence), 100+ restores re-detection to baseline level. 300 covers long
    # absences while staying bounded -> OOM-safe on sequences of thousands of
    # frames (keeping all of them ran out of memory).
    # This is the component the ablation attributes the measured gain to.
    MEM_WINDOW = 300

    # -- VRAM fix: offload frame outputs to CPU --
    def init_state_tw(self):
        state = super().init_state_tw()
        state["storage_device"] = torch.device("cpu")
        return state

    @torch.inference_mode()
    def initialize(self, image, init_mask, bbox=None):
        out = super().initialize(image, init_mask, bbox=bbox)
        self.kf = None
        self._lost_streak = 0
        self._need_reinit = False
        if self.USE_MOTION:
            init_m = out.get("pred_mask")
            if init_m is not None:
                bb = mask_to_bbox(init_m)
                if bb is not None:
                    self.kf = KalmanBoxTracker(bb)
        return out

    # -- Conditional post-processing: cut only clearly spurious blobs ------
    def _conditional_postproc(self, m):
        if not self.USE_POSTPROC:
            return m
        if int((m == 1).sum()) == 0:
            return m
        labeled, ncomp = ndimage.label(m)
        if ncomp <= 1:
            return m
        sizes = ndimage.sum(np.ones_like(m), labeled, index=range(1, ncomp + 1))
        order = np.argsort(sizes)[::-1]
        largest, second = sizes[order[0]], sizes[order[1]]
        if second < self.SECOND_FRAC * largest:
            # clearly spurious blob -> keep only the dominant component
            return (labeled == (order[0] + 1)).astype(np.uint8)
        return m  # legitimate split -> keep the mask whole

    def _motion_gate_open(self):
        """Gate for the motion tiebreaker. V3: always open (motion intervenes
        whenever SAM is ambiguous and the filter is healthy). Subclasses (v3g)
        can close it when the motion is too slow to be informative, which falls
        back to Base+ behaviour."""
        return True

    @torch.inference_mode()
    def track(self, image, init=False):
        prepared_img = self._prepare_image(image).unsqueeze(0)
        if not init:
            self.frame_index += 1
            self.inference_state["num_frames"] += 1
        self.inference_state["images"][self.frame_index] = prepared_img

        for out in self.predictor.propagate_in_video(
            self.inference_state,
            start_frame_idx=self.frame_index,
            max_frame_num_to_track=0,
            return_all_masks=True,
        ):
            if len(out) == 3:
                out_frame_idx, _, out_mask_logits = out
                m = (out_mask_logits[0][0] > 0.0).float().cpu().numpy().astype(np.uint8)
            else:
                out_frame_idx, _, out_mask_logits, alternative_masks_ious = out
                all_masks, out_all_ious = alternative_masks_ious

                # -- Mask selection: SAM argmax by default (= Base+) -------
                ious = np.asarray(out_all_ious, dtype=np.float64)
                m_idx = int(np.argmax(ious))

                # -- Motion tiebreaker ONLY when ambiguous + filter healthy -
                kf_healthy = (
                    self.USE_MOTION and self.kf is not None
                    and not self._need_reinit and self._lost_streak == 0
                )
                if kf_healthy and len(all_masks) > 1:
                    srt = np.sort(ious)[::-1]
                    if (srt[0] - srt[1]) <= self.TIE_MARGIN and self._motion_gate_open():
                        pred_bbox = self.kf.predict()
                        best, best_score = m_idx, -1.0
                        # only candidates close to the top compete
                        for i, (cand_logits, sam_iou) in enumerate(zip(all_masks, ious)):
                            if (srt[0] - sam_iou) > self.TIE_MARGIN:
                                continue
                            cand_bin = (cand_logits[0][0] > 0.0).cpu().numpy().astype(np.uint8)
                            mot_iou = bbox_iou(mask_to_bbox(cand_bin), pred_bbox)
                            score = (1 - self.MOTION_WEIGHT) * sam_iou + self.MOTION_WEIGHT * mot_iou
                            if score > best_score:
                                best_score, best = score, i
                        m_idx = best

                m_iou = float(ious[m_idx])
                chosen_logits = all_masks[m_idx]
                m = (chosen_logits[0][0] > 0.0).float().cpu().numpy().astype(np.uint8)

                # -- Filter lifecycle --------------------------------------
                if self.USE_MOTION:
                    n_pix = int((m == 1).sum())
                    bb = mask_to_bbox(m) if n_pix >= self.KF_GATE_PIXELS else None
                    size_ok = True
                    if bb is not None and len(self.object_sizes) > 1:
                        med = np.median([s for s in self.object_sizes[-300:] if s >= 1][-10:])
                        if med > 0:
                            r = n_pix / med
                            size_ok = (self.SIZE_LO <= r <= self.SIZE_HI)
                    confident = (bb is not None and m_iou > self.UPDATE_IOU and size_ok)
                    if confident:
                        if self.kf is None or self._need_reinit:
                            self.kf = KalmanBoxTracker(bb)
                            self._need_reinit = False
                        else:
                            self.kf.update(bb)
                        self._lost_streak = 0
                    else:
                        self._lost_streak += 1
                        if self._lost_streak > self.MAX_COAST:
                            self._need_reinit = True

                # -- DRM logic (identical to the original) -----------------
                alternative_masks_list = [
                    mask for i, mask in enumerate(all_masks) if i != m_idx
                ]
                n_pixels = (m == 1).sum()
                self.object_sizes.append(n_pixels)
                if len(self.object_sizes) > 1 and n_pixels >= 1:
                    obj_sizes_ratio = n_pixels / np.median([
                        s for s in self.object_sizes[-300:] if s >= 1
                    ][-10:])
                else:
                    obj_sizes_ratio = -1

                if (
                    m_iou > 0.8
                    and obj_sizes_ratio >= 0.8
                    and obj_sizes_ratio <= 1.2
                    and n_pixels >= 1
                    and (self.frame_index - self.last_added > 5 or self.last_added == -1)
                ):
                    alt_np = [
                        Mask((m_[0][0] > 0.0).cpu().numpy()).rasterize(
                            (0, 0, self.img_width - 1, self.img_height - 1)
                        ).astype(np.uint8)
                        for m_ in alternative_masks_list
                    ]
                    chosen_mask_np = m.copy()
                    chosen_bbox_r = Rectangle.convert(Mask(chosen_mask_np))
                    alt_np = [
                        np.logical_and(a, np.logical_not(chosen_mask_np)).astype(np.uint8)
                        for a in alt_np
                    ]
                    alt_np = [keep_largest_component(a) for a in alt_np if np.sum(a) >= 1]
                    if len(alt_np) > 0:
                        alt_np = [
                            np.logical_or(a, chosen_mask_np).astype(np.uint8) for a in alt_np
                        ]
                        alt_bboxes = [Rectangle.convert(Mask(a)) for a in alt_np]
                        ious_drm = [calculate_overlaps([chosen_bbox_r], [b])[0] for b in alt_bboxes]
                        if np.min(np.array(ious_drm)) <= 0.7:
                            self.last_added = self.frame_index
                            drm_frames = (
                                self.inference_state.get("adds_in_drm_per_obj", {}).get(0, [])
                            )
                            if len(drm_frames) < DRM_MAX_FRAMES:
                                self.predictor.add_to_drm(
                                    inference_state=self.inference_state,
                                    frame_idx=out_frame_idx,
                                    obj_id=0,
                                )

            # -- Conditional post-processing of the output mask ------------
            m = self._conditional_postproc(m)

            self.inference_state["images"].pop(self.frame_index)

            # -- Pruning non-conditioning frame outputs (sliding window) ---
            # Keep the last MEM_WINDOW frames: SAM2 needs num_maskmem(7) x
            # stride(5) ~= 35 frames of history for temporal attention and
            # re-detection. The earlier keep-1 policy destroyed re-acquisition
            # after an absence (rabbit never came back).
            output_dict = self.inference_state["output_dict"]
            cutoff = self.frame_index - self.MEM_WINDOW
            for f in list(output_dict["non_cond_frame_outputs"].keys()):
                if f < cutoff:
                    del output_dict["non_cond_frame_outputs"][f]
                    for obj_state in self.inference_state["output_dict_per_obj"].values():
                        obj_state["non_cond_frame_outputs"].pop(f, None)
                    self.inference_state["consolidated_frame_inds"]["non_cond_frame_outputs"].discard(f)

            return {"pred_mask": m}
