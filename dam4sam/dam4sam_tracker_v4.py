"""
DAM4SAMTrackerV4 -- v3 + explicit DISAPPEARANCE / REAPPEARANCE handling (long-term).

VOTS2026 emphasises objects that leave the frame and come back. Our dominant
failure mode: during an absence SAM2 hallucinates a mask on a distractor (the
occlusion head, object_score_logits > 0, is over-confident) and pollutes memory,
after which the true target is no longer re-detected correctly.

On top of v3's surgical fixes, V4 adds a training-free presence/absence STATE
MACHINE with hysteresis, driven by three signals that are free at inference time:

  - object_score_logits  : SAM2's native occlusion head (read from predictor.curr_out)
  - m_iou                : predicted IoU of the selected mask
  - appearance_sim       : HSV histogram match between the masked region and the
                           initialization template (rejects distractors)

Behaviour:
  PRESENT --(K consecutive "bad" frames)--> ABSENT  (emit an EMPTY mask)
  ABSENT  --(J consecutive "good" frames)--> PRESENT (re-init the filter, resume)

While ABSENT: the output mask is empty, the filter is not updated, and nothing is
added to the DRM (memory hygiene -> re-detection works from clean templates).

The same appearance_sim signal doubles as a ground-truth-free validation metric:
how well appearance matches on non-empty frames is a proxy for the confidence
that the tracked region is the real object rather than a distractor.

Subclasses DAM4SAMTrackerV3, so the base files are not modified.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import torch
import cv2

from dam4sam_tracker_v3 import DAM4SAMTrackerV3, DRM_MAX_FRAMES
from vot.region.shapes import Mask, Rectangle
from vot.region.raster import calculate_overlaps
from utils.utils import keep_largest_component
from motion import KalmanBoxTracker, mask_to_bbox, bbox_iou


class DAM4SAMTrackerV4(DAM4SAMTrackerV3):
    # -- Absence state machine toggles -------------------------------------
    USE_ABSENCE     = True
    USE_APPEARANCE  = True

    # -- Thresholds for a "good detection" ---------------------------------
    OBJ_LOGIT_THR = 0.0    # SAM2's native occlusion head (logit > 0 = present)
    ABS_IOU_THR   = 0.50   # predicted IoU below this = suspicious
    APP_SIM_THR   = 0.45   # appearance similarity below this = possible distractor
    MIN_PIX       = 10

    # -- Hysteresis ---------------------------------------------------------
    ABSENT_K  = 3          # consecutive bad frames -> ABSENT
    PRESENT_J = 2          # consecutive good frames -> PRESENT
    APP_EMA   = 0.05       # update rate of the appearance template (slow)

    # -- Appearance ---------------------------------------------------------
    HS_BINS = (16, 16)     # Hue-Saturation histogram bins

    # -- Bounded non-conditioning memory window (re-detection fix) ---------
    # Bisected on rabbit: the window must still contain the "object present"
    # anchors from before the absence for SAM2 to re-detect. 60 is too small
    # (emptiness 0.85), 100+ restores baseline level (0.51). 300 covers long
    # absences and stays bounded (keeping all frames OOM'd on uav1 at frame 1499).
    MEM_WINDOW = 300

    # =====================================================================
    @torch.inference_mode()
    def initialize(self, image, init_mask, bbox=None):
        out = super().initialize(image, init_mask, bbox=bbox)
        self.state = "PRESENT"
        self.absent_streak = 0
        self.present_streak = 0
        self.init_hist = None
        self.ref_hist = None
        if self.USE_APPEARANCE:
            rgb = np.asarray(image.convert("RGB"))
            init_m = out.get("pred_mask")
            if init_m is not None and int((init_m == 1).sum()) >= self.MIN_PIX:
                self.init_hist = self._hs_hist(rgb, init_m)
                self.ref_hist = self.init_hist
        return out

    # -- Normalised Hue-Saturation histogram over the masked region --------
    def _hs_hist(self, rgb, mask):
        ys, xs = np.where(mask > 0)
        if len(xs) < self.MIN_PIX:
            return None
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        m = (mask > 0).astype(np.uint8)
        hist = cv2.calcHist([hsv], [0, 1], m, list(self.HS_BINS), [0, 180, 0, 256])
        cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
        return hist.flatten().astype(np.float32)

    def _is_good_declaration(self, n_pix, obj_logit, m_iou, app_sim):
        """Criteria for NOT incrementing absent_streak (PRESENT branch). Overridden in v6."""
        return (
            n_pix >= self.MIN_PIX
            and obj_logit > self.OBJ_LOGIT_THR
            and m_iou > self.ABS_IOU_THR
            and (not self.USE_APPEARANCE or app_sim >= self.APP_SIM_THR)
        )

    def _is_good_redetection(self, n_pix, obj_logit, m_iou, app_sim):
        """Criterii pt a incrementa present_streak (ABSENT branch). Override in v6."""
        return self._is_good_declaration(n_pix, obj_logit, m_iou, app_sim)

    def _app_sim(self, h):
        """Corelatie histograma curenta vs template de referinta (0..1)."""
        if h is None or self.ref_hist is None:
            return 0.0
        a = self.ref_hist.reshape(-1, 1)
        b = h.reshape(-1, 1)
        s = cv2.compareHist(a, b, cv2.HISTCMP_CORREL)
        return float(max(0.0, s))

    def _obj_logit(self):
        try:
            t = self.predictor.curr_out["object_score_logits"]
            return float(np.asarray(t.flatten()[0].cpu()))
        except Exception:
            return 10.0  # if unavailable, assume present (as SAM2 does)

    # =====================================================================
    @torch.inference_mode()
    def track(self, image, init=False):
        prepared_img = self._prepare_image(image).unsqueeze(0)
        if not init:
            self.frame_index += 1
            self.inference_state["num_frames"] += 1
        self.inference_state["images"][self.frame_index] = prepared_img
        rgb = np.asarray(image.convert("RGB")) if self.USE_APPEARANCE else None

        for out in self.predictor.propagate_in_video(
            self.inference_state,
            start_frame_idx=self.frame_index,
            max_frame_num_to_track=0,
            return_all_masks=True,
        ):
            do_drm = True
            if len(out) == 3:
                out_frame_idx, _, out_mask_logits = out
                m = (out_mask_logits[0][0] > 0.0).float().cpu().numpy().astype(np.uint8)
            else:
                out_frame_idx, _, out_mask_logits, alternative_masks_ious = out
                all_masks, out_all_ious = alternative_masks_ious

                # -- Mask selection (as in v3: SAM argmax + filter tiebreak) -
                ious = np.asarray(out_all_ious, dtype=np.float64)
                m_idx = int(np.argmax(ious))
                kf_healthy = (
                    self.USE_MOTION and self.kf is not None
                    and not self._need_reinit and self._lost_streak == 0
                )
                if kf_healthy and len(all_masks) > 1:
                    srt = np.sort(ious)[::-1]
                    if (srt[0] - srt[1]) <= self.TIE_MARGIN:
                        pred_bbox = self.kf.predict()
                        best, best_score = m_idx, -1.0
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
                n_pix = int((m == 1).sum())

                # ── Semnale prezenta ─────────────────────────────────────
                obj_logit = self._obj_logit()
                app_sim = 1.0
                cur_hist = None
                if self.USE_APPEARANCE and n_pix >= self.MIN_PIX:
                    cur_hist = self._hs_hist(rgb, m)
                    app_sim = self._app_sim(cur_hist)

                # -- Presence/absence state machine -------------------------
                if self.USE_ABSENCE:
                    if self.state == "PRESENT":
                        if self._is_good_declaration(n_pix, obj_logit, m_iou, app_sim):
                            self.absent_streak = 0
                            if cur_hist is not None:
                                self.ref_hist = (1 - self.APP_EMA) * self.ref_hist + self.APP_EMA * cur_hist
                        else:
                            self.absent_streak += 1
                            if self.absent_streak >= self.ABSENT_K:
                                self.state = "ABSENT"
                                self.present_streak = 0
                                self._need_reinit = True  # filter unsafe after a loss
                    else:  # ABSENT
                        if self._is_good_redetection(n_pix, obj_logit, m_iou, app_sim):
                            self.present_streak += 1
                            if self.present_streak >= self.PRESENT_J:
                                self.state = "PRESENT"
                                self.absent_streak = 0
                        else:
                            self.present_streak = 0

                in_absent = self.USE_ABSENCE and self.state == "ABSENT"
                do_drm = not in_absent  # memory hygiene: no additions while absent

                # -- Filter lifecycle (as in v3, suspended while absent) ----
                if self.USE_MOTION and not in_absent:
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

                # ── Logica DRM (gardata de starea PRESENT) ───────────────
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
                    do_drm
                    and m_iou > 0.8
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

                # -- While ABSENT: emit an empty mask -----------------------
                if in_absent:
                    m = np.zeros_like(m)

            # ── Postproc conditionat (din v3) ────────────────────────────
            m = self._conditional_postproc(m)

            self.inference_state["images"].pop(self.frame_index)

            # ── Pruning non-cond frame outputs (fereastra glisanta) ──────
            self._prune_memory()

            return {"pred_mask": m}

    # -- Non-conditioning memory pruning (factored out for v5's adaptive override)
    # Keep the last MEM_WINDOW frames (SAM2 needs ~35-100 for re-detection) and
    # drop only what is older: an OOM fix that does not destroy the temporal
    # memory re-acquisition after an absence depends on.
    def _prune_memory(self):
        cutoff = self.frame_index - self.MEM_WINDOW
        self._prune_below(cutoff)

    def _prune_below(self, cutoff):
        output_dict = self.inference_state["output_dict"]
        for f in list(output_dict["non_cond_frame_outputs"].keys()):
            if f < cutoff:
                del output_dict["non_cond_frame_outputs"][f]
                for obj_state in self.inference_state["output_dict_per_obj"].values():
                    obj_state["non_cond_frame_outputs"].pop(f, None)
                self.inference_state["consolidated_frame_inds"]["non_cond_frame_outputs"].discard(f)
