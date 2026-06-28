"""
DAM4SAMTrackerV2 — DAM4SAM + SAM2.1-B+ cu doua imbunatatiri training-free:

1. Selectie motion-aware (SAMURAI-style): filtru Kalman re-ordoneaza mastile
   candidate SAM2 dupa consistenta de miscare, in loc de pur IoU SAM.
   Tinteste miscarea rapida si distractorii.

2. Post-procesare masca de output: keep_largest_component elimina blob-urile
   spurioase de distractori din masca finala.

Subclaseaza DAM4SAMTracker — NU modifica dam4sam_tracker.py.
Logica DRM ramane identica cu originalul (praguri din paper, neschimbate).
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import torch

from dam4sam_tracker import DAM4SAMTracker, DRM_MAX_FRAMES
from vot.region.shapes import Mask, Rectangle
from vot.region.raster import calculate_overlaps
from utils.utils import keep_largest_component
from motion import KalmanBoxTracker, mask_to_bbox, bbox_iou


class DAM4SAMTrackerV2(DAM4SAMTracker):
    """
    DAM4SAM + motion-aware mask selection + output post-processing.

    Toggles (atribute de clasa — schimba-le pentru ablatie):
      USE_MOTION   = True   re-ordoneaza mastile candidate dupa KF motion score
      MOTION_WEIGHT = 0.20  scor = (1-w)*sam_iou + w*motion_iou
      USE_POSTPROC = True   keep_largest_component pe masca de output
      KF_GATE_PIXELS = 1    actualizeaza KF doar daca obiectul e prezent
    """
    USE_MOTION    = True
    MOTION_WEIGHT = 0.20
    USE_POSTPROC  = True
    KF_GATE_PIXELS = 1

    # -- VRAM fix: offload frame outputs pe CPU (ca in vot_wrapper_large.py) --
    def init_state_tw(self):
        state = super().init_state_tw()
        state["storage_device"] = torch.device("cpu")
        return state

    # -- initialize: super() + init KF din masca de init --
    @torch.inference_mode()
    def initialize(self, image, init_mask, bbox=None):
        out = super().initialize(image, init_mask, bbox=bbox)
        self.kf = None
        if self.USE_MOTION:
            init_m = out.get("pred_mask")
            if init_m is not None:
                bb = mask_to_bbox(init_m)
                if bb is not None:
                    self.kf = KalmanBoxTracker(bb)
        return out

    # -- track: motion-aware selection + post-proc --
    @torch.inference_mode()
    def track(self, image, init=False):
        # Pregatire imagine (identic cu super)
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
                # Frame de initializare — nu avem alternative masks
                out_frame_idx, _, out_mask_logits = out
                m = (out_mask_logits[0][0] > 0.0).float().cpu().numpy().astype(np.uint8)

            else:
                out_frame_idx, _, out_mask_logits, alternative_masks_ious = out
                all_masks, out_all_ious = alternative_masks_ious

                # ── Selectie mask: motion-aware sau original ──────────────────
                if self.USE_MOTION and self.kf is not None and len(all_masks) > 0:
                    pred_bbox = self.kf.predict()

                    combined_scores = []
                    for i, (cand_logits, sam_iou) in enumerate(zip(all_masks, out_all_ious)):
                        cand_bin = (cand_logits[0][0] > 0.0).cpu().numpy().astype(np.uint8)
                        cand_bbox = mask_to_bbox(cand_bin)
                        mot_iou = bbox_iou(cand_bbox, pred_bbox)
                        score = (1.0 - self.MOTION_WEIGHT) * sam_iou + self.MOTION_WEIGHT * mot_iou
                        combined_scores.append(score)

                    m_idx = int(np.argmax(combined_scores))
                else:
                    m_idx = int(np.argmax(out_all_ious))

                m_iou = float(out_all_ious[m_idx])
                chosen_logits = all_masks[m_idx]
                m = (chosen_logits[0][0] > 0.0).float().cpu().numpy().astype(np.uint8)

                # ── Aggiorna KF cu bbox-ul ales ───────────────────────────────
                if self.USE_MOTION and self.kf is not None:
                    n_pix = int((m == 1).sum())
                    if n_pix >= self.KF_GATE_PIXELS:
                        bb = mask_to_bbox(m)
                        if bb is not None:
                            self.kf.update(bb)

                # ── Logica DRM (identica cu originalul) ──────────────────────
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
                    chosen_bbox_r  = Rectangle.convert(Mask(chosen_mask_np))

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
                        ious = [calculate_overlaps([chosen_bbox_r], [b])[0] for b in alt_bboxes]

                        if np.min(np.array(ious)) <= 0.7:
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

            # ── Post-procesare masca de output ────────────────────────────────
            if self.USE_POSTPROC and int((m == 1).sum()) > 0:
                m = keep_largest_component(m)

            self.inference_state["images"].pop(self.frame_index)

            # ── Pruning non-cond frame outputs (fix OOM pe secvente lungi) ────
            # Fara pruning, output-urile non-cond se acumuleaza frame cu frame
            # pana la OOM (chiar si pe CPU, SAM2 le muta pe GPU la fiecare inferenta).
            # Acelasi fix ca in dam4sam_tracker_sam3.py.
            output_dict = self.inference_state["output_dict"]
            for f in list(output_dict["non_cond_frame_outputs"].keys()):
                if f < self.frame_index:
                    del output_dict["non_cond_frame_outputs"][f]
                    for obj_state in self.inference_state["output_dict_per_obj"].values():
                        obj_state["non_cond_frame_outputs"].pop(f, None)
                    self.inference_state["consolidated_frame_inds"]["non_cond_frame_outputs"].discard(f)

            return {"pred_mask": m}
