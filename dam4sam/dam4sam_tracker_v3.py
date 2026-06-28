"""
DAM4SAMTrackerV3 — corectarea chirurgicala a V2.

Diagnostic V2 (pe setul greu, metrica empty_rate): interventie cu varianta mare,
net ~0. Castiga mare (seagull -0.33, fish2 -0.20, tank-16 -0.11) DAR pierde mare
(rabbit +0.36, bull +0.30, uav1 +0.16). Semnatura pierderilor = filtru Kalman care
ramane pe viteza veche dupa o pierdere si se blocheaza pe un distractor.

V3 pastreaza castigurile si taie pierderile prin 3 schimbari:

  1. MOTION CA TIEBREAKER, NU BLEND GLOBAL.
     Default ramane argmax(sam_iou) — identic cu Base+. Consultam Kalman DOAR cand
     primele 2 IoU-uri SAM sunt apropiate (<= TIE_MARGIN) si KF e "sanatos".
     => pe cadrele neambigue ne purtam EXACT ca Base+ ⇒ nu putem pierde mare.

  2. CICLU DE VIATA KF CORECT.
     - actualizam KF doar cand masca aleasa e increzatoare (IoU>UPDATE_IOU,
       dimensiune stabila, >= KF_GATE_PIXELS) ⇒ nu otravim viteza cu gunoi.
     - daca pierdem obiectul MAX_COAST cadre la rand ⇒ KF devine nesigur,
       il reinitializam la urmatoarea re-achizitie buna in loc sa-l "coasting"-uim.

  3. POSTPROC CONDITIONAT.
     keep_largest_component DOAR cand a 2-a componenta < SECOND_FRAC * prima
     (blob spurios clar). Altfel pastram masca intreaga — nu taiem obiecte care
     se despart legitim (pescarus in zbor, iepure partial ocludat).

Subclaseaza DAM4SAMTracker — NU modifica dam4sam_tracker.py. Logica DRM identica.
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
    # ── Toggles principale ──────────────────────────────────────────────
    USE_MOTION   = True
    USE_POSTPROC = True

    # ── Motion ca tiebreaker ────────────────────────────────────────────
    TIE_MARGIN   = 0.15   # consult KF doar daca top1-top2 IoU SAM <= asta
    MOTION_WEIGHT = 0.50  # in zona ambigua: (1-w)*sam + w*motion

    # ── Ciclu de viata KF ───────────────────────────────────────────────
    UPDATE_IOU     = 0.70  # actualizeaza KF doar peste acest IoU prezis
    KF_GATE_PIXELS = 10    # ... si peste atatia pixeli
    SIZE_LO, SIZE_HI = 0.5, 2.0  # ... si daca aria ramane in [0.5x, 2x] mediana
    MAX_COAST      = 5     # dupa atatea cadre pierdute la rand -> reinit KF

    # ── Postproc conditionat ────────────────────────────────────────────
    SECOND_FRAC = 0.30     # taie doar daca comp#2 < 0.30 * comp#1

    # ── Fereastra memorie non-cond pastrata (fix re-detectie + OOM) ──────
    # Bisectat pe rabbit: 60 prea mic (re-detectie esueaza dupa absenta lunga),
    # 100+ restaureaza re-detectia la nivel base. 300 acopera absente lungi dar
    # ramane marginit -> OOM-safe pe secvente de mii de cadre (keep-all a dat OOM).
    MEM_WINDOW = 300

    # -- VRAM fix: offload frame outputs pe CPU --
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

    # ── Postproc conditionat: taie doar blob-uri spurioase clare ─────────
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
            # blob spurios clar -> pastram doar componenta dominanta
            return (labeled == (order[0] + 1)).astype(np.uint8)
        return m  # despartire legitima -> pastram intreg

    def _motion_gate_open(self):
        """Poarta tiebreaker-ului de motion. V3: mereu deschisa (motion intervine
        ori de cate ori SAM e ambiguu + KF sanatos). Subclasele (v3g) o pot inchide
        cand miscarea e prea lenta ca sa fie informativa -> cad inapoi pe Base+."""
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

                # ── Selectie mask: default SAM argmax (= Base+) ──────────
                ious = np.asarray(out_all_ious, dtype=np.float64)
                m_idx = int(np.argmax(ious))

                # ── Tiebreaker motion DOAR in zona ambigua + KF sanatos ──
                kf_healthy = (
                    self.USE_MOTION and self.kf is not None
                    and not self._need_reinit and self._lost_streak == 0
                )
                if kf_healthy and len(all_masks) > 1:
                    srt = np.sort(ious)[::-1]
                    if (srt[0] - srt[1]) <= self.TIE_MARGIN and self._motion_gate_open():
                        pred_bbox = self.kf.predict()
                        best, best_score = m_idx, -1.0
                        # doar candidatii apropiati de varf intra in disputa
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

                # ── Ciclu de viata KF ────────────────────────────────────
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

                # ── Logica DRM (identica cu originalul) ──────────────────
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

            # ── Postproc conditionat pe masca de output ──────────────────
            m = self._conditional_postproc(m)

            self.inference_state["images"].pop(self.frame_index)

            # ── Pruning non-cond frame outputs (fereastra glisanta) ──────
            # Pastram ultimele MEM_WINDOW cadre: SAM2 are nevoie de num_maskmem(7)
            # x stride(5) ~= 35 cadre inapoi pentru atentie temporala / re-detectie.
            # Keep-1 vechi distrugea reacapararea dupa absenta (rabbit ramanea mort).
            output_dict = self.inference_state["output_dict"]
            cutoff = self.frame_index - self.MEM_WINDOW
            for f in list(output_dict["non_cond_frame_outputs"].keys()):
                if f < cutoff:
                    del output_dict["non_cond_frame_outputs"][f]
                    for obj_state in self.inference_state["output_dict_per_obj"].values():
                        obj_state["non_cond_frame_outputs"].pop(f, None)
                    self.inference_state["consolidated_frame_inds"]["non_cond_frame_outputs"].discard(f)

            return {"pred_mask": m}
