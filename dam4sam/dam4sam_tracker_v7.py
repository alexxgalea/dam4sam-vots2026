"""
DAM4SAMTrackerV7 -- an appearance/identity channel independent of memory.

Two modes. With PROBE=False and USE_D2=False the class is bit-identical to V3g
(nothing is installed). PROBE=True logs per-candidate identity signals for every
frame without touching selection; USE_D2=True turns those signals into an
admission rule for DAM4SAM's distractor-resolving memory (see below).

Signals extracted per candidate mask (M=3):
  1. Per-candidate obj_ptr: a read-only forward hook on sam_mask_decoder captures
     sam_output_tokens [b,3,c] (use_multimask_token_for_obj_ptr=true in the
     config), to which predictor.obj_ptr_proj is then applied here. Note that the
     decoder receives a memory-conditioned pix_feat, so these tokens inherit part
     of that conditioning.
  2. Mask pooling over the RAW backbone_fpn stride-16 features (cached_features),
     a channel fully independent of memory -- it never passes through memory
     attention.

Reference prototypes:
  - P0  = the initialization frame (immutable). Caveat: the init obj_ptr comes
    from the single-mask token (slice [0:1]), a different subspace than the
    multimask tokens, which is why a second reference is kept:
  - LAST = the most recent confidently selected frame (m_iou > 0.8), a proxy for
    an adaptive prototype bank.

Subclasses V3g. dam4sam_tracker.py and sam2/ are not modified.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from dam4sam_tracker_v3g import DAM4SAMTrackerV3g


class DAM4SAMTrackerV7(DAM4SAMTrackerV3g):
    PROBE = False          # True: log identity signals (no behavioural change)
    CONF_IOU = 0.8         # "confidently selected frame" threshold for the LAST reference

    # -- D2: memory hygiene (admission discipline) -------------------------
    # Calibrated on a LaSOT drift subset: the identity score of the selected mask
    # is p25=0.967 / p50=0.99 on clean frames vs p50=0.586 during drift. A 0.7
    # threshold keeps 98% of the clean learning and blocks 79% of the learning
    # poisoned by drift.
    USE_D2 = False
    TAU_ADMIT_DRM = 0.7    # add_to_drm only if id_score(selected mask) >= this and not suspect
    # Memory dropping is DISABLED by default: on drone-13 it starved memory and
    # made things worse (drift fraction 0.055 vs 0.025 without it). DRM gating on
    # its own beat v3g (4 drift events vs 5).
    TAU_DROP_MEM = -1.0    # >0 would drop suspect low-identity frames from non-cond memory
    TAU_BANK = 0.85        # prototype admission into the bank (strictly > TAU_ADMIT_DRM)
    LEASH_TAU = 0.45       # leash to P0: a new prototype must resemble init at least this much
    BANK_CAP = 5           # P0 + at most 4 FIFO prototypes
    BANK_GAP = 30          # minimum number of frames between admissions
    # online switch detector (high-recall configuration; the cost of a false
    # suspect is only "do not learn from this frame", which is nearly free)
    SD_JUMP, SD_AREA, SD_HOLD, SD_AMIN = 0.8, 2.0, 8, 25.0

    # ------------------------------------------------------------------ init
    @torch.inference_mode()
    def initialize(self, image, init_mask, bbox=None):
        out = super().initialize(image, init_mask, bbox=bbox)
        if not (self.PROBE or self.USE_D2):
            return out

        self._install_decoder_hook()
        self._cap = None
        self._probe_records = []
        self._probe_ref_mask = None    # pseudo-GT set by the runner before track()

        # P0 obj_ptr: the init frame's pointer from the output dict (post-projection)
        try:
            p0 = self.inference_state["output_dict"]["cond_frame_outputs"][0]["obj_ptr"]
            self._p0_ptr = F.normalize(p0.flatten().float(), dim=0)
        except Exception:
            self._p0_ptr = None

        # P0 hiera: mask pooling over frame 0's backbone_fpn stride-16 features
        # (with box init, init_mask is None -> use the estimated mask returned)
        m0 = init_mask if init_mask is not None else out.get("pred_mask")
        feat0 = self._raw_feat(0)
        self._p0_hiera = (self._pool(feat0, m0)
                          if feat0 is not None and m0 is not None else None)

        self._last_ptr = None
        self._last_hiera = None

        # -- D2 state ------------------------------------------------------
        # Bank = immutable P0 + a rolling "last-confident" slot + admitted prototypes.
        # The rolling slot is CRITICAL: the thresholds (0.7/0.85) were calibrated on
        # the cosine-vs-last-confident distribution. Without it everything is measured
        # against P0 and learning is choked (drone-13: bank_admits=0, 73% of DRM
        # additions blocked, 277 memory drops, 9 drift events).
        self._bank = [self._p0_hiera] if self._p0_hiera is not None else []
        self._last_slot = None
        self._bank_last_admit = -self.BANK_GAP
        self._mem_drop_streak = 0
        self._sd_hold = 0
        self._sd_prev_c = None     # previous output centroid
        self._sd_prev_a = None     # previous output area
        self._suspect = False
        self._d2_stats = {"drm_blocked": 0, "drm_allowed": 0, "mem_dropped": 0,
                          "bank_admits": 0}
        if self.USE_D2:
            self._install_drm_guard()
        return out

    # ---------------------------------------------------------------- D2
    def _install_drm_guard(self):
        pred = self.predictor
        orig = getattr(pred, "_v7_drm_orig", None) or pred.add_to_drm
        pred._v7_drm_orig = orig
        tracker = self

        def guarded(inference_state=None, frame_idx=None, obj_id=None, **kw):
            if tracker._d2_admit_ok():
                tracker._d2_stats["drm_allowed"] += 1
                return orig(inference_state=inference_state,
                            frame_idx=frame_idx, obj_id=obj_id, **kw)
            tracker._d2_stats["drm_blocked"] += 1
            return None

        pred.add_to_drm = guarded

    def _id_score(self, emb):
        refs = [p for p in self._bank if p is not None]
        if self._last_slot is not None:
            refs.append(self._last_slot)
        if emb is None or not refs:
            return float("nan")
        return max(float(torch.dot(emb, p)) for p in refs)

    def _d2_admit_ok(self):
        """Called mid-way through super().track() by the DRM guard, on the current frame.
        Uses the decoder capture (already set by propagate) plus the switch
        detector state up to the previous frame (its hold covers continuity)."""
        if self._suspect:
            return False
        if self._cap is None or not self._bank:
            return True   # no signal -> baseline behaviour
        masks_lr, ious_t, _, _ = self._cap
        ious = ious_t[0].float().cpu().numpy()
        ci = int(np.argmax(ious))
        cb = (masks_lr[0, ci] > 0).float().cpu().numpy().astype(np.uint8)
        feat = self._raw_feat(self.frame_index)
        emb = self._pool(feat, cb) if feat is not None else None
        s = self._id_score(emb)
        return not (s == s and s < self.TAU_ADMIT_DRM)   # nan -> allow

    def _d2_post_frame(self, out_mask):
        """After super().track(): update the switch detector, the prototype bank
        and non-conditioning memory hygiene for the frame just processed."""
        t = self.frame_index
        n_pix = int((out_mask == 1).sum())
        if n_pix > 0:
            ys, xs = np.nonzero(out_mask)
            c = np.array([xs.mean(), ys.mean()]); a = float(n_pix)
        else:
            c, a = None, None

        fired = False
        if c is not None and self._sd_prev_c is not None:
            a_prev = max(self._sd_prev_a or self.SD_AMIN, self.SD_AMIN)
            if np.linalg.norm(c - self._sd_prev_c) / np.sqrt(a_prev) > self.SD_JUMP:
                fired = True
            lo = max(min(a, self._sd_prev_a or a), self.SD_AMIN)
            hi = max(a, self._sd_prev_a or a, self.SD_AMIN)
            if hi / lo > self.SD_AREA:
                fired = True
        self._sd_hold = self.SD_HOLD if fired else max(self._sd_hold - 1, 0)
        self._suspect = self._sd_hold > 0
        if c is not None:
            self._sd_prev_c, self._sd_prev_a = c, a

        # embedding of the output mask (for the bank + memory hygiene)
        feat = self._raw_feat(t)
        emb = self._pool(feat, out_mask) if feat is not None and n_pix > 0 else None
        s = self._id_score(emb)

        # hygiene: suspect + low identity -> do not keep the frame in non-cond memory.
        # Consecutive drops are capped so memory is never starved (the rabbit/v6 lesson).
        if (self._suspect and s == s and s < self.TAU_DROP_MEM
                and self._mem_drop_streak < 3):
            od = self.inference_state["output_dict"]
            if t in od["non_cond_frame_outputs"]:
                del od["non_cond_frame_outputs"][t]
                for obj_state in self.inference_state["output_dict_per_obj"].values():
                    obj_state["non_cond_frame_outputs"].pop(t, None)
                self.inference_state["consolidated_frame_inds"][
                    "non_cond_frame_outputs"].discard(t)
                self._d2_stats["mem_dropped"] += 1
                self._mem_drop_streak += 1
        else:
            self._mem_drop_streak = 0

        # rolling "last-confident" slot + strict admission into the bank
        if emb is not None and self._cap is not None:
            ious = self._cap[1][0].float().cpu().numpy()
            srt = np.sort(ious)[::-1]
            tie_closed = len(srt) < 2 or (srt[0] - srt[1]) > self.TIE_MARGIN
            obj_ok = float(self._cap[3].flatten()[0]) > 0
            confident = float(srt[0]) > self.CONF_IOU and tie_closed and obj_ok

            # the rolling slot follows the probe recipe (iou > 0.8) + not suspect;
            # it keeps the id_score distribution aligned with the offline calibration
            if confident and not self._suspect:
                self._last_slot = emb

            # PERSISTENT prototype: additionally leashed to P0 + gap + strict threshold
            if (confident and not self._suspect
                    and t - self._bank_last_admit >= self.BANK_GAP
                    and self._p0_hiera is not None
                    and self._id_score(emb) >= self.TAU_BANK
                    and float(torch.dot(emb, self._p0_hiera)) >= self.LEASH_TAU):
                self._bank.append(emb)
                if len(self._bank) > self.BANK_CAP:
                    self._bank.pop(1)   # FIFO over the non-P0 slots; P0 is immutable
                self._bank_last_admit = t
                self._d2_stats["bank_admits"] += 1

    # -------------------------------------------------------------- plumbing
    def _install_decoder_hook(self):
        pred = self.predictor
        old = getattr(pred, "_v7_hook_handle", None)
        if old is not None:
            old.remove()

        def _hook(module, args, kwargs, output):
            masks, ious, tokens, obj_logits = output
            if tokens.shape[1] > 1:   # multimask calls only (tracking)
                self._cap = (masks.detach(), ious.detach(),
                             tokens.detach(), obj_logits.detach())

        pred._v7_hook_handle = pred.sam_mask_decoder.register_forward_hook(
            _hook, with_kwargs=True)

    def _raw_feat(self, frame_idx):
        """RAW backbone_fpn stride-16 features (not memory-conditioned) from cache."""
        cached = self.inference_state.get("cached_features", {})
        entry = cached.get(frame_idx)
        if entry is None:
            return None
        return entry[1]["backbone_fpn"][-1]   # (1, C, H, W)

    def _pool(self, feat, mask_np):
        """Masked mean-pool + L2-norm. mask_np la orice rezolutie (resized aici)."""
        _, C, H, W = feat.shape
        w = cv2.resize(mask_np.astype(np.float32), (W, H), interpolation=cv2.INTER_AREA)
        ws = float(w.sum())
        if ws < 1e-6:
            return None
        wt = torch.from_numpy(w).to(device=feat.device, dtype=feat.dtype)
        emb = (feat[0] * wt).sum(dim=(1, 2)) / ws
        return F.normalize(emb.float(), dim=0)

    @staticmethod
    def _cos(a, b):
        if a is None or b is None:
            return float("nan")
        return float(torch.dot(a, b))

    # ---------------------------------------------------------------- track
    @torch.inference_mode()
    def track(self, image, init=False):
        if not (self.PROBE or self.USE_D2):
            return super().track(image, init=init)

        self._cap = None
        out = super().track(image, init=init)
        if self.USE_D2:
            self._d2_post_frame(out["pred_mask"])
        if not self.PROBE or self._cap is None:
            return out

        masks_lr, ious_t, tokens, obj_logit = self._cap
        ious = ious_t[0].float().cpu().numpy()                       # (M,)
        M = tokens.shape[1]

        # canal 1: obj_ptr per-candidat
        ptrs = self.predictor.obj_ptr_proj(tokens[0].float())        # (M, C)
        ptrs = F.normalize(ptrs, dim=1)

        # canal 2: hiera-pool per-candidat pe feature RAW al cadrului curent
        feat = self._raw_feat(self.frame_index)
        cand_bins = [(masks_lr[0, i] > 0).float().cpu().numpy().astype(np.uint8)
                     for i in range(M)]
        hieras = [self._pool(feat, cb) if feat is not None else None
                  for cb in cand_bins]

        # bbox per candidat in coordonate IMAGINE originala (low-res 256 -> img)
        sx, sy = self.img_width / masks_lr.shape[-1], self.img_height / masks_lr.shape[-2]
        cand_boxes = []
        for cb in cand_bins:
            ys, xs = np.nonzero(cb)
            if len(xs) == 0:
                cand_boxes.append([np.nan] * 4)
            else:
                cand_boxes.append([xs.min() * sx, ys.min() * sy,
                                   (xs.max() - xs.min() + 1) * sx,
                                   (ys.max() - ys.min() + 1) * sy])

        rec = {
            "frame": self.frame_index,
            "ious": ious,
            "cand_boxes": cand_boxes,
            "obj_logit": float(obj_logit.flatten()[0]),
            "tie": float(np.sort(ious)[::-1][0] - np.sort(ious)[::-1][1]) <= self.TIE_MARGIN,
            "cos_ptr_p0":    [self._cos(ptrs[i], self._p0_ptr) for i in range(M)],
            "cos_ptr_last":  [self._cos(ptrs[i], self._last_ptr) for i in range(M)],
            "cos_hiera_p0":  [self._cos(h, self._p0_hiera) for h in hieras],
            "cos_hiera_last": [self._cos(h, self._last_hiera) for h in hieras],
        }

        # pseudo-GT (traiectoria baseline) — IoU per candidat in spatiul low-res
        ref = self._probe_ref_mask
        if ref is not None and ref.sum() > 0:
            S = masks_lr.shape[-1]
            ref_lr = cv2.resize(ref.astype(np.float32), (S, S),
                                interpolation=cv2.INTER_AREA) > 0.5
            rec["iou_ref"] = [
                float(np.logical_and(cb, ref_lr).sum() / max(np.logical_or(cb, ref_lr).sum(), 1))
                for cb in cand_bins
            ]
        else:
            rec["iou_ref"] = [float("nan")] * M

        # the index actually selected (after the v3g gate): candidate closest to the output
        out_m = out["pred_mask"]
        S = masks_lr.shape[-1]
        out_lr = cv2.resize(out_m.astype(np.float32), (S, S),
                            interpolation=cv2.INTER_AREA) > 0.5
        overlaps = [np.logical_and(cb, out_lr).sum() / max(np.logical_or(cb, out_lr).sum(), 1)
                    for cb in cand_bins]
        rec["chosen"] = int(np.argmax(overlaps))
        rec["sam_argmax"] = int(np.argmax(ious))
        self._probe_records.append(rec)

        # update the LAST reference only on confidently selected frames
        ci = rec["chosen"]
        if ious[ci] > self.CONF_IOU and cand_bins[ci].sum() > 0:
            self._last_ptr = ptrs[ci].clone()
            if hieras[ci] is not None:
                self._last_hiera = hieras[ci].clone()

        return out
