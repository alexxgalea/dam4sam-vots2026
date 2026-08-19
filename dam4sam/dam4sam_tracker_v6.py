"""
DAM4SAMTrackerV6 -- conservative absence declaration + adaptive memory.

V4/V5 bug: APP_SIM_THR=0.45 and ABS_IOU_THR=0.50 took part in DECLARING an
absence, which produced false absences on objects that change appearance or are
small (bull: v3=0.10 -> v4=0.32; wubble_bubble: v3=0.28 -> v4=0.41). Net dR for
v4 was +0.050 over 13/22 sequences -- worse than the baseline.

V6 fix:
  1. Absence is declared from SAM2's own occlusion head ALONE (obj_logit < 0 for
     ABSENT_K consecutive frames). The appearance signal takes no part in it.
  2. Re-detection is confirmed using appearance (APP_REDET_THR=0.30, more
     permissive) to avoid recovering onto a distractor.
  3. Adaptive memory (inherited from v5):
       PRESENT -> short window (30 frames): forget stale appearance (fish2/seagull)
       ABSENT  -> long window (400 frames): keep the anchors (rabbit)

Subclasses DAM4SAMTrackerV4; overrides _is_good_declaration, _is_good_redetection
and _prune_memory. dam4sam_tracker.py is never modified.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from dam4sam_tracker_v4 import DAM4SAMTrackerV4


class DAM4SAMTrackerV6(DAM4SAMTrackerV4):
    # -- Absence declaration: CONSERVATIVE -------------------------------
    # obj_logit only (SAM2's native signal); no APP_SIM_THR, no ABS_IOU_THR.
    # ABSENT_K raised to 5 (from 3 in v4) for extra hysteresis.
    ABSENT_K = 5

    # -- Re-detection: permissive appearance check ------------------------
    # Lower than v4's APP_SIM_THR (0.45) so real reappearances are not missed.
    APP_REDET_THR = 0.30

    # -- Adaptive memory (logic from v5) ----------------------------------
    MEM_WINDOW_SHORT = 30   # PRESENT: forget stale appearance (fish2/seagull)
    MEM_WINDOW_LONG  = 400  # ABSENT:  keep the pre-absence anchors (rabbit)

    # -- Absence declaration: obj_logit ONLY -------------------------------
    def _is_good_declaration(self, n_pix, obj_logit, m_iou, app_sim):
        return (
            n_pix >= self.MIN_PIX
            and obj_logit > self.OBJ_LOGIT_THR
            # deliberate: no ABS_IOU_THR, no APP_SIM_THR
        )

    # -- Re-detection: obj_logit + permissive appearance -------------------
    def _is_good_redetection(self, n_pix, obj_logit, m_iou, app_sim):
        return (
            n_pix >= self.MIN_PIX
            and obj_logit > self.OBJ_LOGIT_THR
            and (not self.USE_APPEARANCE or app_sim >= self.APP_REDET_THR)
        )

    # -- Adaptive memory ---------------------------------------------------
    def _prune_memory(self):
        absent = getattr(self, "state", "PRESENT") == "ABSENT"
        window = self.MEM_WINDOW_LONG if absent else self.MEM_WINDOW_SHORT
        self._prune_below(self.frame_index - window)
