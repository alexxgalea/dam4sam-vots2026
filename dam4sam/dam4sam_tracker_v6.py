"""
DAM4SAMTrackerV6 — absenta conservatoare + memorie adaptiva.

CONTEXT GREEDY: avem deadline 22 iunie si un pariu de 200EUR. Fiecare secventa
rulata pe directia gresita e timp pierdut. Principiu de operare:
  - daca vedem drapaj consistent (dR cumulativ pozitiv dupa 8+ secvente), OPRIM
    si diagnosticam inainte de a risipi ore de GPU.
  - valida pe secventele decisive (fish2/bull/rabbit/seagull) inainte de full-22.

BUG V4/V5: APP_SIM_THR=0.45 si ABS_IOU_THR=0.50 in gata de DECLARARE absenta
-> false absente pe obiecte care isi schimba aspectul sau sunt mici (bull: v3=0.10
-> v4=0.32; wubble_bubble: v3=0.28 -> v4=0.41). Net dR v4 = +0.050 pe 13/22 (mai
rau decat baseline).

FIX V6:
  1. Declarare absenta DOAR pe SAM2's own occlusion head (obj_logit<0 pentru
     ABSENT_K cadre consecutive). Semnalul de aspect NU intervine in declarare.
  2. Confirmarea re-detectiei foloseste aspect (APP_REDET_THR=0.30, mai permisiv)
     pentru a evita recuperarea pe distractori.
  3. Memorie adaptiva (mostenita din v5):
       PRESENT -> fereastra scurta (30 cadre): uita aspectul stale (fish2/seagull)
       ABSENT  -> fereastra lunga (400 cadre): pastreaza ancorele (rabbit)

Subclaseaza DAM4SAMTrackerV4; suprascrie _is_good_declaration, _is_good_redetection
si _prune_memory. Nu modifica dam4sam_tracker.py niciodata.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from dam4sam_tracker_v4 import DAM4SAMTrackerV4


class DAM4SAMTrackerV6(DAM4SAMTrackerV4):
    # ── Declarare absenta: CONSERVATOARE ────────────────────────────────
    # Doar obj_logit (semnalul nativ SAM2); fara APP_SIM_THR, fara ABS_IOU_THR.
    # ABSENT_K crescut la 5 (fata de 3 in v4) pentru histerezis suplimentar.
    ABSENT_K = 5

    # ── Re-detectie: aspect permisiv ────────────────────────────────────
    # Mai mic decat APP_SIM_THR din v4 (0.45) pt a nu rata re-aparitii reale.
    APP_REDET_THR = 0.30

    # ── Memorie adaptiva (logica din v5) ────────────────────────────────
    MEM_WINDOW_SHORT = 30   # PRESENT: uita aspectul stale (fish2/seagull)
    MEM_WINDOW_LONG  = 400  # ABSENT:  pastreaza ancorele pre-absenta (rabbit)

    # ── Declarare absenta: NUMAI obj_logit ──────────────────────────────
    def _is_good_declaration(self, n_pix, obj_logit, m_iou, app_sim):
        return (
            n_pix >= self.MIN_PIX
            and obj_logit > self.OBJ_LOGIT_THR
            # intentionat: fara ABS_IOU_THR, fara APP_SIM_THR
        )

    # ── Re-detectie: obj_logit + aspect permisiv ────────────────────────
    def _is_good_redetection(self, n_pix, obj_logit, m_iou, app_sim):
        return (
            n_pix >= self.MIN_PIX
            and obj_logit > self.OBJ_LOGIT_THR
            and (not self.USE_APPEARANCE or app_sim >= self.APP_REDET_THR)
        )

    # ── Memorie adaptiva ────────────────────────────────────────────────
    def _prune_memory(self):
        absent = getattr(self, "state", "PRESENT") == "ABSENT"
        window = self.MEM_WINDOW_LONG if absent else self.MEM_WINDOW_SHORT
        self._prune_below(self.frame_index - window)
