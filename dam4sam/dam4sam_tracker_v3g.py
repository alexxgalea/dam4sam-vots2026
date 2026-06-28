"""
DAM4SAMTrackerV3g — V3 cu POARTA DE VITEZA (motion gated).

Motivatie (din calibrarea pe Q real, 09-Jun-2026):
  Pe VOST-val (Q real cu GT public) V3 e USOR MAI SLAB ca baseline (0.381 vs 0.392).
  Cauza: re-ranking-ul de motion intervine si pe obiecte LENTE/DEFORMABILE (manipulare
  in bucatarie), unde modelul Kalman constant-velocity prezice gresit (obiectul sta/se
  deformeaza). Tiebreaker-ul V3 se declanseaza cand SAM e ambiguu, dar NU verifica daca
  miscarea e informativa.

  Diagnosticul serverului VOTS2026 (baseline Q=0.69): pierderea principala e DRE (drift,
  0.09) >> NRE (ratari, 0.05). Deci tinta corecta e reducerea drift-ului pe miscare rapida,
  FARA sa atingem secventele lente unde baseline e deja bun.

Idee: pe langa conditia V3 (SAM ambiguu + KF sanatos), cerem si ca obiectul sa se miste
SUFICIENT DE REPEDE ca motion-ul sa fie informativ. Altfel poarta se inchide -> argmax(sam)
pur = comportament Base+.

Prag SCALE-INVARIANT: viteza centroidului raportata la dimensiunea obiectului.
  speed_ratio = |(vx, vy)| / sqrt(w*h)
  poarta deschisa daca speed_ratio >= VEL_RATIO_THR.
=> independent de rezolutie si de marimea obiectului. Pe obiecte lente (kitchen) ratio~0,
   poarta inchisa -> NU putem fi mai rai ca baseline acolo. Pe miscare rapida (uav, racing,
   seagull) ratio mare -> motion intervine si reduce drift-ul, ca in V3.

Prin constructie: V3g <= regresii(V3) si pastreaza castigurile V3 pe miscare rapida.
Subclaseaza V3 — schimba DOAR poarta. NU modifica dam4sam_tracker.py.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from dam4sam_tracker_v3 import DAM4SAMTrackerV3


class DAM4SAMTrackerV3g(DAM4SAMTrackerV3):
    # Prag viteza/dimensiune: obiectul trebuie sa se mute >= atat din propria
    # dimensiune pe cadru ca sa consideram motion-ul informativ. Tunabil.
    VEL_RATIO_THR = 0.03

    def _motion_gate_open(self):
        kf = self.kf
        if kf is None:
            return False
        vx = float(kf.x[4, 0]); vy = float(kf.x[5, 0])
        w = float(kf.x[2, 0]);  h = float(kf.x[3, 0])
        scale = max(1.0, (max(w, 1.0) * max(h, 1.0)) ** 0.5)
        speed_ratio = (vx * vx + vy * vy) ** 0.5 / scale
        return speed_ratio >= self.VEL_RATIO_THR
