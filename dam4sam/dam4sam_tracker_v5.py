"""
DAM4SAMTrackerV5 — v4 + MEMORIE ADAPTIVA (rezolva compromisul ferestrei).

Descoperire (22 secvente grele, metrica gradata):
  - memorie SCURTA (keep-1): fish2 -0.70, seagull -0.41 (uita aspectul stale) DAR
    rabbit +0.86 (fara ancora pre-absenta -> nu re-detecteaza)
  - memorie LUNGA (keep-300): rabbit reparat DAR fish2 pierde avantajul
  Sunt opuse pentru o fereastra FIXA.

V5 leaga fereastra de starea automatului de absenta din v4:
  - PRESENT : fereastra SCURTA (MEM_WINDOW_SHORT) -> uita aspectul stale,
              pastreaza beneficiul fish2/seagull.
  - ABSENT  : fereastra LUNGA (MEM_WINDOW_LONG) -> NU prune-uieste ancorele
              pre-absenta (ultimele cadre cu obiect prezent), ca SAM2 sa poata
              re-detecta cand obiectul revine (beneficiul rabbit).

La debutul absentei, cadrele recente (cu obiect, dinainte de disparitie) sunt
inca in fereastra scurta; trecerea la fereastra lunga le ingheata -> ancore
disponibile pe toata durata absentei.

Subclaseaza DAM4SAMTrackerV4; suprascrie DOAR _prune_memory().
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from dam4sam_tracker_v4 import DAM4SAMTrackerV4


class DAM4SAMTrackerV5(DAM4SAMTrackerV4):
    # fereastra cand obiectul e prezent (uitare rapida -> fish2/seagull)
    MEM_WINDOW_SHORT = 30
    # fereastra cand obiectul e absent (pastreaza ancorele -> re-detectie rabbit)
    MEM_WINDOW_LONG = 400

    def _prune_memory(self):
        absent = getattr(self, "state", "PRESENT") == "ABSENT"
        window = self.MEM_WINDOW_LONG if absent else self.MEM_WINDOW_SHORT
        self._prune_below(self.frame_index - window)
