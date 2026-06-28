"""
DAM4SAMTrackerV3b — v3 cu TIE_MARGIN redus (0.15 -> 0.05).

Problema v3: TIE_MARGIN=0.15 face ca Kalman sa intervina prea des pe secventele
"usoare" unde SAM2 e deja sigur (gap top-2 intre 0.05-0.15). O singura alegere
gresita (babychimp frame 622) -> derivare -> +0.409 regresie catastrofala.

Fix: TIE_MARGIN=0.05 -> KF intervine NUMAI la tie-uri extrem de stranse.
Pe secvente grele (fish2, racing-16): SAM2 e confuz, gap < 0.05 des -> KF tot ajuta.
Pe secvente usoare: SAM2 e sigur, gap > 0.05 -> KF nu intervine -> fara derivare.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from dam4sam_tracker_v3 import DAM4SAMTrackerV3


class DAM4SAMTrackerV3b(DAM4SAMTrackerV3):
    TIE_MARGIN = 0.05
