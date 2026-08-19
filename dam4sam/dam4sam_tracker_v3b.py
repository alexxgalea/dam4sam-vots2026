"""
DAM4SAMTrackerV3b -- v3 with a tightened TIE_MARGIN (0.15 -> 0.05).

Problem with v3: at TIE_MARGIN=0.15 the filter intervenes too often on "easy"
sequences where SAM2 is already confident (top-2 gap between 0.05 and 0.15). One
wrong pick (babychimp, frame 622) is enough to start a drift -- a +0.409
catastrophic regression on the proxy metric.

Fix: TIE_MARGIN=0.05, so the filter is consulted ONLY on extremely tight ties.
On hard sequences (fish2, racing-16) SAM2 is confused and the gap is frequently
below 0.05, so the filter still helps. On easy sequences the gap stays above
0.05, the filter never intervenes, and there is nothing to drift.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from dam4sam_tracker_v3 import DAM4SAMTrackerV3


class DAM4SAMTrackerV3b(DAM4SAMTrackerV3):
    TIE_MARGIN = 0.05
