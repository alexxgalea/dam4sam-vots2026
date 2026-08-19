"""
DAM4SAMTrackerV5 -- v4 + ADAPTIVE MEMORY (resolves the window trade-off).

Finding (22 hard sequences, graded proxy metric):
  - SHORT memory (keep-1): fish2 -0.70, seagull -0.41 (stale appearance is
    forgotten) BUT rabbit +0.86 (no pre-absence anchor -> no re-detection)
  - LONG memory (keep-300): rabbit is fixed BUT fish2 loses its advantage
  For a FIXED window the two requirements are opposites.

V5 ties the window to v4's absence state machine:
  - PRESENT : SHORT window (MEM_WINDOW_SHORT) -> forget stale appearance, keep
              the fish2/seagull benefit.
  - ABSENT  : LONG window (MEM_WINDOW_LONG) -> do NOT prune the pre-absence
              anchors (the last frames with the object visible), so SAM2 can
              re-detect when the object returns (the rabbit benefit).

At the onset of an absence the recent frames (object still visible) are still
inside the short window; switching to the long window freezes them, so the
anchors stay available for the whole absence.

Subclasses DAM4SAMTrackerV4; overrides ONLY _prune_memory().
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from dam4sam_tracker_v4 import DAM4SAMTrackerV4


class DAM4SAMTrackerV5(DAM4SAMTrackerV4):
    # window while the object is present (fast forgetting -> fish2/seagull)
    MEM_WINDOW_SHORT = 30
    # window while the object is absent (keeps the anchors -> rabbit re-detection)
    MEM_WINDOW_LONG = 400

    def _prune_memory(self):
        absent = getattr(self, "state", "PRESENT") == "ABSENT"
        window = self.MEM_WINDOW_LONG if absent else self.MEM_WINDOW_SHORT
        self._prune_below(self.frame_index - window)
