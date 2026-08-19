"""
DAM4SAMTrackerV3g -- V3 with a SPEED GATE on the motion tiebreaker.

Motivation (from the calibration against real Q, June 2026):
  On VOST-val (real Q, public ground truth) V3 is slightly WORSE than the baseline
  (0.381 vs 0.392). Cause: motion re-ranking also fires on SLOW / DEFORMABLE objects
  (kitchen manipulation), where a constant-velocity Kalman model mispredicts (the
  object is stationary or deforming). V3's tiebreaker triggers when SAM is ambiguous
  but never checks whether the motion is informative at all.

  Server diagnosis on VOTS2026 (baseline Q=0.69): the dominant loss is DRE (drift,
  0.09) >> NRE (missed detections, 0.05). The right target is therefore drift under
  fast motion, WITHOUT touching the slow sequences where the baseline is already good.

Idea: on top of V3's condition (SAM ambiguous + filter healthy), also require the
object to move FAST ENOUGH for motion to be informative. Otherwise the gate closes
and selection falls back to plain argmax(sam) = Base+ behaviour.

SCALE-INVARIANT threshold: centroid speed relative to object size.
  speed_ratio = |(vx, vy)| / sqrt(w*h)
  the gate opens if speed_ratio >= VEL_RATIO_THR.
=> independent of resolution and object size. On slow objects (kitchen) the ratio is
   ~0 and the gate stays shut, so we cannot be worse than the baseline there. On fast
   motion (uav, racing, seagull) the ratio is large, motion intervenes and reduces
   drift, as in V3.

By construction: V3g <= regressions(V3), while keeping V3's gains on fast motion.
Subclasses V3 -- ONLY the gate changes. dam4sam_tracker.py is not modified.

POST-CHALLENGE FINDING (docs/ABLATION.md): this gate never opens. The velocity it
tests can only become non-zero through the filter's predict() step, and predict() is
called only inside the branch the gate guards -- a deadlock. Removing the entire
motion pathway leaves the output masks byte-identical on all 375 target-sequences,
so the improvement measured for this tracker belongs to V3's bounded memory window,
not to the gate. Forcing the gate open (ko_speed) costs -0.010 Q.
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from dam4sam_tracker_v3 import DAM4SAMTrackerV3


class DAM4SAMTrackerV3g(DAM4SAMTrackerV3):
    # Speed/size threshold: the object must move at least this fraction of its
    # own size per frame before motion is considered informative.
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
