#!/usr/bin/env python3
"""The speed gate of DAM4SAMTrackerV3g cannot open: a numerical demonstration.

docs/ABLATION.md attributes the provable inertness of the motion pathway to a
deadlock among three facts. The byte-level trajectory comparison
(scripts/compare_trajectories.py) shows *that* the pathway never ran; this
script shows *why*, on the filter itself, with no GPU and no dataset:

  1. The gate tests the filter's velocity estimate, which starts at zero.
  2. Only predict() can move it. The initial covariance is block-diagonal, so
     the Kalman gain's velocity rows are identically zero and update() cannot
     touch the velocity components; position-velocity correlation appears only
     through P <- F P F^T + Q inside predict().
  3. predict() is called at exactly one site -- inside the branch the gate
     guards (dam4sam/dam4sam_tracker_v3.py, in the motion tiebreaker).

So the gate needs a velocity that only opening the gate can produce.

Run: python scripts/test_gate_deadlock.py
Exit code 0 = the deadlock reproduces as documented.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "dam4sam"))
from motion import KalmanBoxTracker  # noqa: E402

VEL_RATIO_THR = 0.03   # DAM4SAMTrackerV3g.VEL_RATIO_THR
STEPS, SPEED, W, H = 15, 8.0, 100.0, 100.0


def speed_ratio(kf):
    """DAM4SAMTrackerV3g._motion_gate_open, verbatim."""
    vx, vy = float(kf.x[4, 0]), float(kf.x[5, 0])
    w, h = float(kf.x[2, 0]), float(kf.x[3, 0])
    scale = max(1.0, (max(w, 1.0) * max(h, 1.0)) ** 0.5)
    return (vx * vx + vy * vy) ** 0.5 / scale


def drive(call_predict):
    """Feed the filter a target moving at SPEED px/frame along x."""
    kf = KalmanBoxTracker(np.array([100.0, 100.0, W, H]))
    for i in range(1, STEPS + 1):
        if call_predict:
            kf.predict()
        kf.update(np.array([100.0 + SPEED * i, 100.0, W, H]))
    return kf


def main():
    ok = True

    # -- Fact 2a: update() alone cannot move the velocity ------------------
    kf = drive(call_predict=False)
    vx, rho = float(kf.x[4, 0]), speed_ratio(kf)
    print(f"update() only     : vx={vx:.10f}  rho={rho:.4f}  gate={rho >= VEL_RATIO_THR}")
    ok &= (vx == 0.0) and (rho < VEL_RATIO_THR)

    # -- Fact 2b: the full cycle recovers the true velocity and opens it ---
    kf = drive(call_predict=True)
    vx, rho = float(kf.x[4, 0]), speed_ratio(kf)
    print(f"predict()+update(): vx={vx:.10f}  rho={rho:.4f}  gate={rho >= VEL_RATIO_THR}")
    ok &= abs(vx - SPEED) < 1e-3 and rho >= VEL_RATIO_THR

    # -- The structural reason: the gain cannot reach the velocity rows ----
    kf = KalmanBoxTracker(np.array([100.0, 100.0, W, H]))
    S = kf.H @ kf.P @ kf.H.T + kf.R
    K = kf.P @ kf.H.T @ np.linalg.inv(S)
    print(f"max |K[4:, :]|    : {np.abs(K[4:, :]).max():.1e}   "
          "(velocity rows of the Kalman gain)")
    ok &= np.abs(K[4:, :]).max() == 0.0

    print("\nDeadlock reproduces as documented." if ok else "\nMISMATCH.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
