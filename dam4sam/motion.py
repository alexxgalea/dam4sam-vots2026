"""
Kalman filter constant-velocity tracker for bounding boxes.

Used by DAM4SAMTrackerV2 to re-rank SAM2 mask candidates by motion consistency
(SAMURAI-style motion-aware memory selection).

State: [cx, cy, w, h, vx, vy, vw, vh] — 8D constant-velocity model.
Observation: [cx, cy, w, h] — 4D bbox centroid + size.
"""
import numpy as np


def mask_to_bbox(mask):
    """Convert binary mask (H×W uint8) to (cx, cy, w, h). Returns None if empty."""
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    x1, x2 = xs.min(), xs.max()
    y1, y2 = ys.min(), ys.max()
    w = float(x2 - x1 + 1)
    h = float(y2 - y1 + 1)
    cx = float(x1) + w / 2.0
    cy = float(y1) + h / 2.0
    return np.array([cx, cy, w, h], dtype=np.float64)


def bbox_iou(a, b):
    """
    IoU between two bboxes in (cx, cy, w, h) format.
    Returns 0.0 if either bbox is None or degenerate.
    """
    if a is None or b is None:
        return 0.0
    ax1, ay1 = a[0] - a[2] / 2, a[1] - a[3] / 2
    ax2, ay2 = a[0] + a[2] / 2, a[1] + a[3] / 2
    bx1, by1 = b[0] - b[2] / 2, b[1] - b[3] / 2
    bx2, by2 = b[0] + b[2] / 2, b[1] + b[3] / 2
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter == 0.0:
        return 0.0
    union = a[2] * a[3] + b[2] * b[3] - inter
    return float(inter / union) if union > 0 else 0.0


class KalmanBoxTracker:
    """
    Constant-velocity Kalman filter for a single bounding box.

    State vector: [cx, cy, w, h, vx, vy, vw, vh]
    Observation:  [cx, cy, w, h]

    Modeled after SORT (Bewley et al.) with standard noise values.
    """

    def __init__(self, bbox):
        """
        bbox: (cx, cy, w, h) array — initial observation.
        """
        dt = 1.0

        # State transition matrix (constant velocity)
        self.F = np.eye(8, dtype=np.float64)
        for i in range(4):
            self.F[i, i + 4] = dt

        # Observation matrix
        self.H = np.zeros((4, 8), dtype=np.float64)
        self.H[:4, :4] = np.eye(4)

        # Process noise
        self.Q = np.eye(8, dtype=np.float64)
        self.Q[4:, 4:] *= 0.01

        # Measurement noise
        self.R = np.eye(4, dtype=np.float64)
        self.R[2:, 2:] *= 10.0  # size measurements are noisier

        # Covariance
        self.P = np.eye(8, dtype=np.float64)
        self.P[4:, 4:] *= 1000.0  # high uncertainty in initial velocity
        self.P *= 10.0

        # Initial state
        self.x = np.zeros((8, 1), dtype=np.float64)
        self.x[:4, 0] = bbox

        self._predicted = False

    def predict(self):
        """Advance state by one step and return predicted bbox (cx, cy, w, h)."""
        # Prevent negative size
        if self.x[2, 0] <= 0:
            self.x[2, 0] = 1.0
        if self.x[3, 0] <= 0:
            self.x[3, 0] = 1.0

        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        self._predicted = True
        return self.x[:4, 0].copy()

    def update(self, bbox):
        """Correct state with observed bbox (cx, cy, w, h)."""
        z = np.array(bbox, dtype=np.float64).reshape(4, 1)
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ (z - self.H @ self.x)
        self.P = (np.eye(8) - K @ self.H) @ self.P
