"""
Compute VOTS2025 Quality (Q) metric from dam4sam_b results.
Q = mean per-object IoU across all non-init frames, averaged over all sequences.
"""
import os
import numpy as np
from vot.region.io import read_trajectory
from vot.region.shapes import Mask, Rectangle
from vot.region import Special
from vot.region.raster import calculate_overlaps

SEQ_BASE = "workspace/sequences"
RES_BASE = "workspace/results/dam4sam_b/baseline"


def get_seq_dims(seq):
    info = open(f"{SEQ_BASE}/{seq}/sequence").read()
    w, h = None, None
    for line in info.splitlines():
        if line.startswith("width="):    w = int(line.split("=")[1])
        if line.startswith("height="):   h = int(line.split("=")[1])
    return w, h


def compute_seq_iou(seq):
    res_dir = f"{RES_BASE}/{seq}"
    seq_dir = f"{SEQ_BASE}/{seq}"
    if not os.path.isdir(res_dir):
        return None, 0  # skipped/failed

    # Determine result files and GT files
    result_bins = sorted([f for f in os.listdir(res_dir) if f.endswith(".bin")])
    if not result_bins:
        return None, 0

    w, h = get_seq_dims(seq)

    # Multi-object: result file per object
    # Single-object sequences have one bin, GT is groundtruth.txt or groundtruth_obj1.txt
    gt_files = sorted([f for f in os.listdir(seq_dir) if f.startswith("groundtruth_obj") and f.endswith(".txt")])
    if not gt_files:
        gt_files = ["groundtruth.txt"] if os.path.exists(f"{seq_dir}/groundtruth.txt") else []

    all_ious = []
    n_frames = 0

    import re
    for bin_file in result_bins:
        # bin naming: {seq}_001.bin (single obj) or {seq}_obj{n}_001.bin (multi obj)
        traj = read_trajectory(f"{res_dir}/{bin_file}")

        # Find matching GT
        m = re.search(r'_obj(\d+)_', bin_file)
        if m:
            obj_idx = int(m.group(1))
            gt_path = f"{seq_dir}/groundtruth_obj{obj_idx}.txt"
        else:
            # Single-object: use groundtruth.txt or groundtruth_obj1.txt
            gt_path = f"{seq_dir}/groundtruth.txt"
            if not os.path.exists(gt_path):
                gt_path = f"{seq_dir}/groundtruth_obj1.txt"

        if not os.path.exists(gt_path):
            continue

        gt_traj = read_trajectory(gt_path)
        n = min(len(traj), len(gt_traj))

        frame_ious = []
        for i in range(1, n):  # skip frame 0 (init)
            pred = traj[i]
            gt = gt_traj[i]

            if isinstance(gt, Special):
                continue  # skip absence/special frames

            if isinstance(pred, Special) or pred is None:
                frame_ious.append(0.0)
                continue

            try:
                iou = calculate_overlaps([pred], [gt], (w, h))[0]
                frame_ious.append(float(iou))
            except Exception:
                frame_ious.append(0.0)

        if frame_ious:
            all_ious.extend(frame_ious)
            n_frames += len(frame_ious)

    if not all_ious:
        return None, 0
    return float(np.mean(all_ious)), n_frames


def main():
    seqs = sorted([s for s in os.listdir(RES_BASE)
                   if os.path.isdir(f"{RES_BASE}/{s}")])

    all_seqs = sorted([s for s in os.listdir(SEQ_BASE)
                       if os.path.isdir(f"{SEQ_BASE}/{s}") and s != "list.txt"])

    print(f"{'Sequence':<25} {'Q':>8} {'Frames':>8}")
    print("-" * 45)

    seq_qs = []
    failed = []

    for seq in all_seqs:
        q, n = compute_seq_iou(seq)
        if q is None:
            failed.append(seq)
            print(f"{seq:<25} {'FAILED':>8} {'—':>8}")
        else:
            seq_qs.append(q)
            print(f"{seq:<25} {q:8.4f} {n:8d}")

    print("-" * 45)
    print(f"\n{'Overall Q (mean over sequences):':<35} {np.mean(seq_qs):.4f}")
    print(f"{'Sequences evaluated:':<35} {len(seq_qs)}/144")
    print(f"{'Sequences failed/skipped:':<35} {len(failed)} {failed}")


if __name__ == "__main__":
    main()
