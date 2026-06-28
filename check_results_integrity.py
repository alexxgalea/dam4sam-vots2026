#!/usr/bin/env python3
"""Pre-pack integrity check (PLAN_WIN_VOTS2026 §6.3).

For every sequence in the workspace, verify the tracker results:
  - all expected per-object .bin files exist (catches the v3 silent 0/7-masks bug)
  - trajectory length == frame count
  - not all masks empty after init (degenerate output)

Usage: python check_results_integrity.py --workspace workspace2026 --tracker dam4sam_v3g
Exit code 0 = clean (PENDING sequences allowed with --allow-pending), 1 = FAIL.
"""
import argparse
import os
import re
import sys

from vot.region.io import read_trajectory
from vot.region import Mask


def expected_objects(seq_dir):
    objs = sorted(
        int(m.group(1))
        for f in os.listdir(seq_dir)
        if (m := re.match(r"groundtruth_obj(\d+)\.txt$", f))
    )
    if objs:
        return objs
    return [0]  # single-object: groundtruth.txt -> <seq>_001.bin


def check_sequence(seq, seq_dir, res_dir):
    n_frames = len([f for f in os.listdir(os.path.join(seq_dir, "color")) if f.endswith(".jpg")])
    problems = []
    if not os.path.isdir(res_dir):
        return None, ["PENDING (no result dir)"]
    for obj in expected_objects(seq_dir):
        name = f"{seq}_001.bin" if obj == 0 else f"{seq}_obj{obj}_001.bin"
        path = os.path.join(res_dir, name)
        if not os.path.isfile(path):
            problems.append(f"MISSING {name}")
            continue
        try:
            traj = read_trajectory(path)
        except Exception as e:
            problems.append(f"UNREADABLE {name}: {e}")
            continue
        if len(traj) != n_frames:
            problems.append(f"LENGTH {name}: {len(traj)} != {n_frames} frames")
        nonempty = sum(
            1 for r in traj[1:] if isinstance(r, Mask) and r.mask is not None and r.mask.sum() > 0
        )
        if nonempty == 0:
            problems.append(f"ALL-EMPTY {name} (degenerate after init)")
    return n_frames, problems


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workspace", default="workspace2026")
    ap.add_argument("--tracker", required=True)
    ap.add_argument("--experiment", default="baseline")
    ap.add_argument("--allow-pending", action="store_true",
                    help="pending sequences (no result dir) are not failures")
    args = ap.parse_args()

    seq_root = os.path.join(args.workspace, "sequences")
    res_root = os.path.join(args.workspace, "results", args.tracker, args.experiment)
    list_file = os.path.join(seq_root, "list.txt")
    if os.path.isfile(list_file):
        seqs = [l.strip() for l in open(list_file) if l.strip()]
    else:
        seqs = sorted(d for d in os.listdir(seq_root)
                      if os.path.isdir(os.path.join(seq_root, d, "color")))

    ok, pending, failed = 0, [], []
    for seq in seqs:
        _, problems = check_sequence(seq, os.path.join(seq_root, seq),
                                     os.path.join(res_root, seq))
        if not problems:
            ok += 1
        elif problems == ["PENDING (no result dir)"]:
            pending.append(seq)
        else:
            failed.append(seq)
            for p in problems:
                print(f"FAIL {seq}: {p}")

    print(f"\n{args.tracker}: {ok} OK / {len(pending)} pending / {len(failed)} FAILED "
          f"(of {len(seqs)})")
    if pending:
        print("pending:", " ".join(pending))
    if failed:
        sys.exit(1)
    if pending and not args.allow_pending:
        sys.exit(1)


if __name__ == "__main__":
    main()
