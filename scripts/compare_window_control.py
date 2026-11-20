#!/usr/bin/env python3
"""Compare the window-off and bounded runs produced by run_window_control.py.

For every trajectory present in both runs it reports
  hit        first frame at which the unbounded run selected a RAM frame that the
             window would already have evicted (exact, from the selection log);
  sel_diff   first frame at which the two runs filled their RAM slots differently;
  mask_diff  first frame at which the output masks differ.
The account of how the bound acts predicts: no hit -> identical output; with a
hit -> sel_diff == hit and mask_diff >= hit. Anything else contradicts it.

Usage: python scripts/compare_window_control.py --tag ac3060 [--csv out.csv]
"""
import argparse
import csv
import os
from pathlib import Path

import numpy as np
from vot.region.io import read_trajectory
from vot.region.shapes import Mask

REPO = Path(__file__).resolve().parents[1]
W = 300


def content(r):
    if not isinstance(r, Mask):
        return None
    m = r.mask
    rows = m.any(axis=1)
    if not rows.any():
        return None
    cols = m.any(axis=0)
    y0, y1 = int(np.argmax(rows)), len(rows) - 1 - int(np.argmax(rows[::-1]))
    x0, x1 = int(np.argmax(cols)), len(cols) - 1 - int(np.argmax(cols[::-1]))
    return r.offset[0] + x0, r.offset[1] + y0, m[y0:y1 + 1, x0:x1 + 1].tobytes(), y1 - y0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--csv")
    args = ap.parse_args()
    root = REPO / "workspace2026/results"
    a_root, b_root = root / f"{args.tag}_nowindow/baseline", root / f"{args.tag}_chassis/baseline"
    rows = []
    for a_bin in sorted(a_root.glob("*/*.bin")):
        b_bin = b_root / a_bin.parent.name / a_bin.name
        if not b_bin.exists():
            continue
        sa = np.load(str(a_bin)[:-4] + "_ramsel.npy")
        sb = np.load(str(b_bin)[:-4] + "_ramsel.npy")
        slots = sa[:, 2:]
        deep = ((slots >= 0) & (slots < (sa[:, :1] - 1 - W))).any(axis=1)
        hit = int(sa[deep, 0][0]) if deep.any() else None
        depth = np.where(slots >= 0, sa[:, :1] - slots, 0).max(axis=1)   # how far back each frame reached
        n = min(len(sa), len(sb))
        d = np.nonzero((sa[:n] != sb[:n]).any(axis=1))[0]
        sel_diff = int(sa[d[0], 0]) if len(d) else None
        ta, tb = read_trajectory(str(a_bin)), read_trajectory(str(b_bin))
        ca, cb = [content(r) for r in ta], [content(r) for r in tb]
        mask_diff = next((f for f in range(1, min(len(ca), len(cb))) if ca[f] != cb[f]), None)
        rows.append(dict(
            traj=f"{a_bin.parent.name}/{a_bin.name}", frames=len(ta), hit=hit, sel_diff=sel_diff,
            mask_diff=mask_diff, frames_window_active=int(deep.sum()), max_depth=int(depth.max()),
            empty_nowindow=sum(c is None for c in ca[1:]), empty_window=sum(c is None for c in cb[1:]),
            n_masks_differ=sum(x != y for x, y in zip(ca[1:], cb[1:])),
            byte_identical=open(a_bin, "rb").read() == open(b_bin, "rb").read()))
    if not rows:
        print("no trajectory present in both runs yet")
        return
    print(f"{'trajectory':<38}{'frames':>7}{'hit':>7}{'sel_diff':>9}{'mask_diff':>10}{'active':>7}"
          f"{'maxdepth':>9}{'empty off':>10}{'empty on':>9}{'differ':>7}")
    for r in rows:
        f = lambda v: "-" if v is None else str(v)
        print(f"{r['traj']:<38}{r['frames']:>7}{f(r['hit']):>7}{f(r['sel_diff']):>9}{f(r['mask_diff']):>10}"
              f"{r['frames_window_active']:>7}{r['max_depth']:>9}{r['empty_nowindow']:>10}{r['empty_window']:>9}"
              f"{r['n_masks_differ']:>7}")
    inert = [r for r in rows if r["hit"] is None]
    act = [r for r in rows if r["hit"] is not None]
    print(f"\n{len(rows)} trajectories compared")
    print(f"window never reached ({len(inert)}): byte-identical {sum(r['byte_identical'] for r in inert)}, "
          f"differing {sum(not r['byte_identical'] for r in inert)}   <- any differing one contradicts the account")
    print(f"window reached ({len(act)}): sel_diff == hit on {sum(r['sel_diff'] == r['hit'] for r in act)}, "
          f"mask differs before hit on {sum(r['mask_diff'] is not None and r['mask_diff'] < r['hit'] for r in act)}, "
          f"output identical anyway on {sum(r['mask_diff'] is None for r in act)}")
    if act:
        e0, e1 = sum(r["empty_nowindow"] for r in act), sum(r["empty_window"] for r in act)
        print(f"empty frames in those: window off {e0}, window on {e1} ({e1 - e0:+d})")
    if args.csv:
        with open(args.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, list(rows[0]))
            w.writeheader()
            w.writerows(rows)


if __name__ == "__main__":
    main()
