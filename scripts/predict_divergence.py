#!/usr/bin/env python3
"""Predict, from stored trajectories alone, where the memory bound first acts.

DAM4SAM fills its recent-appearance (RAM) slots by walking backwards through the
non-conditioning outputs until it finds a frame on which the target was visible
(dam4sam/sam2/modeling/sam2_base.py, the two `while True` loops). The walk has
no lower bound. The sliding window (MEM_WINDOW) can therefore change the
tracker's behaviour at frame t only if that walk would have selected a frame
that the window has already evicted.

Until the first such frame the bounded and the unbounded tracker hold identical
state, so the *baseline* trajectory is enough to compute it. This script

  1. replays the slot selection on the baseline's per-frame visibility and
     reports, per target-sequence, the first frame t* at which the unbounded
     walk reaches past the window (or "never");
  2. if a bounded variant's trajectories are given, finds the first frame d at
     which its stored output actually differs from the baseline's.

The mechanism makes two falsifiable predictions:
  - t* is "never"  ->  the two trajectories are identical on every frame;
  - t* is a frame  ->  d >= t*, and in practice d is at or shortly after t*.
A trajectory that differs *before* t*, or differs where t* is "never",
contradicts the account. No GPU and no ground truth are needed.

Approximations (the instrumented run is what removes them):
  - visibility is read from the stored output mask (non-empty), while the code
    tests `n_pixels_pos` on the low-resolution logits;
  - frames promoted to the DRM leave the non-conditioning store and are skipped
    by the walk; promotions are not recoverable from trajectories, so they are
    ignored;
  - the number of RAM slots is 7 minus the number of selected DRM frames (1..4),
    i.e. between 3 and 6. Both extremes are reported: 6 slots gives the earliest
    possible t*, 3 slots the latest.

Usage:
  python scripts/predict_divergence.py --selftest
  python scripts/predict_divergence.py --reference dam4sam_b
  python scripts/predict_divergence.py --reference dam4sam_b \
         --bounded dam4sam_ko_chassis --csv divergence.csv
"""
import argparse
import csv
import os
import random
import sys

STRIDE = 5        # memory_temporal_stride_for_eval in sam21pp_hiera_b+.yaml
MEM_WINDOW = 300  # DAM4SAMV3Tracker.MEM_WINDOW


# --------------------------------------------------------------------------
# Slot selection
# --------------------------------------------------------------------------
def select_reference(vis, t, n_slots, window=None, r=STRIDE):
    """Line-by-line replay of the RAM selection in sam2_base.py at frame t.

    vis[f] is True when the tracker's output on frame f was non-empty. Frame 0
    is the conditioning (initialisation) frame and is never a RAM candidate.
    With `window`, entries older than (t - 1 - window) have been evicted, which
    is the state the pruning in dam4sam_tracker_v3.py leaves before frame t.
    Returns the selected frame index per slot, None for an empty slot.
    """
    cutoff = 1 if window is None else max(1, t - 1 - window)

    def ok(p):
        return cutoff <= p < t and vis[p]

    selected = []
    p = None
    for i in range(n_slots):
        found = False
        if i == 0:
            p = t - 1
            found = ok(p)
            while not found:
                p -= 1
                if p > 0:
                    found = ok(p)
                else:
                    break
        elif p >= 0:
            p = ((p - 1) // r) * r
            found = ok(p)
            while not found:
                p -= r
                if p > 0:
                    found = ok(p)
                else:
                    break
        selected.append(p if found else None)
    return selected


def build_prev_visible(vis, r=STRIDE):
    """prev1[p]: largest visible q with 1 <= q <= p (0 if none).
    prevr[p]: largest visible multiple of r with 1 <= q <= p (0 if none)."""
    n = len(vis)
    prev1 = [0] * n
    prevr = [0] * n
    last1 = lastr = 0
    for p in range(1, n):
        if vis[p]:
            last1 = p
            if p % r == 0:
                lastr = p
        prev1[p] = last1
        prevr[p] = lastr
    return prev1, prevr


def select_unbounded_fast(prev1, prevr, t, n_slots, r=STRIDE):
    """Same result as select_reference(vis, t, n_slots, None), in O(n_slots)."""
    selected = []
    p = prev1[t - 1] if t >= 1 else 0
    for i in range(n_slots):
        if i > 0:
            if p > 0:
                q = ((p - 1) // r) * r
                p = prevr[q] if q > 0 else 0
            else:
                p = 0
        selected.append(p if p > 0 else None)
    return selected


def first_window_hit(vis, n_slots, window=MEM_WINDOW, r=STRIDE):
    """First frame t at which the unbounded walk selects an evicted frame."""
    prev1, prevr = build_prev_visible(vis, r)
    for t in range(1, len(vis)):
        cutoff = t - 1 - window
        if cutoff <= 1:
            continue
        sel = select_unbounded_fast(prev1, prevr, t, n_slots, r)
        if any(s is not None and s < cutoff for s in sel):
            return t
    return None


def selftest(trials=3000, seed=0):
    rng = random.Random(seed)
    for _ in range(trials):
        n = rng.randint(2, 500)
        # long runs of absence, so that the window actually binds
        vis, state = [], True
        for _f in range(n):
            if rng.random() < rng.choice([0.01, 0.05, 0.3]):
                state = not state
            vis.append(state)
        vis[0] = True
        n_slots = rng.randint(3, 6)
        window = rng.choice([10, 40, 120, 300])
        prev1, prevr = build_prev_visible(vis)
        hit = first_window_hit(vis, n_slots, window)
        for t in range(1, n):
            ref = select_reference(vis, t, n_slots)
            assert ref == select_unbounded_fast(prev1, prevr, t, n_slots), (vis, t)
            differs = ref != select_reference(vis, t, n_slots, window)
            if hit is None or t < hit:
                assert not differs, ("bounded differs before predicted hit", t, hit)
            elif t == hit:
                assert differs, ("no difference at predicted hit", t)
    print(f"selftest ok: {trials} random sequences, fast walk == reference walk, "
          "predicted hit == first frame the bounded selection differs")


# --------------------------------------------------------------------------
# Trajectories
# --------------------------------------------------------------------------
def load_trajectory(path):
    from vot.region.io import read_trajectory
    return read_trajectory(path)


def region_key(region):
    """Hashable content of one stored region, for exact frame-level comparison."""
    from vot.region import Special
    from vot.region.shapes import Mask
    if isinstance(region, Mask):
        return ("mask", tuple(region.offset), region.mask.shape, region.mask.tobytes())
    if isinstance(region, Special):
        return ("special", region.code)
    return (type(region).__name__, str(region))


def is_visible(region):
    from vot.region import Special
    from vot.region.shapes import Mask
    if isinstance(region, Mask):
        return bool(region.mask.any())
    if isinstance(region, Special):
        return False
    return not region.is_empty()


def trajectory_files(root):
    out = {}
    if not os.path.isdir(root):
        return out
    for seq in sorted(os.listdir(root)):
        seq_dir = os.path.join(root, seq)
        if not os.path.isdir(seq_dir):
            continue
        for f in sorted(os.listdir(seq_dir)):
            if f.endswith(".bin"):
                out[f"{seq}/{f}"] = os.path.join(seq_dir, f)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--workspace", default="workspace2026")
    ap.add_argument("--experiment", default="baseline")
    ap.add_argument("--reference", help="unbounded tracker, e.g. dam4sam_b")
    ap.add_argument("--bounded", help="tracker with the memory bound only, e.g. dam4sam_ko_chassis")
    ap.add_argument("--window", type=int, default=MEM_WINDOW)
    ap.add_argument("--csv", help="write the per-trajectory table here")
    args = ap.parse_args()

    if args.selftest:
        selftest()
        return
    if not args.reference:
        ap.error("--reference is required (or use --selftest)")

    res_root = os.path.join(args.workspace, "results")
    ref_files = trajectory_files(os.path.join(res_root, args.reference, args.experiment))
    if not ref_files:
        sys.exit(f"no trajectories for {args.reference!r} under {res_root}")
    var_files = (trajectory_files(os.path.join(res_root, args.bounded, args.experiment))
                 if args.bounded else {})
    if args.bounded and not var_files:
        sys.exit(f"no trajectories for {args.bounded!r} under {res_root}")

    rows = []
    for key, path in ref_files.items():
        ref = load_trajectory(path)
        vis = [is_visible(r) for r in ref]
        vis[0] = True  # initialisation frame
        longest_gap = gap = 0
        for v in vis[1:]:
            gap = 0 if v else gap + 1
            longest_gap = max(longest_gap, gap)
        row = {
            "trajectory": key,
            "frames": len(vis),
            "empty_frames": vis.count(False),
            "longest_empty_run": longest_gap,
            "hit_6slots": first_window_hit(vis, 6, args.window),
            "hit_3slots": first_window_hit(vis, 3, args.window),
        }
        if args.bounded:
            vpath = var_files.get(key)
            if vpath is None:
                row["first_diff"] = "missing"
            else:
                var = load_trajectory(vpath)
                d = next((f for f, (a, b) in enumerate(zip(ref, var))
                          if region_key(a) != region_key(b)), None)
                if d is None and len(ref) != len(var):
                    d = min(len(ref), len(var))
                row["first_diff"] = d
        rows.append(row)

    n = len(rows)
    predicted = [r for r in rows if r["hit_6slots"] is not None]
    print(f"{args.reference}: {n} trajectories, window = {args.window}")
    print(f"  window can act (6 slots): {len(predicted)}   never: {n - len(predicted)}")
    print(f"  window can act (3 slots): {sum(r['hit_3slots'] is not None for r in rows)}")

    if args.bounded:
        comparable = [r for r in rows if r["first_diff"] != "missing"]
        differ = [r for r in comparable if r["first_diff"] is not None]
        tp = [r for r in differ if r["hit_6slots"] is not None]
        unexplained = [r for r in differ if r["hit_6slots"] is None]
        silent = [r for r in comparable if r["first_diff"] is None and r["hit_6slots"] is not None]
        early = [r for r in tp if r["first_diff"] < r["hit_6slots"]]
        print(f"\n{args.bounded} vs {args.reference}: {len(differ)} of {len(comparable)} differ")
        print(f"  differ, and the window was predicted to act : {len(tp)}")
        print(f"  differ, but the window was predicted inert  : {len(unexplained)}   <- contradicts the account")
        print(f"  differ before the predicted frame           : {len(early)}   <- contradicts the account")
        print(f"  predicted to act, yet identical             : {len(silent)}")
        if tp:
            lags = sorted(r["first_diff"] - r["hit_6slots"] for r in tp)
            print(f"  lag (first_diff - predicted), frames: min {lags[0]}, "
                  f"median {lags[len(lags) // 2]}, max {lags[-1]}; "
                  f"exactly 0 on {sum(l == 0 for l in lags)}")
        for label, group in (("predicted inert but differ", unexplained), ("differ early", early)):
            for r in group[:15]:
                print(f"    {label}: {r['trajectory']}  first_diff={r['first_diff']} "
                      f"hit={r['hit_6slots']} longest_empty_run={r['longest_empty_run']}")

    if args.csv:
        with open(args.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\nwrote {args.csv}")


if __name__ == "__main__":
    main()
