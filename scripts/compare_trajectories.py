#!/usr/bin/env python3
"""Byte-level comparison of stored VOT trajectories between tracker variants.

The knock-out ablation in docs/ABLATION.md rests on a distinction that metrics
alone cannot make: a component that *contributes nothing* looks exactly like a
component that *never executed*. Comparing the stored per-object trajectory
files byte for byte settles it -- an identical output on every sequence means
the disabled code path never influenced a single frame.

It also guards the harness itself: a knock-out that differs on no sequence
could equally well be a plumbing bug (flag not wired through), so the variants
that *are* expected to differ must actually differ.

Needs no ground truth and no GPU; it reads the results directory only.

Usage:
  python scripts/compare_trajectories.py --reference dam4sam_v3g --variants 'dam4sam_ko_*'
  python scripts/compare_trajectories.py --reference dam4sam_ko_postproc \
                                         --variants dam4sam_ko_chassis --list-differing

Exit code 0 = every variant was compared against a complete reference,
1 = a variant is missing files the reference has (incomparable).
"""
import argparse
import filecmp
import fnmatch
import os
import sys


def trajectory_files(root):
    """Map 'sequence/file.bin' -> absolute path for one tracker/experiment."""
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
    ap = argparse.ArgumentParser()
    ap.add_argument("--workspace", default="workspace2026")
    ap.add_argument("--experiment", default="baseline")
    ap.add_argument("--reference", required=True,
                    help="tracker whose results are the comparison baseline")
    ap.add_argument("--variants", nargs="+", required=True,
                    help="tracker names or glob patterns, e.g. 'dam4sam_ko_*'")
    ap.add_argument("--list-differing", action="store_true",
                    help="print the sequences that differ, not just the counts")
    args = ap.parse_args()

    res_root = os.path.join(args.workspace, "results")
    ref = trajectory_files(os.path.join(res_root, args.reference, args.experiment))
    if not ref:
        sys.exit(f"no trajectories found for reference {args.reference!r} in {res_root}")

    available = sorted(d for d in os.listdir(res_root)
                       if os.path.isdir(os.path.join(res_root, d)) and d != args.reference)
    names = []
    for pattern in args.variants:
        matched = fnmatch.filter(available, pattern)
        names.extend(matched if matched else [pattern])
    names = sorted(dict.fromkeys(names))

    print(f"reference: {args.reference} ({len(ref)} trajectory files)\n")
    print(f"{'variant':<28} {'identical':>9} {'differ':>7} {'missing':>8}")
    print("-" * 56)

    incomparable = False
    for name in names:
        var = trajectory_files(os.path.join(res_root, name, args.experiment))
        identical, differing, missing = 0, [], []
        for key, ref_path in ref.items():
            var_path = var.get(key)
            if var_path is None:
                missing.append(key)
            elif filecmp.cmp(ref_path, var_path, shallow=False):
                identical += 1
            else:
                differing.append(key)
        print(f"{name:<28} {identical:>9} {len(differing):>7} {len(missing):>8}"
              + ("   <- byte-identical on all files" if identical == len(ref) else ""))
        if missing:
            incomparable = True
        if args.list_differing and differing:
            for key in differing:
                print(f"    differs: {key}")
        if missing:
            print(f"    MISSING {len(missing)} file(s), e.g. {missing[0]}")

    sys.exit(1 if incomparable else 0)


if __name__ == "__main__":
    main()
