"""
GT-free diagnostic evaluation of VOTS2026 tracker results.

Detects failure modes (object loss, mask instability) from predicted masks alone —
no ground truth, no images, no GPU needed. Results are failure PROXIES, not Q scores.

CAVEAT: These metrics detect object loss and gross instability, but CANNOT detect
silent drift onto a distractor — a stable mask on the wrong object looks "healthy"
here. "Clean" ≠ "high quality"; but "many dropouts/permanent losses" ≈ low robustness/Q.

Output: console report + workspace2026/analysis/diagnostics_nogt.json
"""
import sys
import re
import math
import json
import glob
import argparse
import numpy as np
from pathlib import Path

sys.path.insert(0, "/home/alexx/Projects/VOTSTracker")
from vot.region.io import read_trajectory
from vot.region import Special


WORKSPACE   = "/home/alexx/Projects/VOTSTracker/workspace2026"
TRACKERS    = ["dam4sam_b", "dam4sam_sam3"]
OUT_FILE    = "/home/alexx/Projects/VOTSTracker/workspace2026/analysis/diagnostics_nogt.json"


# ── helpers ──────────────────────────────────────────────────────────────────

def region_area(r):
    """Return pixel area of a Mask region, or 0 if empty/Special."""
    if isinstance(r, Special) or r.is_empty():
        return 0
    try:
        return int(np.asarray(r.mask).sum())
    except Exception:
        return 0


def analyse_trajectory(bin_path):
    """Compute per-object failure-proxy metrics from a single .bin trajectory file."""
    try:
        traj = read_trajectory(str(bin_path))
    except Exception as e:
        return {"error": str(e)}

    # frame 0 is the Special init marker — skip it
    frames = traj[1:]
    n = len(frames)
    if n == 0:
        return {"error": "empty trajectory (no frames after init)"}

    is_empty_arr = []
    area_arr     = []
    for r in frames:
        empty = isinstance(r, Special) or r.is_empty()
        is_empty_arr.append(empty)
        area_arr.append(0 if empty else region_area(r))

    # ── basic counts ──────────────────────────────────────────────────────────
    empty_count = sum(is_empty_arr)
    empty_rate  = empty_count / n

    # dropout  = present → empty
    # recovery = empty   → present
    dropouts   = 0
    recoveries = 0
    for i in range(1, n):
        if not is_empty_arr[i-1] and is_empty_arr[i]:
            dropouts += 1
        if is_empty_arr[i-1] and not is_empty_arr[i]:
            recoveries += 1

    # longest consecutive empty run
    longest_gap = 0
    cur_gap     = 0
    for e in is_empty_arr:
        if e:
            cur_gap += 1
            longest_gap = max(longest_gap, cur_gap)
        else:
            cur_gap = 0

    # permanent loss: had a mask at some point AND trailing tail is all empty
    ever_present = any(not e for e in is_empty_arr)
    tail_len     = max(10, math.ceil(0.1 * n))
    permanent_loss = ever_present and all(is_empty_arr[-tail_len:])

    # ── area stats ────────────────────────────────────────────────────────────
    non_empty_areas = [a for a in area_arr if a > 0]
    init_area   = non_empty_areas[0]  if non_empty_areas else 0
    final_area  = area_arr[-1]
    median_area = float(np.median(non_empty_areas)) if non_empty_areas else 0.0
    max_area    = max(non_empty_areas) if non_empty_areas else 0

    final_to_init = (final_area / init_area) if init_area > 0 else 0.0

    # area volatility: jumps >3× or <1/3× vs previous non-empty frame
    volatility = 0
    prev_area  = None
    for a in area_arr:
        if a > 0:
            if prev_area is not None and prev_area > 0:
                ratio = a / prev_area
                if ratio > 3.0 or ratio < (1.0 / 3.0):
                    volatility += 1
            prev_area = a

    # ── composite trouble score ───────────────────────────────────────────────
    trouble = (
        empty_rate
        + (1.0 if permanent_loss else 0.0)
        + min(volatility / n, 1.0)
    )

    return {
        "n_frames":         n,
        "empty_count":      empty_count,
        "empty_rate":       round(empty_rate, 4),
        "dropout_events":   dropouts,
        "recovery_events":  recoveries,
        "longest_empty_gap": longest_gap,
        "permanent_loss":   permanent_loss,
        "init_area":        init_area,
        "median_area":      round(median_area, 1),
        "max_area":         max_area,
        "final_area":       final_area,
        "final_to_init_ratio": round(final_to_init, 3),
        "area_volatility":  volatility,
        "trouble_score":    round(trouble, 4),
    }


def load_tracker_results(baseline_dir):
    """
    Returns: { seq_name: { obj_id: metrics_dict } }
    Handles both single-object (<seq>_001.bin) and multi-object (<seq>_obj<N>_001.bin).
    """
    results = {}
    baseline_path = Path(baseline_dir)
    for seq_dir in sorted(baseline_path.iterdir()):
        if not seq_dir.is_dir():
            continue
        seq = seq_dir.name
        bins = sorted(seq_dir.glob("*_001.bin"))
        if not bins:
            results[seq] = {}
            continue
        obj_metrics = {}
        for b in bins:
            m = re.search(r"_obj(\d+)_001\.bin$", b.name)
            obj_id = int(m.group(1)) if m else 1
            obj_metrics[obj_id] = analyse_trajectory(b)
        results[seq] = obj_metrics
    return results


def seq_trouble(obj_metrics):
    """Sequence-level trouble: mean per-object trouble, +1 per permanently-lost obj."""
    scores = []
    for m in obj_metrics.values():
        if "error" in m:
            scores.append(2.0)  # penalise errors heavily
        else:
            scores.append(m["trouble_score"])
    return float(np.mean(scores)) if scores else 0.0


def parse_time_file(path):
    """Read per-frame times from a _time.value file; return list of floats (seconds)."""
    try:
        with open(path) as f:
            return [float(line.strip()) for line in f if line.strip()]
    except Exception:
        return []


def mean_fps(baseline_dir):
    """Compute mean FPS across all _time.value files for a tracker."""
    all_times = []
    for tf in Path(baseline_dir).glob("**/*_time.value"):
        all_times.extend(parse_time_file(tf))
    all_times = [t for t in all_times if t > 0]
    if not all_times:
        return None
    mean_time = float(np.mean(all_times))
    return round(1.0 / mean_time, 2) if mean_time > 0 else None


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", default=WORKSPACE)
    parser.add_argument("--trackers",  nargs="+", default=TRACKERS)
    args = parser.parse_args()

    ws = Path(args.workspace)

    print()
    print("=" * 70)
    print("VOTS2026 GT-FREE DIAGNOSTIC (failure proxies, NOT quality scores)")
    print("=" * 70)
    print()
    print("CAVEAT: metrics detect object loss and gross mask instability only.")
    print("  'Clean' ≠ 'high quality'. Silent distractor drift is invisible here.")
    print("  Many dropouts / permanent losses ≈ low robustness (low Q on server).")
    print()

    all_results = {}

    for tracker in args.trackers:
        baseline_dir = ws / "results" / tracker / "baseline"
        if not baseline_dir.exists():
            print(f"[SKIP] {tracker}: no results dir at {baseline_dir}")
            continue
        all_results[tracker] = load_tracker_results(baseline_dir)

    # ── per-tracker summaries ─────────────────────────────────────────────────
    summary = {}
    for tracker, seqs in all_results.items():
        all_obj_metrics = []
        permanently_lost = []
        total_dropouts   = 0
        total_recoveries = 0

        for seq, objs in seqs.items():
            for obj_id, m in objs.items():
                if "error" in m:
                    continue
                all_obj_metrics.append(m)
                if m["permanent_loss"]:
                    permanently_lost.append((seq, obj_id))
                total_dropouts   += m["dropout_events"]
                total_recoveries += m["recovery_events"]

        n_seqs = len(seqs)
        n_objs = sum(len(o) for o in seqs.values())
        rates  = [m["empty_rate"] for m in all_obj_metrics]
        fps    = mean_fps(ws / "results" / tracker / "baseline")

        summary[tracker] = {
            "n_sequences":          n_seqs,
            "n_objects":            n_objs,
            "n_permanently_lost":   len(permanently_lost),
            "permanently_lost_list": [(s, oi) for s, oi in permanently_lost],
            "mean_empty_rate":      round(float(np.mean(rates)), 4)  if rates else 0.0,
            "median_empty_rate":    round(float(np.median(rates)), 4) if rates else 0.0,
            "total_dropout_events": total_dropouts,
            "total_recovery_events": total_recoveries,
            "mean_fps":             fps,
        }

        print(f"{'─'*70}")
        print(f"Tracker: {tracker}")
        print(f"  Sequences:          {n_seqs}")
        print(f"  Objects (total):    {n_objs}")
        print(f"  Permanently lost:   {len(permanently_lost)}")
        print(f"  Mean empty rate:    {summary[tracker]['mean_empty_rate']:.4f}")
        print(f"  Median empty rate:  {summary[tracker]['median_empty_rate']:.4f}")
        print(f"  Dropout events:     {total_dropouts}")
        print(f"  Recovery events:    {total_recoveries}")
        if fps:
            print(f"  Mean FPS:           {fps}")
        print()

    # ── worst-15 sequences per tracker ────────────────────────────────────────
    print("=" * 70)
    print("WORST 15 SEQUENCES PER TRACKER  (by trouble score):")
    print()
    for tracker, seqs in all_results.items():
        ranked = sorted(
            [(seq, seq_trouble(objs), len(objs)) for seq, objs in seqs.items()],
            key=lambda x: x[1], reverse=True
        )
        print(f"{tracker} worst:")
        print(f"  {'trouble':>8}  {'objects':>7}  sequence")
        for seq, trouble, n_obj in ranked[:15]:
            print(f"  {trouble:8.4f}  {n_obj:7d}  {seq}")
        print()

    # ── head-to-head ─────────────────────────────────────────────────────────
    head_to_head = []
    if len(args.trackers) == 2:
        a, b = args.trackers
        shared = set(all_results.get(a, {}).keys()) & set(all_results.get(b, {}).keys())
        for seq in shared:
            ta = seq_trouble(all_results[a][seq])
            tb = seq_trouble(all_results[b][seq])
            head_to_head.append({
                "seq":      seq,
                "trouble_a": round(ta, 4),
                "trouble_b": round(tb, 4),
                "divergence": round(tb - ta, 4),  # positive = b worse
            })
        head_to_head.sort(key=lambda x: x["divergence"])

        eps = 0.05
        b_worse  = sum(1 for h in head_to_head if h["divergence"] >  eps)
        b_better = sum(1 for h in head_to_head if h["divergence"] < -eps)
        equal    = len(head_to_head) - b_worse - b_better

        print("=" * 70)
        print(f"HEAD-TO-HEAD: {b} vs {a}")
        print(f"  divergence = trouble({b}) - trouble({a})")
        print(f"  positive → {b} worse;  negative → {b} better")
        print(f"  {b} worse: {b_worse} seqs  |  {b} better: {b_better} seqs  |  ~equal: {equal} seqs")
        print()
        print(f"BIGGEST REGRESSIONS (top-10 where {b} is WORSE):")
        print(f"  {'div':>8}  {a[:12]:>12} → {b[:12]:<12}  sequence")
        for h in reversed(head_to_head[-10:]):
            print(f"  {h['divergence']:+8.4f}  {h['trouble_a']:>12.4f} → {h['trouble_b']:<12.4f}  {h['seq']}")
        print()
        print(f"BIGGEST WINS (top-10 where {b} is BETTER):")
        print(f"  {'div':>8}  {a[:12]:>12} → {b[:12]:<12}  sequence")
        for h in head_to_head[:10]:
            print(f"  {h['divergence']:+8.4f}  {h['trouble_a']:>12.4f} → {h['trouble_b']:<12.4f}  {h['seq']}")

    # ── sanity spot-checks ────────────────────────────────────────────────────
    print()
    print("=" * 70)
    print("SPOT-CHECKS (known sequences):")
    for check_seq in ["robot-19", "ants1"]:
        print(f"\n  {check_seq}:")
        for tracker in args.trackers:
            objs = all_results.get(tracker, {}).get(check_seq, {})
            if not objs:
                print(f"    {tracker}: not found")
                continue
            # count .bin files on disk for cross-check
            bin_count = len(list(
                (ws / "results" / tracker / "baseline" / check_seq).glob("*_001.bin")
            ))
            for obj_id, m in sorted(objs.items()):
                if "error" in m:
                    print(f"    {tracker} obj={obj_id}: ERROR {m['error']}")
                else:
                    print(f"    {tracker} obj={obj_id}: "
                          f"n={m['n_frames']} empty={m['empty_rate']:.3f} "
                          f"perm_loss={m['permanent_loss']} "
                          f"trouble={m['trouble_score']:.4f}")
            print(f"    {tracker}: {len(objs)} objects in results, {bin_count} .bin files on disk")

    # ── write JSON ────────────────────────────────────────────────────────────
    Path(OUT_FILE).parent.mkdir(parents=True, exist_ok=True)

    # convert per-sequence data: make obj keys strings for JSON
    per_sequence = {}
    for tracker, seqs in all_results.items():
        per_sequence[tracker] = {}
        for seq, objs in seqs.items():
            per_sequence[tracker][seq] = {
                "n_objects":   len(objs),
                "seq_trouble": round(seq_trouble(objs), 4),
                "per_obj":     {str(k): v for k, v in objs.items()},
            }

    out = {
        "caveats": (
            "Failure proxies only — no GT used. Detects object loss and gross mask "
            "instability. Cannot detect silent distractor drift. "
            "'Clean' != 'high quality'."
        ),
        "trackers": args.trackers,
        "summary":  summary,
        "per_sequence": per_sequence,
        "head_to_head": head_to_head,
    }
    with open(OUT_FILE, "w") as f:
        json.dump(out, f, indent=2)
    print()
    print(f"Wrote full breakdown to {OUT_FILE}")
    print()


if __name__ == "__main__":
    main()
