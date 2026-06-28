"""
Local VOTS2026 analysis — bypasses the broken evaluation server.

Computes per-sequence and overall Q (Quality / average IoU) for each tracker
using the groundtruth masks already in the workspace. Multi-object aware.

Output: prints summary table + writes per-sequence breakdown to analysis_local.json.
"""
import sys
import json
import numpy as np
from pathlib import Path

sys.path.insert(0, "/home/alexx/Projects/VOTSTracker")
from vot.workspace import Workspace
from vot.region.raster import calculate_overlaps
from vot.region import Special


WORKSPACE = "/home/alexx/Projects/VOTSTracker/workspace2026"
TRACKERS = ["dam4sam_b", "dam4sam_sam3"]
OUT_FILE = "/home/alexx/Projects/VOTSTracker/workspace2026/analysis/local_metrics.json"


def per_object_iou(tracker_regions, gt_regions, bounds):
    """Per-frame IoU between predicted regions and groundtruth.

    Frame is skipped (ignored from average) if groundtruth is Special (invisible/unknown).
    If groundtruth is Mask and tracker is Special → IoU=0 (failure to report).
    """
    ious = []
    skipped = 0
    n = min(len(tracker_regions), len(gt_regions))
    for i in range(n):
        g, r = gt_regions[i], tracker_regions[i]
        if isinstance(g, Special):
            # Groundtruth says invisible/unknown → skip from quality measure
            skipped += 1
            continue
        if isinstance(r, Special):
            # Tracker didn't report when it should → IoU=0
            ious.append(0.0)
            continue
        try:
            iou = calculate_overlaps([r], [g], bounds=bounds)[0]
        except Exception:
            iou = 0.0
        ious.append(float(iou))
    return ious, skipped


def main():
    ws = Workspace.load(WORKSPACE)
    exp = list(ws.stack)[0]
    sequences = list(ws.dataset)
    trackers = [ws.registry[t] for t in TRACKERS]

    print(f"Loaded {len(sequences)} sequences, {len(trackers)} trackers, experiment={exp.identifier}")
    print()

    # tracker -> sequence -> {q_per_obj, n_frames_visible, n_objects}
    results = {tr.identifier: {} for tr in trackers}

    from PIL import Image

    for s_idx, seq in enumerate(sequences):
        # seq.size is (None, None) — read dimensions from the first frame
        first_frame_path = seq.frame(0).filename()
        w, h = Image.open(first_frame_path).size
        bounds = (w - 1, h - 1)
        for tr in trackers:
            # gather() returns one Trajectory per object
            try:
                trajs = exp.gather(tr, seq, pad=True)
            except Exception as e:
                results[tr.identifier][seq.name] = {"error": str(e)}
                continue

            # per-object groundtruth — seq.object(id) returns a list of per-frame regions directly
            obj_ids = list(seq.objects())
            per_obj_q = []
            per_obj_visible = []
            for obj_idx, obj_id in enumerate(obj_ids):
                if obj_idx >= len(trajs):
                    per_obj_q.append(0.0)
                    per_obj_visible.append(0)
                    continue
                pred = trajs[obj_idx].regions()
                gt = seq.object(obj_id)
                ious, skipped = per_object_iou(pred, gt, bounds)
                n_visible = len(ious)
                q = float(np.mean(ious)) if ious else 0.0
                per_obj_q.append(q)
                per_obj_visible.append(n_visible)

            # Mean Q across objects for this sequence (VOTS convention)
            seq_q = float(np.mean(per_obj_q)) if per_obj_q else 0.0
            results[tr.identifier][seq.name] = {
                "q": seq_q,
                "per_obj_q": per_obj_q,
                "per_obj_visible": per_obj_visible,
                "n_objects": len(obj_ids),
            }

        if (s_idx + 1) % 10 == 0:
            print(f"  processed {s_idx + 1}/{len(sequences)} sequences...", flush=True)

    # Aggregate overall metrics
    print()
    print("=" * 60)
    summary = {}
    for tr_id, seqs in results.items():
        qs = [s["q"] for s in seqs.values() if "q" in s]
        weighted_qs = []
        total_objects = 0
        for s in seqs.values():
            if "q" not in s:
                continue
            for q, vis in zip(s["per_obj_q"], s["per_obj_visible"]):
                weighted_qs.append(q)  # equal-weight per object
            total_objects += s["n_objects"]
        summary[tr_id] = {
            "Q_seq_mean": float(np.mean(qs)),
            "Q_obj_mean": float(np.mean(weighted_qs)) if weighted_qs else 0.0,
            "n_sequences": len(qs),
            "n_objects": total_objects,
        }
        print(f"{tr_id}:")
        print(f"  Mean Q (avg of per-seq):      {summary[tr_id]['Q_seq_mean']:.4f}")
        print(f"  Mean Q (avg of per-object):   {summary[tr_id]['Q_obj_mean']:.4f}")
        print(f"  Sequences: {summary[tr_id]['n_sequences']} | Objects: {summary[tr_id]['n_objects']}")
        print()

    # Per-sequence comparison (top 15 weakest for each tracker)
    print("=" * 60)
    print("WORST 15 SEQUENCES PER TRACKER:")
    print()
    for tr_id, seqs in results.items():
        sorted_seqs = sorted(((name, s["q"]) for name, s in seqs.items() if "q" in s), key=lambda x: x[1])
        print(f"{tr_id} worst:")
        for name, q in sorted_seqs[:15]:
            print(f"  {q:.4f}  {name}")
        print()

    # Where SAM3 beat/lost vs baseline
    if len(TRACKERS) == 2:
        a, b = TRACKERS  # b vs sam3 typically
        diffs = []
        for name in results[a]:
            if "q" in results[a].get(name, {}) and "q" in results[b].get(name, {}):
                diff = results[b][name]["q"] - results[a][name]["q"]
                diffs.append((name, diff, results[a][name]["q"], results[b][name]["q"]))
        diffs.sort(key=lambda x: x[1])  # ascending = sam3 worst regressions first
        print("=" * 60)
        print(f"BIGGEST REGRESSIONS ({b} - {a}): negative = {b} worse")
        for name, diff, qa, qb in diffs[:10]:
            print(f"  {diff:+.4f}   {qa:.3f} -> {qb:.3f}   {name}")
        print()
        print(f"BIGGEST WINS ({b} - {a}):  positive = {b} better")
        for name, diff, qa, qb in diffs[-10:][::-1]:
            print(f"  {diff:+.4f}   {qa:.3f} -> {qb:.3f}   {name}")

    # Write JSON
    Path(OUT_FILE).parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_FILE, "w") as f:
        json.dump({"summary": summary, "per_sequence": results}, f, indent=2)
    print()
    print(f"Wrote detailed results to {OUT_FILE}")


if __name__ == "__main__":
    main()
