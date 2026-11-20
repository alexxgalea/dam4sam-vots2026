#!/usr/bin/env python3
"""Single-constant control for the memory bound: run `ko_nowindow` and
`ko_chassis` back to back, sequence by sequence, in the same process setup.

The two variants differ by MEM_WINDOW alone, so with a deterministic pipeline
any difference between their outputs is the window's doing. Both runs log the
RAM slot selection (--log-selection), which gives the exact frame at which the
unbounded search first reaches past the window.

Sequences are processed in the given order; a sequence is started only once
all of its frames are on disk (so the script can run while the dataset is
still downloading). Finished trajectories are skipped, so it resumes.

Usage:
  python scripts/run_window_control.py --seqs bus,car-9 --tag ac3060
  python scripts/run_window_control.py --seq-file logs/stage1.txt --tag ac3060 --fp32
"""
import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SEQS = REPO / "workspace2026/sequences"
DESCRIPTION = "https://data.votchallenge.net/vots2026/main/description.json"


def expected_lengths():
    cache = REPO / "logs/vots2026_description.json"
    if not cache.exists():
        cache.parent.mkdir(exist_ok=True)
        urllib.request.urlretrieve(DESCRIPTION, cache)
    return {s["name"]: s["length"] for s in json.load(open(cache))["sequences"]}


def ready(seq, length):
    d = SEQS / seq
    n = len(list((d / "color").glob("*.jpg"))) + len(list((d / "color").glob("*.png")))
    has_gt = (d / "groundtruth.txt").exists() or any(d.glob("groundtruth_obj*.txt"))
    return n == length and has_gt


def evict_page_cache(root):
    """Ask the kernel to drop cached pages of the files under root (no root needed).

    Under WSL2 a full page cache starves the GPU paravirtualisation of host
    memory: allocations fail with 'CUDA error: out of memory' while VRAM is
    nearly empty (dmesg: dxgkio_create_allocation / GPADL failures).
    """
    for dirpath, _dirs, files in os.walk(root, followlinks=True):
        for name in files:
            try:
                fd = os.open(os.path.join(dirpath, name), os.O_RDONLY)
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
                os.close(fd)
            except OSError:
                pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seqs")
    ap.add_argument("--seq-file")
    ap.add_argument("--tag", required=True, help="output dirs are <tag>_nowindow and <tag>_chassis")
    ap.add_argument("--fp32", action="store_true", help="default is autocast, as the toolkit wrappers")
    ap.add_argument("--after", help="wait until this file contains the line 'DOWNLOAD DONE'")
    args = ap.parse_args()

    seqs = args.seqs.split(",") if args.seqs else Path(args.seq_file).read_text().split()
    lengths = expected_lengths()
    unknown = [s for s in seqs if s not in lengths]
    if unknown:
        sys.exit(f"unknown sequences: {unknown}")

    if args.after:
        print(f"waiting for the download to finish ({args.after})", flush=True)
        while "DOWNLOAD DONE" not in Path(args.after).read_text():
            time.sleep(30)
        os.sync()
        evict_page_cache(SEQS)
        print("download finished, page cache evicted", flush=True)

    t0 = time.time()
    for k, seq in enumerate(seqs, 1):
        waited = False
        while not ready(seq, lengths[seq]):
            if not waited:
                print(f"[{k}/{len(seqs)}] {seq}: waiting for frames on disk", flush=True)
                waited = True
            time.sleep(60)
        for variant in ("ko_nowindow", "ko_chassis"):
            out = f"{args.tag}_{variant[3:]}"
            cmd = [sys.executable, str(REPO / "scripts/run_eval_170.py"), "--tracker", variant,
                   "--seqs", seq, "--out", out, "--log-selection", "--strict", "--blocking-copies"]
            if not args.fp32:
                cmd.append("--autocast")
            t1 = time.time()
            res = subprocess.run(cmd, capture_output=True, text=True)
            keep = [l for l in res.stdout.splitlines()
                    if "frames ->" in l or "eroare" in l or "SKIP" in l]
            status = "ok" if res.returncode == 0 else f"EXIT {res.returncode}"
            print(f"[{k}/{len(seqs)}] {seq} {variant}: {status}, {time.time() - t1:.0f}s "
                  f"(total {(time.time() - t0) / 3600:.2f} h)", flush=True)
            for l in keep:
                print("    " + l.strip(), flush=True)
            if res.returncode != 0:
                print(res.stdout[-1500:], flush=True)
                print(res.stderr[-1500:], flush=True)
                sys.exit(f"STOPPED: {seq} {variant} failed; nothing was written for the failing object")
        evict_page_cache(SEQS / seq)
    print("ALL DONE", flush=True)


if __name__ == "__main__":
    main()
