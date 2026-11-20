#!/usr/bin/env python3
"""Download the VOTS2026 sequences with the VOT toolkit, restarting on stalls.

The toolkit's downloader has no network timeout: a dropped connection hangs it
for good. This supervisor runs it in a child process and restarts it when the
child writes nothing for STALL seconds. Finished sequences are skipped by the
toolkit (it checks the `sequence` file), so a restart loses one sequence at most.

Usage: python scripts/download_sequences.py [target_dir]   (default workspace/sequences)
"""
import subprocess
import sys
import time
from pathlib import Path

URL = "https://data.votchallenge.net/vots2026/main/description.json"
STALL = 180
REPO = Path(__file__).resolve().parents[1]
target = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO / "workspace/sequences"
child = ("import logging,sys; logging.basicConfig(level=logging.INFO); "
         "from vot.dataset.common import download_dataset_meta; "
         f"download_dataset_meta({URL!r}, {str(target)!r})")


def written(pid):
    try:
        for line in open(f"/proc/{pid}/io"):
            if line.startswith("wchar"):
                return int(line.split()[1])
    except OSError:
        return None


attempt = 0
while True:
    attempt += 1
    p = subprocess.Popen([sys.executable, "-c", child], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    last, last_change = written(p.pid), time.time()
    while p.poll() is None:
        time.sleep(20)
        now = written(p.pid)
        if now != last:
            last, last_change = now, time.time()
        elif time.time() - last_change > STALL:
            p.kill()
            p.wait()
            break
    done = sum(1 for d in target.iterdir() if (d / "sequence").is_file()) if target.exists() else 0
    print(f"{time.strftime('%H:%M:%S')} attempt {attempt}: exit={p.returncode}, {done} sequences complete", flush=True)
    if p.returncode == 0:
        break
    time.sleep(5)
print("DOWNLOAD DONE", flush=True)
