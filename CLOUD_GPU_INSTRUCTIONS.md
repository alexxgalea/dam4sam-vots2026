# Cloud GPU — Hiera-L 170-run (S2) — user instructions

**Goal:** run DAM4SAM baseline with the SAM2.1-**Large** backbone on all 170 VOTS2026 sequences.
Expected gain vs B+ (Q=0.69): +0.02–0.04. Cost: ~$15–30 total.

## 1. Rent the instance (only step Claude cannot do)
- vast.ai or runpod.io → **RTX 4090 24GB** (or A100/A10). Filters: **CUDA ≥12.1**,
  **disk ≥ 60GB**, Ubuntu 22.04 / PyTorch image, good download bandwidth (≥200 Mbps).
- Optional speed-up: rent **2** instances and use `half1`/`half2` mode (~1 day instead of ~2).
- Get SSH access (`ssh -p PORT root@HOST`). Then either paste the SSH command to Claude,
  or run the steps below yourself.

## 2. Upload code (from the local machine, ~12MB — dataset & weights download on the instance)
```bash
cd /home/alexx/Projects/VOTSTracker
rsync -avz -e "ssh -p PORT" --relative \
  ./dam4sam --exclude dam4sam/.venv --exclude dam4sam/checkpoints \
  ./run_eval_170.py ./check_results_integrity.py ./cloud_setup.sh ./cloud_run_l.sh \
  root@HOST:/home/alexx/Projects/VOTSTracker/
```

## 3. On the instance
```bash
cd /home/alexx/Projects/VOTSTracker
bash cloud_setup.sh        # venvs + deps + Large ckpt + 26GB dataset (~20-40 min)
bash cloud_run_l.sh smoke  # MANDATORY: 1 sequence end-to-end + integrity (guardrail §6.1)
bash cloud_run_l.sh full   # or half1 / half2 on a 2-instance setup
tail -f watch_l_full.log   # progress every 5 min
```
Use `tmux`/`screen` so disconnects don't matter (`nohup` already protects the run itself).

## 4. Pull results back (from local, when watcher reports done + integrity OK)
```bash
rsync -avz -e "ssh -p PORT" \
  root@HOST:/home/alexx/Projects/VOTSTracker/workspace2026/results/dam4sam_l_cloud \
  /home/alexx/Projects/VOTSTracker/workspace2026/results/
.venv/bin/python check_results_integrity.py --tracker dam4sam_l_cloud   # re-verify locally
```
Then Claude packs (`vot pack`) and prepares submission **S2**. Don't kill the instance until
results are verified locally; after that, destroy it (avoid idle billing).

## Notes
- Throughput estimate: B+ on the 3080 did 170 seqs in ~36-48h; L is ~2-2.5× heavier, a 4090 is
  ~2× faster → ~1.5-2 days on one instance, ~1 day split across two. Start it TODAY.
- The run is resumable (`run_eval_170.py` skips finished objects) — if the instance dies,
  re-run `cloud_run_l.sh` on a new one after re-uploading + re-downloading (or snapshot the volume).
- Everything lives under `/home/alexx/Projects/VOTSTracker` on purpose — scripts hardcode it.
