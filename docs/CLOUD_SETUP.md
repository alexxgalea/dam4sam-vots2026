# Running the full evaluation on rented cloud GPUs

The SAM2.1-Large runs over the full VOTS2026 sequence set (~26 GB, ~300k frames) are impractical
on a single 12 GB local GPU. This is the workflow used for the official submissions, on rented
RTX 4090 instances (e.g. vast.ai), including the durability tooling that survived instance
preemptions and a GPU fault.

## 1. Upload the code

```bash
rsync -az -e "ssh -p <PORT>" --exclude '.venv' --exclude 'workspace2026/sequences' \
  --exclude '**/checkpoints' --exclude 'archive' \
  ./ root@<HOST>:<REMOTE_CLONE_PATH>/
```

Keep the remote clone path identical to the local one — the rescue puller mirrors
`workspace2026/results/` between the two by path.

## 2. Environment, weights, dataset (on the instance)

```bash
bash scripts/cloud/cloud_setup.sh
```

This installs a pinned Python 3.10 environment (torch 2.5.1 / cu121, vot-toolkit 0.8.0),
downloads the SAM2.1-Large checkpoint, pulls the dataset via `vot initialize vots2026/main`,
and runs a CUDA sanity check.

## 3. Smoke test, then launch

```bash
bash scripts/cloud/cloud_run_l.sh smoke   # 1 shortest sequence end-to-end + integrity check
bash scripts/cloud/cloud_run_l.sh full    # all sequences, resumable, with a watcher log
```

The smoke test is mandatory: it validates the whole chain (tracker → masks → readable VOT
trajectories) on one sequence before committing GPU-days. For a two-instance split use
`half1` / `half2`, or `cloud_run_ld2.sh <seqs_file>` with an explicit sequence list.

## 4. Start the rescue puller (locally, immediately)

Rented instances die: budgets run out, GPUs fault. The cloud box cannot push to a home machine
behind NAT, so durability lives on the local side — an incremental `rsync` mirror of the results
directory:

```bash
nohup bash scripts/cloud/pull_cloud_results.sh <PORT> <HOST> dam4sam_l_cloud 180 \
  > pull_l_cloud.log 2>&1 &
```

A dead instance then costs at most one interval of work. This saved two runs: both 4090 boxes
died at budget exhaustion with 165/170 sequences already mirrored locally; the remainder was
completed on the local GPU thanks to the resumable runner.

## 5. Verify and pack

```bash
python scripts/check_results_integrity.py --workspace workspace2026 --tracker <name>
vot pack --workspace workspace2026 <name>
```

Never pack without the integrity check: `vot evaluate --persist` silently tolerates failed
sequences, and `rsync --partial` can leave truncated trajectory files — both were caught by this
check before submission.
