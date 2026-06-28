# Phase 2 — DAM4SAM reproduction on VOTS2025

**Executor:** Sonnet
**Owner:** alexx
**Target completion:** end of week of 2026-05-27
**Working dir:** `/home/alexx/Projects/VOTSTracker`
**Host:** WSL2, RTX 3080 **10 GB** VRAM (nvidia-smi confirmed — user said 12 GB, actual is 10 GB)

## Goal

Clone DAM4SAM, set up its Python environment, and run it through `vot evaluate` / `vot analysis` on the VOTS2025 workspace established in Phase 1. Phase 2 is done when the analysis JSON contains real quality/accuracy/robustness scores for DAM4SAM — numbers will be lower than SOTA but must not be error values. **No architecture changes in this phase.**

## Key facts established by pre-research

- `vot_wrapper_dam4sam.py` already uses `vot.VOT("mask", multiobject=True)` — the VOTS multiobject mask protocol is mostly wired up.
- **Known bug in the wrapper**: `status = [outputs['pred_mask']]` wraps all objects into one list element. For multiobject VOTS this is wrong — needs fixing (see Task 6).
- Model variants: `sam21pp-T` (tiny), `sam21pp-S` (small), `sam21pp-B` (base+), `sam21pp-L` (large). Wrapper hardcodes `sam21pp-L`.
- **VRAM budget: use `sam21pp-B` (base+)**. SAM2.1-L needs ~10-11 GB; we have exactly 10 GB and need headroom for memory bank. SAM2.1-B+ needs ~7-8 GB — safe. Upgrade to L only if B+ evaluation is complete and you confirm GPU headroom with `nvidia-smi`.
- DAM4SAM's `requirements.txt` pins `vot-toolkit==0.7.1`. Our workspace uses `0.8.0`. These must stay separate — DAM4SAM gets its own venv.
- No conda. Use `uv` with Python 3.10, consistent with this project's pattern.

## Preconditions to verify first

1. `nvidia-smi` — confirm GPU visible and ~10 GB free VRAM.
2. Check CUDA version: `nvidia-smi | grep "CUDA Version"` — need 12.1+ for torch 2.1.0+cu121. Driver 596.36 (seen previously) supports this.
3. `uv --version` — should be present at `~/.local/bin/uv`.
4. `df -h /home/alexx` — need ≥15 GB free beyond the 27 GB already used by sequences (checkpoints + DAM4SAM repo + venv = ~8 GB).
5. Keep-awake running: `ps aux | grep powershell | grep -v grep`. If not, start it:
   ```bash
   powershell.exe -ExecutionPolicy Bypass -File /home/alexx/Projects/VOTSTracker/keep_awake.ps1 &
   ```

## Tasks

### 1. Clone DAM4SAM

```bash
cd /home/alexx/Projects/VOTSTracker
git clone https://github.com/jovanavidenovic/DAM4SAM.git dam4sam
```

Verify: `ls dam4sam/` shows `dam4sam_tracker.py`, `vot_wrapper_dam4sam.py`, `requirements.txt`, `checkpoints/`, `sam2/`.

### 2. Create Python 3.10 venv for DAM4SAM

```bash
uv venv --python 3.10 dam4sam/.venv
```

Verify: `dam4sam/.venv/bin/python --version` shows Python 3.10.x.

### 3. Install PyTorch with CUDA 12.1

```bash
uv pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121 --python dam4sam/.venv
```

**Note:** The repo's README says torch==2.1.0 but SAM2.1 (released Sep 2024) requires torch ≥2.3.1 for its memory-efficient attention. Use torch 2.5.1 which is stable with CUDA 12.1 and SAM2.1. If this causes import errors in SAM2, fall back to 2.3.1 — stop and report rather than guessing.

Verify:
```bash
dam4sam/.venv/bin/python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```
Must print a version and `True`.

### 4. Install DAM4SAM dependencies

DAM4SAM's `requirements.txt` includes many notebook/Jupyter deps we don't need and pins `vot-toolkit==0.7.1` (incompatible with our workspace). Install a filtered set:

```bash
uv pip install --python dam4sam/.venv \
  numpy==1.26.4 pillow opencv-python scipy matplotlib tqdm \
  hydra-core omegaconf iopath \
  pyyaml attrs
```

Then install SAM2 from the repo's bundled copy (it's a git submodule or vendored copy):
```bash
cd /home/alexx/Projects/VOTSTracker/dam4sam
dam4sam/.venv/bin/pip install -e sam2/
```

Wait — check first if `sam2/` is a proper Python package:
```bash
ls dam4sam/sam2/
```
If it contains `setup.py` or `pyproject.toml`, run the above. If it's just source files, it will be importable via PYTHONPATH instead (Task 7 handles this).

Build the SAM2 C++ extension (required for mask decoding):
```bash
cd /home/alexx/Projects/VOTSTracker/dam4sam
dam4sam/.venv/bin/python setup.py build_ext --inplace 2>&1
```

Verify: `ls dam4sam/*.so 2>/dev/null || ls dam4sam/sam2/*.so 2>/dev/null` — should show a compiled `.so` file.

**If build fails**: check that gcc/g++ is installed (`which gcc`). If missing: `sudo apt install build-essential`. Report back rather than continuing with broken build.

### 5. Download checkpoints

Download only SAM2.1-B+ and SAM2.1-L (skip tiny/small to save time — we need B+ for evaluation, L only as a stretch goal):

```bash
cd /home/alexx/Projects/VOTSTracker/dam4sam/checkpoints
wget https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt
wget https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt
```

These are ~323 MB and ~898 MB respectively. Run in background and monitor for stalls (same pattern as Phase 1 — check modification time, not just process alive).

Verify: `ls -lh dam4sam/checkpoints/*.pt` shows both files with expected sizes (~323 MB and ~898 MB).

### 6. Write the VOTS2025 wrapper

The existing `vot_wrapper_dam4sam.py` has a bug for multiobject: it does `status = [outputs['pred_mask']]` which always returns one mask. For VOTS2025 we need one mask per tracked object.

First, read `dam4sam_tracker.py`'s `track()` method to understand what `outputs['pred_mask']` actually returns (single mask or list). Then:

**Option A** — if `pred_mask` is already a list of per-object masks:
```python
status = outputs['pred_mask']  # already a list
handle.report(status)
```

**Option B** — if `pred_mask` is a single merged mask (all objects in one array with different pixel values):
Need to split by object ID. Look at how `get_vot_mask()` merges masks to understand the inverse operation.

Create `/home/alexx/Projects/VOTSTracker/dam4sam/vot_wrapper_vots2025.py` as a copy of `vot_wrapper_dam4sam.py` with:
1. `tracker_name` changed from `"sam21pp-L"` to `"sam21pp-B"` 
2. The `status` output bug fixed based on whichever option above applies
3. `@torch.cuda.amp.autocast()` kept (needed for VRAM efficiency)

Do not modify the original `vot_wrapper_dam4sam.py`.

**Stop and ask** if the `track()` output format is ambiguous — this fix is load-bearing.

### 7. Register DAM4SAM in trackers.ini

Append to `/home/alexx/Projects/VOTSTracker/workspace/trackers.ini`:

```ini
[dam4sam_b]
label = DAM4SAM (SAM2.1-B+)
protocol = traxpython
command = vot_wrapper_vots2025
paths = /home/alexx/Projects/VOTSTracker/dam4sam
python = /home/alexx/Projects/VOTSTracker/dam4sam/.venv/bin/python3
```

The `paths` field adds the dam4sam root to PYTHONPATH, making all DAM4SAM modules importable.

Verify registration: from workspace dir:
```bash
/home/alexx/Projects/VOTSTracker/.venv/bin/vot test dam4sam_b
```
This runs on a short synthetic sequence — fast sanity check before committing to the full evaluate. Watch for:
- CUDA OOM → model too large, switch to `sam21pp-B` if not already, or try `sam21pp-S`
- Import errors → PYTHONPATH issue or missing dep
- `vot.VOT` API mismatch → vot_helper compatibility issue with vot-toolkit 0.8.0 (see note below)

**vot_helper compatibility note**: DAM4SAM's `utils/vot_helper.py` is a bundled copy of the old VOT Python helper. It may conflict with vot-toolkit 0.8.0's trax layer. If `vot test` fails with trax protocol errors, replace the `import utils.vot_helper as vot` + `vot.VOT(...)` calls in the wrapper with the direct trax API used in Phase 1's dummy tracker (`from trax import Image, Region, Server, TraxStatus`). The logic translation is straightforward — stop and ask if unclear.

### 8. Run evaluate + analysis

```bash
cd /home/alexx/Projects/VOTSTracker/workspace
/home/alexx/Projects/VOTSTracker/.venv/bin/vot evaluate dam4sam_b
```

This will take hours — DAM4SAM runs GPU inference on 144 sequences. Keep-awake must be running. Monitor every 15 minutes:
- `nvidia-smi` — watch VRAM and GPU utilization (should be 80-100%)
- `ls -lt workspace/results/dam4sam_b/ | head -3` — confirm results are being written
- If GPU utilization drops to 0% for >10 min with process alive → stall, restart

When evaluate completes:
```bash
/home/alexx/Projects/VOTSTracker/.venv/bin/vot analysis dam4sam_b
```

### 9. Record results

Create `/home/alexx/Projects/VOTSTracker/PHASE2_NOTES.md` with:
- Model variant used (`sam21pp-B` or fallback)
- Peak VRAM usage (from nvidia-smi during run)
- Wall-clock time for evaluate
- Any deviations from this plan
- Analysis report path and key metric values (quality/accuracy/robustness scores from the JSON)
- Any wrapper fixes made and why

## Done criteria

- [ ] `dam4sam/` cloned and dependencies installed
- [ ] SAM2.1-B+ checkpoint present
- [ ] `vot test dam4sam_b` passes on synthetic sequence
- [ ] `vot evaluate dam4sam_b` completes without unhandled errors
- [ ] `vot analysis dam4sam_b` produces report with real scores
- [ ] `PHASE2_NOTES.md` written

## Out of scope

- Don't train or fine-tune anything.
- Don't try SAM2.1-L unless B+ evaluation is complete and VRAM headroom is confirmed.
- Don't start Phase 3 SAM3 work — Phase 2 is baseline reproduction only.
- Don't install conda — use uv.

## When to stop and ask

- CUDA not available after torch install.
- SAM2 C++ build fails.
- `vot test` crashes with a non-VRAM error that doesn't resolve with the vot_helper fix.
- `pred_mask` output format from `track()` is unclear — wrong fix here breaks all of Phase 2.
- OOM even with `sam21pp-B` (try `sam21pp-S` first but confirm with alexx).
- Evaluate crashes on >2 sequences with the same error class.
