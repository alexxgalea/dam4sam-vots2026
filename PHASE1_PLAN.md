# Phase 1 — VOTS2025 toolkit + dummy tracker end-to-end

**Executor:** Sonnet
**Owner:** alexx
**Target completion:** end of week of 2026-05-20
**Working dir:** `/home/alexx/Projects/VOTSTracker`
**Host:** WSL2, RTX 3080 12GB (GPU not needed for this phase)

## Goal

Stand up a working `vot-toolkit-python` workspace for the **vots2025** stack and push a trivial "dummy" tracker through the full `vot evaluate` → `vot analysis` pipeline. Phase 1 is done when an analysis report is generated against real VOTS2025 sequences, even with terrible scores. **No tracking quality work in this phase.**

## Why this matters

The VOTS2025 challenge submission requires a `vot pack` artifact (Phase 4). Every later phase (DAM4SAM reproduction, SAM3 swap) plugs into the same toolkit harness, so getting the harness solid now de-risks weeks 2-4. The dummy tracker also serves as a reference template for the real tracker integration.

## Preconditions to verify first

Before running commands, confirm (and report back if any fail):

1. `python3 --version` — need 3.10 or 3.11. **Avoid 3.12+** until you've confirmed vot-toolkit-python compatibility on PyPI; if only 3.12 is available, check the latest vot-toolkit release notes before proceeding.
2. `df -h /home/alexx` — VOTS2025 sequences are several GB. Need ≥20 GB free for sequences + results.
3. `which git curl wget` — all should resolve.
4. Working directory `/home/alexx/Projects/VOTSTracker` exists and is empty (it is, as of plan creation).

If any precondition fails, **stop and ask** rather than improvising.

## Tasks

### 1. Python environment

Create an isolated venv at the project root.

```bash
cd /home/alexx/Projects/VOTSTracker
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
```

Verify: `which python` shows `.venv/bin/python`.

### 2. Install vot-toolkit-python

```bash
pip install vot-toolkit
```

Verify:
- `vot --help` lists subcommands including `initialize`, `evaluate`, `analysis`, `pack`.
- `pip show vot-toolkit` shows a version (record it in `PHASE1_NOTES.md` — see step 7).

**If install fails** on a transitive dep (common culprits: `numba`, `opencv-python-headless`, `shapely`), stop and report the error. Do not try random workarounds.

### 3. Initialize the vots2025 workspace

Create a workspace subdirectory so the venv and workspace are siblings.

```bash
mkdir -p /home/alexx/Projects/VOTSTracker/workspace
cd /home/alexx/Projects/VOTSTracker/workspace
vot initialize vots2025
```

This will:
- Download the VOTS2025 stack definition
- Fetch sequences (several GB — may take a while)
- Create `sequences/`, `trackers.ini`, `config.yaml` (or similar)

Verify:
- `ls sequences/` shows multiple sequence directories
- `cat trackers.ini` exists (likely empty or with examples)
- `sequences/list.txt` (or stack equivalent) lists all sequences

**If the stack name `vots2025` is rejected**, run `vot initialize --help` and `vot stacks` (or check the toolkit docs) to find the correct stack identifier. Common alternatives: `vots2024`, `vot2024`. Confirm with alexx before falling back to a different year.

### 4. Build the dummy tracker

VOTS is a **segmentation** challenge with **multi-object** sequences. The dummy must emit masks and handle multiple objects per sequence. Simplest possible behavior: return the initialization masks unchanged for every frame.

Create `/home/alexx/Projects/VOTSTracker/dummy_tracker/tracker.py`:

```python
import vot

handle = vot.VOT("mask", multiobject=True)
objects = handle.objects()
imagefile = handle.frame()
if not imagefile:
    raise SystemExit

while True:
    imagefile = handle.frame()
    if not imagefile:
        break
    handle.report(objects)
```

Notes for the executor:
- The `vot` Python module is provided by vot-toolkit's trax integration. If `import vot` fails at runtime, you may need to `pip install vot-trax` separately (toolkit version dependent) — check before patching the script.
- Do not attempt to "improve" the dummy. Returning init masks unchanged is the entire point.

### 5. Register the dummy in `trackers.ini`

Append to `/home/alexx/Projects/VOTSTracker/workspace/trackers.ini`:

```ini
[dummy]
label = Dummy
protocol = traxpython
command = tracker
paths = /home/alexx/Projects/VOTSTracker/dummy_tracker
env_PATH = <PATH>;${PATH}
```

The exact field names depend on toolkit version. **Confirm by running `vot initialize`'s generated example** if any field is rejected — toolkit ships a commented example block in fresh `trackers.ini`.

### 6. Run evaluate + analysis

From the workspace dir:

```bash
cd /home/alexx/Projects/VOTSTracker/workspace
vot evaluate --workspace . dummy
```

Expect this to take a while (CPU-only, but iterates every sequence × every object). Watch for:
- Any `trax` protocol errors → tracker script bug; stop and inspect logs.
- Crash on a specific sequence → record sequence name, continue if `--retry` or skip flag available, otherwise stop and report.

When evaluate completes:

```bash
vot analysis --workspace . dummy
```

Verify:
- `results/dummy/` (or similar) contains per-sequence outputs.
- `analysis/` directory contains a report (HTML, PDF, or JSON).
- Report shows non-error scores for **quality / accuracy / robustness** (numbers will be bad — that's fine).

### 7. Record what you did

Create `/home/alexx/Projects/VOTSTracker/PHASE1_NOTES.md` with:
- vot-toolkit version installed
- Python version used
- Stack name actually used (e.g. `vots2025` or fallback)
- Disk used by `sequences/` (`du -sh sequences/`)
- Wall-clock time for `vot evaluate dummy`
- Any deviations from this plan and why
- Final analysis report path

## Done criteria

All must hold:

- [ ] `vot --help` works in the project venv
- [ ] `workspace/sequences/` populated with VOTS sequences
- [ ] `dummy_tracker/tracker.py` exists and is registered in `trackers.ini`
- [ ] `vot evaluate dummy` completes without unhandled errors
- [ ] `vot analysis dummy` produces a report file
- [ ] `PHASE1_NOTES.md` filled in

## Out of scope (do NOT do)

- Don't install PyTorch, SAM2, SAM3, or DAM4SAM — those belong to Phase 2/3.
- Don't tune the dummy tracker to improve scores.
- Don't run `vot pack` — that's Phase 4.
- Don't commit anything to git (no repo exists yet; alexx will decide on repo init later).
- Don't modify this plan file; if a step is wrong, stop and ask.

## When to stop and ask

- Any precondition fails.
- vot-toolkit install fails after a clean retry.
- The `vots2025` stack name is not recognized.
- `trax` / `vot` module import fails at tracker runtime.
- `vot evaluate` crashes on >1 sequence with the same error class.
- Disk fills up.
- Anything else surprising — Phase 1 is plumbing; surprises here usually mean a wrong assumption, not a clever fix.
