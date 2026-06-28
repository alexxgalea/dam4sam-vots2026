# VOT2026 Tracker Development — Progress Report
**Date:** 2026-05-23  
**Author:** Alexandru Galea  
**Project:** Entry for the VOT2026 (Visual Object Tracking) Challenge  
**Hardware:** NVIDIA RTX 3080 12 GB, WSL2 (Windows Subsystem for Linux), Python 3.12.13  

---

## 1. What Are We Trying to Do?

The **VOT Challenge** (Visual Object Tracking) is an international computer vision competition where participants submit automated systems that can track one or more objects through a video sequence — for example, following a specific person, car, or animal frame by frame, even when it partially disappears or changes appearance.

Our goal is to submit a competitive entry to **VOT2026** by the deadline of **~17 June 2026**.

### Our strategy: build on top of an existing strong tracker

Rather than building from scratch, we are adapting **DAM4SAM** (Distractor-Aware Memory for SAM), a state-of-the-art tracker that won VOTS2024. It uses Meta's **SAM2** (Segment Anything Model 2) as its backbone.

Our plan has two layers:
- **Layer 1 (Baseline):** Run the original DAM4SAM+SAM2.1 on VOT2026 data and submit those results.
- **Layer 3 (Improved):** Swap SAM2.1 for the newer **SAM3** backbone, which promises better accuracy. If this works, it becomes our main submission.

---

## 2. Project Timeline (4-Week Sprint)

| Week | Goal | Status |
|------|------|--------|
| W1 (May 19–25) | Environment setup, baseline eval, SAM3 recon | ✅ In progress |
| W2 (May 26 – Jun 1) | Run SAM3 evaluation on all 170 sequences | Pending |
| W3 (Jun 2–8) | Tune DRM thresholds, hyperparameter sweep | Pending |
| W4 (Jun 9–17) | Final pack, writeup, submit | Pending |

---

## 3. Environment Setup

### Tools installed
- **Python 3.12.13** via `uv` (a fast Python package manager)
- **vot-toolkit 0.8.0** — the official VOT challenge evaluation framework
- **VOT2026 dataset** — 170 video sequences, ~26 GB on disk
- **Multiple Python virtual environments** to keep dependencies isolated:
  - `.venv` — main environment with vot-toolkit
  - `dam4sam/.venv` — DAM4SAM + SAM2.1 dependencies
  - `sam3_venv` — SAM3 dependencies (separate because SAM3 needs different versions)

### Dataset
The VOT2026 dataset lives at `/home/alexx/Projects/VOTSTracker/workspace2026/`. It contains 170 sequences ranging from short clips (a few hundred frames) to very long ones — the longest is `carchase` with **10,699 frames**.

---

## 4. Understanding the Code Architecture

### What is DAM4SAM?
DAM4SAM stands for **Distractor-Aware Memory for SAM**. It tracks objects by:
1. Initialising with a mask (a black-and-white silhouette of the object) on frame 0.
2. Using SAM2's "video predictor" to propagate the mask through subsequent frames.
3. Maintaining a **memory bank** of past frames so the model can recognise the object even if something similar (a "distractor") appears nearby.

### What is DRM?
**DRM** = Distractor-Responsive Memory. When the tracker is confident it has found the object cleanly (good score, stable size), it promotes that frame into the "conditioning memory" — essentially telling the model: "remember this frame, it's a reliable reference."

### Key files
```
VOTSTracker/
├── dam4sam/
│   ├── dam4sam_tracker.py          # Original DAM4SAM tracker (SAM2 backbone)
│   ├── dam4sam_tracker_sam3.py     # NEW: DAM4SAM tracker (SAM3 backbone)
│   ├── vot_wrapper_multiobject.py  # VOT wrapper for SAM2 version
│   └── vot_wrapper_sam3.py         # NEW: VOT wrapper for SAM3 version
├── sam3_upstream/
│   ├── sam3/                       # SAM3 model source code
│   └── checkpoints/sam3.pt        # 3.3 GB checkpoint (downloaded from HuggingFace)
└── workspace2026/
    ├── config.yaml                 # Points to VOT2026 dataset
    ├── trackers.ini                # Defines our tracker configurations
    ├── sequences/                  # 170 video sequences
    └── results/                    # Evaluation output (one folder per sequence)
```

---

## 5. Phase 1: Getting DAM4SAM Running (Before This Sprint)

The original DAM4SAM code was designed to run offline (process an entire video at once). The VOT challenge requires **online** operation — the tracker must process frames one at a time, receiving each frame from the VOT framework and immediately returning its prediction.

The VOT framework uses a communication protocol called **TRAX** (Tracking API eXchange). The tracker runs as a subprocess and exchanges messages with the framework via stdin/stdout pipes.

This was set up and working before the current sprint began.

---

## 6. Phase 2: Switching to VOT2026 Data

VOT releases a new dataset each year. We needed to switch from VOT2025 to VOT2026:
- Updated `config.yaml` to point to the new dataset (`stack: vots2026/main`)
- Downloaded the 170-sequence VOT2026 dataset (~26 GB)
- Verified the tracker still worked with the new data

---

## 7. Phase 3 Week 1 — Key Work

### 7a. DRM Memory Bank Cap (Bug Fix)

**Problem discovered:** On two very long sequences (`hat-5` with 8,000 frames and `robot-19` with 5,797 frames), the tracker was running out of GPU memory (Out-Of-Memory crash) near the end of the sequence.

**Root cause:** The DRM memory bank has no size limit. Over thousands of frames, the tracker was accumulating hundreds of "conditioning frames" in GPU memory, each holding feature tensors. After enough frames, this exceeded the 12 GB GPU limit.

**Fix:** Added a cap of **30 DRM frames** maximum. When a new frame is added to DRM and the cap is reached, the oldest non-initial frame is evicted (removed) first. The initial frame (frame 0) is always preserved.

```python
DRM_MAX_FRAMES = 30   # never accumulate more than 30 conditioning frames

def _evict_oldest_drm_frame(self, inference_state, obj_idx):
    # removes the oldest non-initial DRM frame to free memory
    ...
```

This was added directly to SAM2's video predictor and also ported to SAM3.

---

### 7b. SAM3 API Reconnaissance

Before committing to porting DAM4SAM to SAM3, we did a detailed technical investigation to check compatibility. The findings were written up in `SAM3_API_NOTES.md`.

**Key findings:**

| Question | Answer |
|----------|--------|
| Does SAM3 have the same `inference_state` structure as SAM2? | ✅ Yes — almost identical |
| Does SAM3 have `add_new_mask()`, `add_new_points()`? | ✅ Yes |
| Does SAM3 have `add_to_drm()`? | ❌ No — we must port it |
| Does SAM3's `propagate_in_video` return the same values? | ⚠️ No — returns **5 values** instead of 3 (adds `obj_scores`) |
| Can SAM3 run on our PyTorch 2.5.1 + CUDA 12.1? | ✅ Yes (no Flash Attention 3 needed) |
| Is the checkpoint available? | ⚠️ Gated on HuggingFace (requires access request) |

**Verdict: GO** — the port is feasible, ~2–3 hours of work.

---

### 7c. SAM3 Checkpoint Download

SAM3's weights are hosted on HuggingFace behind an access gate (you must request permission from Meta/Facebook Research). We submitted the request and it was approved the same day.

The checkpoint (`sam3.pt`) is **3.3 GB** and was downloaded using the HuggingFace CLI.

**Important:** The checkpoint contains the full model (tracker + detector + backbone). Our standalone tracker only needs a subset of the weights. We wrote a custom loader that remaps the keys:

```
Full checkpoint structure:          Standalone tracker needs:
  tracker.*              →           * (direct)
  detector.backbone.*    →           backbone.* (rename prefix)
  detector.head.*        →           (skip — not needed)
```

This loads exactly **773 weight tensors** with no missing or unexpected keys.

---

### 7d. Porting DAM4SAM to SAM3 — The New Files

We created two new files:

#### `dam4sam_tracker_sam3.py`
This is the core tracker class. Key differences from the SAM2 version:
- Uses `build_tracker(with_backbone=True)` instead of SAM2's video predictor
- `propagate_in_video` yields **5 values** (we unpack all 5)
- DRM trigger uses **object presence score** (`sigmoid(obj_scores) > 0.85`) instead of SAM2's alternative-mask IoU check (SAM3 doesn't expose alternative masks)
- Must explicitly call `propagate_in_video_preflight()` after initialisation (SAM2 did this automatically)
- Images stored as `[C, H, W]` tensors (SAM3 adds `.unsqueeze(0)` internally)

#### `vot_wrapper_sam3.py`
This is the VOT protocol wrapper. It:
- Reads the initial masks from the VOT framework
- Loads SAM3 **once** and shares it across all objects in the sequence (saves GPU memory)
- Runs the tracking loop, reporting masks back each frame

---

### 7e. Bugs Encountered and Fixed During SAM3 Port

This section documents every error we hit and how we resolved it. These are listed in the order they were encountered.

---

#### Bug 1: `UnboundLocalError: cannot access local variable 'torch'`

**What happened:** The tracker crashed immediately on import.

**Why:** Python's scoping rules. We had `import torch` inside an `else` block near the bottom of `__init__`. Even though it was never reached, Python's static analysis saw `torch` as a *local* variable in the whole function — so `torch.tensor(...)` earlier in the function failed because Python expected a local `torch` that hadn't been assigned yet.

**Fix:** Removed the inner `import torch` (it was already imported at the top of the file).

---

#### Bug 2: `AttributeError: 'Sam3TrackerPredictor' object has no attribute 'reset_state'`

**What happened:** Our code called `self.predictor.reset_state(...)` during initialisation.

**Why:** SAM3 doesn't have a `reset_state` method. In our SAM2 code, we called it to clear state between sequences. But in our SAM3 code, we already create a **fresh** `inference_state` for each sequence (via `init_state_tw_sam3()`), so there's nothing to reset.

**Fix:** Removed the `reset_state` call.

---

#### Bug 3: 309 missing keys / 1465 unexpected keys when loading checkpoint

**What happened:** Loading `sam3.pt` with `strict=True` gave hundreds of key mismatches.

**Why:** The checkpoint stores the *full* model (tracker + detector + backbone) but our `build_tracker()` creates only the tracker portion. The checkpoint keys look like:
- `tracker.memory_attention.layers.0.weight`
- `detector.backbone.vision_backbone.trunk.blocks.0.weight`
- `detector.head.transformer.layers.0.weight`

Our standalone tracker expects keys without any prefix (e.g., `memory_attention.layers.0.weight`).

**Fix:** Wrote `_load_sam3_tracker_checkpoint()` which remaps keys before loading:
- `tracker.X` → `X` (strip prefix)
- `detector.backbone.vision_backbone.X` → `backbone.vision_backbone.X` (rename prefix)
- Everything else → skip

Result: exact 773/773 key match.

---

#### Bug 4: `RuntimeError: Image features for frame 0 are not cached`

**What happened:** During initialisation, when we called `_get_image_feature()`, the model said it couldn't find the backbone features for frame 0.

**Why:** We initially called `build_tracker(with_backbone=False)` — a mode where the backbone runs separately. Without the backbone, the tracker cannot compute its own image features.

**Fix:** Switched to `build_tracker(with_backbone=True)` so the tracker is self-contained and can compute backbone features from raw images.

---

#### Bug 5: `RuntimeError: Expected 3D or 4D input to conv2d, got [1,1,3,1008,1008]`

**What happened:** During `_get_image_feature()`, PyTorch complained about a 5-dimensional tensor when the convolution expected 4D.

**Why:** We were storing images as `[1, C, H, W]` (already batch-expanded with `.unsqueeze(0)`), then SAM3's `_get_image_feature()` was calling `.unsqueeze(0)` *again*, producing `[1, 1, C, H, W]` — one dimension too many.

**Fix:** Store images as `[C, H, W]` (without the batch dimension). SAM3 adds the batch dimension internally.

---

#### Bug 6: `RuntimeError: No points are provided; please add points first`

**What happened:** On the first tracking frame, `propagate_in_video` complained that there was no annotation.

**Why:** SAM2's `propagate_in_video` automatically runs a "preflight" step at the start that consolidates the initialisation mask into the memory bank. SAM3 does **not** do this automatically — it requires an explicit call to `propagate_in_video_preflight()`.

Without preflight, the init mask sat in a temporary buffer (`temp_output_dict_per_obj`) and never made it into `output_dict["cond_frame_outputs"]`, so the tracker didn't know it had been initialised.

**Fix:** Added an explicit call in `initialize()`:
```python
self.predictor.propagate_in_video_preflight(
    self.inference_state, run_mem_encoder=True
)
```

---

#### Bug 7: `TypeError: Sam3TrackerPredictor.propagate_in_video() missing 1 required positional argument: 'reverse'`

**What happened:** Calling `propagate_in_video()` failed with a missing argument error.

**Why:** SAM3's `propagate_in_video` requires `reverse` as a **positional** argument (not optional), whereas SAM2 made it optional.

**Fix:** Added `reverse=False` explicitly to the call.

---

#### Bug 8: `vot test dam4sam_sam3` — "Exception when waiting for response: Unknown" (most complex bug)

**What happened:** Running the official VOT test (`vot test dam4sam_sam3`) passed initialisation and frame 1, but then died with the error above. Exit code was 0 (clean exit), no Python exception was raised, no segfault.

**Debugging journey:**

1. Added try/except blocks around every call — no exception was caught.
2. Added faulthandler to detect C-level segfaults — no output.
3. Added `except BaseException` (catches even `SystemExit`) — still nothing.
4. Added fine-grained print statements inside the tracking loop — process died in the middle with no error.
5. Redirected all prints to stderr (suspecting stdout pipe corruption) — still died.
6. Added file-based logging (`/tmp/sam3_debug.log`) that bypasses stdout entirely.

**The log revealed:** All 3 trackers completed their `track()` call successfully, `handle.report()` returned OK, **and then** the process exited cleanly on the *second* call to `handle.frame()`.

**Root cause (finally found):** The VOT toolkit launches the tracker subprocess with:
```python
subprocess.Popen(..., stderr=subprocess.STDOUT)
```
This **merges stderr into stdout**. The TRAX protocol uses stdout as a communication pipe. SAM3's `propagate_in_video` uses `tqdm` (a Python progress bar library) which writes carriage-return characters (`\r`) to stderr to update the progress display in-place. 

With `stderr=STDOUT`, these `\r` characters went into the TRAX protocol pipe. The TRAX library reads the pipe line-by-line looking for `@@TRAX:state` messages. The `\r` characters caused partial tqdm updates to be written to the pipe *between* state messages — they got mixed onto the same "line" as protocol messages, making the protocol parser see garbage data instead of valid states. The host-side TRAX library then raised "Exception when waiting for response: Unknown" and sent `@@TRAX:quit` to the tracker.

**Fix:** One argument:
```python
for out in self.predictor.propagate_in_video(
    ...,
    tqdm_disable=True,   # ← prevents \r chars from corrupting the TRAX pipe
):
```

**Why this wasn't obvious:** The existing `dam4sam_b` (SAM2 version) also has print statements and works fine — because SAM2's `propagate_in_video` doesn't use tqdm by default, so there were no `\r` characters in its output.

**After this fix:** `vot test dam4sam_sam3` passed all 49 frames with 3 objects. ✅

---

### 7f. `torch.cuda.empty_cache()` Removal (Performance Optimisation)

**Problem:** The original DAM4SAM code called `torch.cuda.empty_cache()` at the start of every single frame's `track()` call. This function forces the CUDA runtime to:
1. Synchronise all pending GPU operations (a blocking operation)
2. Return unused memory back to the OS

**Why it's wasteful:** During a video sequence, the GPU memory usage pattern is stable — the same amount of memory is used every frame. There's nothing to "empty." The call was likely added during debugging to prevent out-of-memory errors and was never removed.

**Cost:** At ~50–100 ms per call, over 288,000 remaining frames:
```
288,000 frames × 75 ms = 21,600 seconds = 6 hours wasted
```

**Fix:** Removed `torch.cuda.empty_cache()` from both `dam4sam_tracker.py` and `dam4sam_tracker_sam3.py`.

**Note:** OOM risk is low because the DRM cap (max 30 frames) was already implemented to bound memory usage on long sequences.

---

## 8. Current Status

### Evaluation in progress
The DAM4SAM+SAM2.1 baseline evaluation is currently running:
- **19 sequences completed** (out of 170) from a previous session
- **Currently processing:** `bus` sequence (8,087 frames) — the longest after `carchase`
- **Estimated time remaining:** ~24 hours (after removing `empty_cache()`)
- **Process:** PID 307754, running in background, log at `/tmp/vot_baseline_eval.log`
- Results are saved per-sequence, so if interrupted it resumes from where it left off

### SAM3 tracker
- `vot test dam4sam_sam3` ✅ passes all 49 test frames with 3 objects
- Ready to run `vot evaluate dam4sam_sam3` once the baseline eval finishes (or in parallel if memory allows — unlikely on 12 GB VRAM)

---

## 9. What's Left

| Task | Notes |
|------|-------|
| Wait for baseline eval to finish | ~24 hours, running in background |
| `vot pack dam4sam_b` | Creates the submission zip |
| Submit Layer 1 to VOT2026 server | The baseline SAM2.1 result |
| Run `vot evaluate dam4sam_sam3` | ~28 hours, will start after baseline |
| Decide go/no-go for SAM3 | Compare scores; if SAM3 ≥ SAM2.1, use it as primary submission |
| Hyperparameter tuning | DRM thresholds (`DRM_SCORE_THRESH`, `DRM_SIZE_LOW/HIGH`, `DRM_MIN_INTERVAL`) |
| Final pack + writeup + submit | Before June 17 |

---

## 10. Summary of Files Changed / Created

| File | Change |
|------|--------|
| `dam4sam/dam4sam_tracker.py` | Removed `torch.cuda.empty_cache()` per-frame; added DRM cap (30 frames) + eviction |
| `dam4sam/dam4sam_tracker_sam3.py` | **New file** — full DAM4SAM tracker using SAM3 backbone |
| `dam4sam/vot_wrapper_sam3.py` | **New file** — VOT protocol wrapper for SAM3 |
| `sam3_upstream/sam3/model/sam3_tracking_predictor.py` | Added `add_to_drm()`, `_evict_oldest_drm_frame()`, `adds_in_drm_per_obj` injection |
| `workspace2026/trackers.ini` | Added `dam4sam_sam3` tracker entry |
| `workspace2026/config.yaml` | Switched stack to `vots2026/main` |
| `SAM3_API_NOTES.md` | **New file** — compatibility analysis, go/no-go decision document |

---

*Report generated: 2026-05-23*
