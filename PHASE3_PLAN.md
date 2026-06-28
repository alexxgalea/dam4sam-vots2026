# Phase 3 Plan — SAM3 Integration + VOT2026 Submission

**Strategist:** Opus 4.7
**Executor:** Sonnet 4.6 (or as directed)
**Owner:** alexx
**Sprint window:** 2026-05-24 → 2026-06-17 (~3.5 weeks)
**Working dir:** `/home/alexx/Projects/VOTSTracker`
**Host:** WSL2, RTX 3080 10 GB VRAM

---

## Strategic Frame

### What Phase 2 Established
- Working DAM4SAM (SAM2.1-B+) baseline through `vot evaluate` / `vot pack`
- Custom multiobject wrapper (`vot_wrapper_vots2025.py`) with shared-predictor pattern
- vot-toolkit 0.8.0 API patches applied
- Memory offloading enabled (`offload_state_to_cpu = True`)
- **2 failures** (`hat-5`, `robot-19`) from unbounded DRM memory bank growth on long sequences

### What Phase 3 Must Establish
1. **VOT2026 submission pipeline works end-to-end** (dataset → evaluate → pack → server → score)
2. **A real numerical baseline score** on VOT2026 (currently we have *no* validated score)
3. **An improvement direction**: SAM3 integration (high-reward) with DAM4SAM optimization (safe fallback)

### Critical Recalibration Since Sprint Plan Written
- **Novelty anchor is invalid.** VOTS2025_S3-DAM4SAM already exists and scored Q=0.75 in Dec 2025. We are no longer first to combine SAM3 with DAM4SAM. The bet shifts from *novelty* to *quality of execution and possibly architectural twist*.
- **Backbone swap is not a backbone swap.** `add_to_drm` is custom code DAM4SAM authors added to SAM2's video predictor. SAM3 swap = porting DRM machinery onto SAM3's video predictor, not just changing model weights.
- **Target competition is VOT2026, not VOTS2025.** Dataset differs (`vots2026/main` description URL). All evaluation/submission work goes to the live VOT2026 server.

---

## Strategy: Build Three Layers in Parallel Confidence

```
┌────────────────────────────────────────────────────────────┐
│ Layer 3: SAM3 SWAP (high risk, high reward) — Q target: 0.75│
│          ↑ fallback if blocked ↓                           │
│ Layer 2: DAM4SAM-OPTIMIZED (DRM cap, tuning) — Q ~0.72     │
│          ↑ fallback if blocked ↓                           │
│ Layer 1: BASELINE DAM4SAM SUBMITTED to VOT2026 — guaranteed │
└────────────────────────────────────────────────────────────┘
```

At every point in the sprint, we should be able to submit *something* that gives an actual VOT2026 server score. We build upward but never demolish the floor.

---

## Week 1 (May 24-30) — De-risk + Quick Wins

**Objective:** Get a baseline VOT2026 score on the leaderboard, fix the DRM memory bank failure mode, clean up infrastructure.

### Stream A — VOT2026 Pipeline Validation
A1. **Switch workspace stack** in `workspace/config.yaml`:
```yaml
registry: [./trackers.ini]
stack: vots2026/main
```
A2. **Download VOT2026 dataset.** Run `vot evaluate dam4sam_b` from the new workspace; toolkit lazy-downloads on first run. Monitor for stalls (use file-modification-time check, not just process-alive — see [PHASE1_NOTES.md](PHASE1_NOTES.md) for the stall pattern we hit in Phase 1).
A3. **Run DAM4SAM baseline on VOT2026 dataset.** Same wrapper as Phase 2 (`vot_wrapper_vots2025.py` — name it generically: rename to `vot_wrapper_multiobject.py` so it doesn't lie about which competition). Use `--persist` and `timeout = 600`. **Expected runtime: 6-10 hours.** Use tmux: `tmux new-session -d -s vot2026_eval ...`. Background tasks die when Claude session changes.
A4. **Pack and submit.** `vot pack dam4sam_b` → upload to VOT2026 server → record score. This is Layer 1 — our floor.

**Done criteria for Stream A:** Score visible on VOT2026 leaderboard for `dam4sam_b` (baseline).

### Stream B — DRM Memory Bank Cap (fixes Layer 2)
B1. **Audit DRM accumulation.** Locate `add_to_drm()` in `dam4sam/sam2/sam2_video_predictor.py`. Trace what state `add_to_drm` mutates:
  - Does it append to `inference_state["output_dict"]["cond_frame_outputs"]`?
  - Are there other lists/dicts that grow per call?
B2. **Add a configurable max-DRM size** (suggestion: `max_drm_frames = 30`). When the cap is hit, evict the **oldest non-init frame** (always preserve `frame_idx=0`). Implement as a small helper inside `add_to_drm`:
```python
DRM_KEEP_INIT = True
DRM_MAX = 30  # tunable
def _evict_oldest_drm_entry(state, obj_idx):
    cond = state["output_dict_per_obj"][obj_idx]["cond_frame_outputs"]
    # Identify non-init frames, drop oldest
    candidates = [k for k in cond.keys() if k != 0] if DRM_KEEP_INIT else list(cond.keys())
    if candidates:
        del cond[min(candidates)]
```
B3. **Sanity-check on hat-5 + robot-19.** Re-evaluate just those two sequences locally (delete their dummy results first). Confirm they complete within `timeout = 600`.
B4. **Re-evaluate full VOT2026 dataset** with capped DRM. Resubmit. This is Layer 2.

**Done criteria for Stream B:** Long-sequence equivalents on VOT2026 complete without timeout. New score on leaderboard.

### Stream C — SAM3 Repo Reconnaissance
C1. **Clone SAM3.** `git clone https://github.com/facebookresearch/sam3.git` to `/home/alexx/Projects/VOTSTracker/sam3_upstream/`. **Read its README and code structure.** Specifically determine:
  - Does SAM3 have a `video_predictor` analog to `sam2_video_predictor.py`?
  - Are `propagate_in_video`, `add_new_mask`, `init_state` (or equivalents) present?
  - How does its memory mechanism work? Is there a memory bank like SAM2's `output_dict["cond_frame_outputs"]`?
  - Model checkpoint sizes — does a B+ equivalent fit in 10 GB VRAM with a memory bank?
C2. **Write a one-page compatibility report.** Save as `SAM3_API_NOTES.md`. Include:
  - Map of SAM2 → SAM3 API differences relevant to DAM4SAM (one row per method DAM4SAM uses)
  - VRAM estimate for SAM3 + memory bank + 8 objects
  - Estimated porting effort (hours/days)
  - **Go/no-go recommendation for Week 2 SAM3 attempt**

### Stream D — Infrastructure (end of week)
D1. **Move WSL from C: to D:** Standard procedure:
```powershell
# In Windows PowerShell (admin):
wsl --shutdown
wsl --export Ubuntu D:\WSL\ubuntu-backup.tar
wsl --unregister Ubuntu
wsl --import Ubuntu D:\WSL\Ubuntu D:\WSL\ubuntu-backup.tar --version 2
# Set default user back:
ubuntu config --default-user alexx
```
Verify: `wsl --list -v` shows Ubuntu at correct location; `df -h` inside WSL shows the new mount.
D2. **Keep-awake check**: Ensure `keep_awake.ps1` is running during long evaluation runs.

### Week 1 Decision Gate (End of Sat May 30)

After Stream C's report, decide:

| If... | Then... |
|---|---|
| SAM3 has a video predictor with `propagate_in_video`-equivalent and memory bank ✅ | Proceed to Week 2 SAM3 integration (full ambition) |
| SAM3 has video API but radically different memory architecture ⚠️ | Reduce scope: try SAM3 image predictor + custom temporal logic, or stay with DAM4SAM-optimized |
| SAM3 is image-only or doesn't open-source video weights ❌ | **Pivot:** drop SAM3 swap, spend Weeks 2-3 on DAM4SAM optimization (hyperparameter sweep, multi-object batching, prompt engineering for init mask) |

---

## Week 2 (May 31 - Jun 6) — SAM3 Integration

**Objective:** Port DAM4SAM's DRM logic onto SAM3's video predictor and run on VOT2026.

**Only execute if Week 1 decision gate green-lit.** Otherwise see "Pivot Plan" below.

### Tasks
2.1 **Create `sam3dam/` directory** as a parallel installation to `dam4sam/`. Don't modify the existing DAM4SAM directory — we need it as the fallback baseline.
```
VOTSTracker/
├── dam4sam/        # Phase 2 working baseline (untouched)
├── sam3dam/        # New: SAM3 + DRM port
│   ├── .venv/
│   ├── sam3/       # vendored SAM3 source + DRM modifications
│   └── tracker.py  # Port of dam4sam_tracker.py
```

2.2 **Identify the DRM "shape"** in DAM4SAM. Catalog exactly what SAM2 internals DAM4SAM touches:
  - `add_to_drm` (new method on predictor)
  - `propagate_in_video` modifications (returns `alternative_masks_ious` extra arg)
  - `inference_state["adds_in_drm_per_obj"]` (custom state key)
  - Any other custom keys in `inference_state`
List these in `sam3dam/PORT_TARGETS.md`.

2.3 **Implement the port.** For each item from 2.2:
  - Replicate the data path inside SAM3's video predictor
  - Use the same conditions DAM4SAM uses (IoU > 0.8, object size ratio 0.8-1.2, last_added > 5 frames)
  - Preserve the original DAM4SAM control flow in `dam4sam_tracker.py`'s `track()` method

2.4 **Smoke test on 2-3 sequences** before full eval. Pick: a short single-object (`animal`), a long single-object (`bull`), a multi-object (`racing-16` or VOT2026 equivalent).

2.5 **Full VOT2026 evaluation.** Use the same wrapper pattern (shared predictor + per-object trackers).

2.6 **Pack and submit.** This is Layer 3 — our SAM3-based attempt.

### Done Criteria for Week 2
- `sam3dam_b` registered in `trackers.ini`, `vot test sam3dam_b` passes
- Full evaluation completes (allow up to 2 failures with `--persist`)
- Submission to VOT2026 server lands a score
- Score recorded in `PHASE3_NOTES.md`

### Risks
- **SAM3 API isn't drop-in.** Mitigation: timebox 2.3 to 3 days max. If still stuck, fall back to Layer 2 (optimized DAM4SAM) for Weeks 3-4.
- **SAM3 model too large for 10 GB.** Mitigation: try SAM3 Tiny/Small; if even Small OOMs, drop SAM3 swap.
- **DRM logic doesn't transfer well to SAM3's memory format.** Mitigation: accept a simpler memory cap rule for SAM3 even if it sacrifices some DRM behavior.

---

## Week 3 (Jun 7-13) — Iterate

Two parallel directions depending on what Week 2 produced:

### If S3-DAM4SAM is running (best case)
3.1 **Hyperparameter sweep.** Variables to tune:
  - `DRM_MAX` (10, 20, 30, 50)
  - DRM-add IoU threshold (currently 0.8 — try 0.7, 0.85)
  - Object size ratio bounds (currently [0.8, 1.2])
  - SAM3 mask threshold (currently 0.0)
3.2 **Try SAM3.1** if SAM3.0 worked. Diff the configs/APIs. Likely a smaller change than SAM2→SAM3.
3.3 **Submit improved variant** mid-week.

### If S3-DAM4SAM is broken (pivot case)
3.1 **DAM4SAM optimization** in priority order:
  - Parallel object tracking (batch the per-object forward pass)
  - Init mask quality improvements (better `estimate_mask_from_box`)
  - Memory bank tuning (`num_maskmem` parameter in SAM2)
3.2 **Submit optimized DAM4SAM** mid-week.

### Common
3.3 **Address any sequences that still fail.** Investigate why and whether they can be salvaged.
3.4 **Submission strategy:** Don't burn submission slots — many challenge servers limit to N submissions per day. Check VOT2026 server policy before submitting.

---

## Week 4 (Jun 14-17) — Pack + Writeup + Final

4.1 **Final `vot pack`** of the best-scoring variant.
4.2 **Final submission** to VOT2026 (well before deadline — assume Jun 17 23:59 with buffer).
4.3 **Writeup** (`PHASE4_NOTES.md` + a separate `WRITEUP.md` if competition requires submission text):
  - What we built (architecture diagram, even if hand-drawn)
  - What worked, what didn't
  - Ablation results (DRM cap on/off, SAM2 vs SAM3, etc.)
  - VOT2026 server scores at each stage
4.4 **Repository cleanup**: tag commits for each working variant, remove dead code, update `README.md` at repo root.

---

## Pivot Plan (if SAM3 attempt fails entirely)

If Week 2 ends with no working SAM3 tracker, **abandon SAM3 in Week 3** and use Weeks 3-4 to:
1. Implement parallel object tracking in the multiobject wrapper (significant speedup, may reduce failures from sheer evaluation time)
2. Sweep DAM4SAM hyperparameters (DRM thresholds, memory bank size)
3. Improve initialization: try multi-mask SAM2 prompting at frame 0 to pick best init mask
4. Submit best-scoring variant repeatedly with tweaks

Floor in this case is still Layer 2: optimized DAM4SAM on VOT2026 with no failures.

---

## What Phase 3 Sonnet Must Stop and Ask About

These are the points where wrong choices destroy days of work — sonnet should not guess:

1. **SAM3 video predictor doesn't exist or is undocumented.** If the API reconnaissance in Stream C reveals SAM3 is image-only or video API isn't usable → stop, report, wait for Opus/user.
2. **`add_to_drm` port semantically differs.** If you can replicate the data flow but not the conditioning behavior in SAM3 → stop, describe the difference, ask.
3. **`vot evaluate` on VOT2026 fails on >2 sequences with different errors.** Suggests the toolkit/dataset has changed in unexpected ways → stop.
4. **VRAM exhausted with SAM3 even at smallest model + 1 object.** → stop, report, this may be a hard kill on SAM3 path.
5. **VOT2026 server rejects the submission archive.** Format error or auth issue → stop, get user to investigate manually.

---

## File Layout After Phase 3

```
VOTSTracker/
├── PHASE1_PLAN.md  PHASE1_NOTES.md
├── PHASE2_PLAN.md  PHASE2_NOTES.md
├── PHASE3_PLAN.md  PHASE3_NOTES.md   ← this plan, and notes to be written
├── PHASE4_NOTES.md (writeup, end of W4)
├── workspace/
│   ├── config.yaml          # stack: vots2026/main
│   ├── trackers.ini         # registers dam4sam_b, sam3dam_b
│   ├── sequences/           # VOT2026 dataset
│   ├── results/dam4sam_b/   # Layer 1+2 results
│   └── results/sam3dam_b/   # Layer 3 results (if attempted)
├── dam4sam/                 # Phase 2 baseline + DRM cap (Layer 1+2)
├── sam3dam/                 # SAM3-based attempt (Layer 3, may not exist)
├── sam3_upstream/           # SAM3 reference clone (read-only)
└── eval_progress.log
```

---

## Definition of Done for Phase 3

Hard requirements (must achieve):
- [ ] VOT2026 baseline DAM4SAM submitted with score recorded
- [ ] DRM cap implemented, hat-5/robot-19 equivalent failures resolved
- [ ] WSL on D: drive
- [ ] SAM3 reconnaissance report (`SAM3_API_NOTES.md`) written
- [ ] PHASE3_NOTES.md written documenting all decisions

Stretch (sprint success):
- [ ] S3-DAM4SAM port running end-to-end
- [ ] S3-DAM4SAM score on VOT2026 server
- [ ] At least one hyperparameter sweep result documented

Phase 3 succeeds when we have **any** score on the VOT2026 leaderboard, the failure mode from Phase 2 is fixed, and we've made an honest, time-boxed attempt at the SAM3 swap. The submission is the win condition; the leaderboard rank is the bonus.
