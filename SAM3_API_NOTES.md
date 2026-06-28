# SAM3 API Compatibility Notes for DAM4SAM Port

**Written:** 2026-05-23  
**Purpose:** Go/no-go decision for SAM3 backbone swap (Phase 3, Week 1 gate)  
**SAM3 source:** `sam3_upstream/` (cloned from `facebookresearch/sam3`)  
**Target predictor:** `sam3/model/sam3_tracking_predictor.py` → `Sam3TrackerPredictor`  
**Builder function:** `build_tracker()` in `sam3/model_builder.py`

---

## 1. `inference_state` Key Compatibility

SAM3's `Sam3TrackerPredictor.init_state()` creates an `inference_state` dict.
DAM4SAM reads/writes these keys from `inference_state`:

| Key | SAM2 (DAM4SAM) | SAM3 | Status |
|-----|----------------|------|--------|
| `output_dict` | `{"cond_frame_outputs": {}, "non_cond_frame_outputs": {}}` | Same | ✅ |
| `output_dict_per_obj` | `{obj_idx: {"cond_frame_outputs": {}, ...}}` | Same | ✅ |
| `consolidated_frame_inds` | `{"cond_frame_outputs": set(), "non_cond_frame_outputs": set()}` | Same | ✅ |
| `temp_output_dict_per_obj` | Present | Present | ✅ |
| `obj_id_to_idx` | OrderedDict | OrderedDict | ✅ |
| `obj_idx_to_id` | OrderedDict | OrderedDict | ✅ |
| `obj_ids` | list | list | ✅ |
| `frames_already_tracked` | dict | dict | ✅ |
| `tracking_has_started` | bool | bool | ✅ |
| `offload_video_to_cpu` | bool | bool | ✅ |
| `offload_state_to_cpu` | bool | bool | ✅ |
| `storage_device` | torch.device | torch.device | ✅ |
| `adds_in_drm_per_obj` | `{obj_idx: [frame_idx, ...]}` | **MISSING** | 🔴 Must add |

**Summary:** All SAM2 keys are present in SAM3 *except* `adds_in_drm_per_obj`, which is a
DAM4SAM-specific addition that tracks which frames were added to DRM. This key needs to
be injected into SAM3's `init_state` output before calling `add_to_drm`.

---

## 2. Method API Mapping

| DAM4SAM method | SAM3 equivalent | Status | Notes |
|----------------|-----------------|--------|-------|
| `init_state(video_path, num_frames, video_height, video_width, offload_video_to_cpu, offload_state_to_cpu)` | Same signature | ✅ | |
| `add_new_mask(inference_state, obj_id, frame_idx, mask)` | Same | ✅ | |
| `add_new_points_or_box(...)` | Same | ✅ | |
| `add_new_points(...)` | Same (alias) | ✅ | |
| `remove_object(inference_state, obj_id, is_user_action)` | Same | ✅ | |
| `reset_state(inference_state)` | Same | ✅ | |
| `add_to_drm(inference_state, frame_idx, obj_id)` | **NOT PRESENT** | 🔴 Must port | ~50 lines; manipulates `cond_frame_outputs` |
| `_evict_oldest_drm_frame(inference_state, obj_idx)` | **NOT PRESENT** | 🔴 Must port | Added in Phase 3 for DRM cap |
| `propagate_in_video(...)` yields `(frame_idx, obj_ids, masks)` | yields **5 values**: `(frame_idx, obj_ids, low_res_masks, video_res_masks, obj_scores)` | ⚠️ Extra unpacking needed | Update `vot_wrapper_multiobject.py` |

---

## 3. Multimask / Alternative Masks

In DAM4SAM (SAM2), `add_to_drm` calls `propagate_in_video` with `return_all_masks=True`,
gets all candidate masks, and selects the best one based on IoU with the previous DRM mask.
This flag is DAM4SAM-specific and was added to `sam2_video_predictor.py`.

**SAM3 behavior:** `multimask_output_for_tracking=True` is **hardcoded** in `build_tracker()`.
During inference, the best mask is selected internally via `dynamic_multimask_via_stability`
(stability score threshold 0.98, delta 0.05). There is no `return_all_masks` parameter in
`propagate_in_video`.

**Port strategy:** Remove the `return_all_masks` pathway entirely. Trust SAM3's internal
dynamic multimask selection. This is actually cleaner — SAM3 handles the selection internally.

---

## 4. Environment & Dependency Requirements

| Requirement | SAM3 recommended | Current env | Verdict |
|-------------|------------------|-------------|---------|
| Python | 3.12 | 3.12.13 ✅ | ✅ |
| PyTorch | 2.10.0 (cu128) | 2.5.1+cu121 | ⚠️ Below recommended |
| CUDA toolkit | 12.8 | 12.1 | ⚠️ At minimum |
| flash-attn-3 | Optional (`use_fa3=False` default) | Not installed | ✅ Falls back to SDPA |
| cc_torch | Optional (compile only) | Not installed | ✅ |
| einops | Required | TBD | Install in new venv |
| SAM3 checkpoint | `facebook/sam3` (gated HF) | **NOT DOWNLOADED** | 🔴 #1 blocker |

### PyTorch version analysis
SAM3 README says torch 2.10.0 + CUDA 12.8. However:
- `get_sdpa_settings()` only requires torch ≥ 2.2 for Flash Attention v2 fallback
- With `use_fa3=False` (the default), SAM3 uses `F.scaled_dot_product_attention` (available since torch 2.0)
- **Conclusion: torch 2.5.1 should work for SAM3.0 tracker-only usage** (no FA3, no compile)
- If there are import errors on 2.5.1, we create a fresh venv with torch 2.10 (separate from dam4sam)

### New venv needed
SAM3 should be installed in its own venv to avoid polluting the DAM4SAM environment:
```bash
uv venv /home/alexx/Projects/VOTSTracker/sam3_venv --python 3.12
source /home/alexx/Projects/VOTSTracker/sam3_venv/bin/activate
pip install torch==2.10.0 torchvision --index-url https://download.pytorch.org/whl/cu128
pip install einops
pip install -e /home/alexx/Projects/VOTSTracker/sam3_upstream/
```

---

## 5. Checkpoint Status 🔴

**Checkpoint location:** `facebook/sam3` on HuggingFace (gated — requires access request)  
**File:** `sam3.pt` (size unknown; comparable SAM2.1 checkpoints are 308 MB)  
**HuggingFace access URL:** https://huggingface.co/facebook/sam3

**User action required:**
1. Visit https://huggingface.co/facebook/sam3
2. Click "Request access" and fill out the form
3. Wait for approval (typically 1-2 business days for Meta models)
4. Run: `huggingface-cli login` (enter token)
5. The `build_tracker()` function will then auto-download the checkpoint

**Without this step, SAM3 cannot be initialized.**

---

## 6. Porting Workplan (3 tasks, ~2-3 hours total)

### Task 1: Inject `adds_in_drm_per_obj` into SAM3 `init_state`
**File:** `sam3_upstream/sam3/model/sam3_tracking_predictor.py`  
**Effort:** ~5 lines

After line 135 (`self.clear_all_points_in_video(inference_state)`):
```python
# DRM tracking (DAM4SAM port): tracks which frame indices were added to DRM per object
inference_state["adds_in_drm_per_obj"] = {}
```

### Task 2: Port `add_to_drm` and `_evict_oldest_drm_frame` from SAM2VideoPredictor
**Source:** `dam4sam/sam2/sam2_video_predictor.py`  
**Target:** `sam3_upstream/sam3/model/sam3_tracking_predictor.py`  
**Effort:** ~70 lines

The method directly manipulates `cond_frame_outputs`, `output_dict_per_obj`, and
`consolidated_frame_inds` — all of which exist in SAM3 with identical names and structure.
The port should be nearly copy-paste. Remove `return_all_masks=True` pathway; use standard
`propagate_in_video` output (drop `obj_scores` return value; just unpack 5 not 3 values).

### Task 3: Update `vot_wrapper_multiobject.py` to unpack 5 values from SAM3's `propagate_in_video`
**File:** `dam4sam/vot_wrapper_multiobject.py`  
**Effort:** ~3 lines

SAM3's `propagate_in_video` yields:
```python
frame_idx, obj_ids, low_res_masks, video_res_masks, obj_scores = next(propagation)
```
SAM2 yields 3 or 4 values. Add an `obj_scores` unpacking.

---

## 7. Go / No-Go Recommendation

### ✅ GO — if:
- HuggingFace access is approved (user submits request)
- torch 2.5.1 smoke test passes: `python -c "import sam3; print('OK')"`

### ❌ NO-GO — if:
- HF access rejected or takes >5 days
- SAM3 import fails on torch 2.5.1 AND torch 2.10 venv conflicts with CUDA 12.1 driver
- Week 2 time budget is exhausted by env issues (fallback: optimize DRM cap parameters instead)

### Current status: **BLOCKED on HF checkpoint access**
Action item: Request HF access ASAP (today). Porting work (Tasks 1-3) can be done offline.

---

## 8. Risk Register

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| HF access not approved in time | Medium | Fatal for SAM3 plan | Submit request today; fallback to DRM tuning |
| torch 2.5.1 incompatibility | Low | 1-2h delay for new venv | Pre-create torch 2.10 venv now |
| SAM3 memory exceeds 12GB VRAM | Medium | Need more offloading | Enable `offload_state_to_cpu=True` |
| SAM3 slower than SAM2.1 per-frame | Unknown | May need more timeouts | Increase timeout in trackers.ini |
| `add_to_drm` semantic differences | Low | 0.5-1h debug | Port is near-copy; cond_frame_outputs identical |
