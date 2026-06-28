# DAM4SAM-VOTS2026

Training-free extensions of [DAM4SAM](https://github.com/jovanavidenovic/DAM4SAM) for the
[VOTS2026](https://www.votchallenge.net/vots2026/) multi-object video segmentation & tracking
challenge.

The trackers build on **DAM4SAM** (Distractor-Aware Memory for SAM2-based tracking), which uses
**[SAM 2.1](https://github.com/facebookresearch/sam2)** as its backbone. All credit for DAM4SAM and
SAM2 goes to their original authors (see *Acknowledgements*). The original contributions in this
repository are the tracker variants and evaluation tooling described below.

## Submitted trackers

| Tracker name | Backbone | Added logic | File / class |
|---|---|---|---|
| `dam4sam_b` | SAM2.1-B+ | baseline DAM4SAM | `dam4sam/dam4sam_tracker.py` |
| `dam4sam_v3g` | SAM2.1-B+ | scale-invariant motion gate | `dam4sam/dam4sam_tracker_v3g.py` |
| `dam4sam_l_cloud` | SAM2.1-Large | baseline DAM4SAM (large backbone) | `dam4sam/vot_wrapper_large.py` |
| `dam4sam_l_d2` | SAM2.1-Large | motion gate + identity-aware memory gating (D2) | `dam4sam/dam4sam_tracker_v7.py` (`USE_D2=True`) |
| `dam4sam_v7d2` | SAM2.1-B+ | motion gate + identity gating (D2) — ablation | `dam4sam/dam4sam_tracker_v7.py` (`USE_D2=True`) |

Each tracker is a **pure subclass** of the baseline; with its flags disabled it is bit-identical to
DAM4SAM (verified per change).

### Original contributions
- **Motion gate (V3 → V3g)** — a constant-velocity Kalman filter used only as a *gated tiebreaker*,
  activated when SAM2.1 is ambiguous and the object's scale-invariant velocity exceeds a threshold
  (`dam4sam_tracker_v3.py`, `dam4sam_tracker_v3g.py`).
- **Identity channel (D2)** — an independent identity signal (per-candidate SAM2 object pointers +
  mask-pooled backbone features) that gates Distractor-Aware Memory admission via a prototype bank
  (`dam4sam_tracker_v7.py`).
- **Evaluation & infrastructure** — `run_eval_170.py` (sequence runner with proxy metrics),
  `check_results_integrity.py` (pre-submission strict integrity check), and the multi-GPU cloud
  run scripts.

A full technical journal of the experiments (with the real-server ablation results) is in
[`PREZENTARE_VERSIUNI.md`](PREZENTARE_VERSIUNI.md).

## Setup

```bash
# Python 3.10, env + pinned deps
uv venv dam4sam/.venv --python 3.10
uv pip install --python dam4sam/.venv/bin/python \
  torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
uv pip install --python dam4sam/.venv/bin/python \
  numpy==1.26.4 opencv-python hydra-core omegaconf vot-toolkit==0.8.0 scipy pillow tqdm iopath
```

**Model weights are not included** (licensing + size). Download the SAM2.1 checkpoints from the
[SAM2 repository](https://github.com/facebookresearch/sam2) into `dam4sam/checkpoints/`:
`sam2.1_hiera_base_plus.pt`, `sam2.1_hiera_large.pt`.

The VOTS2026 dataset is obtained via the VOT toolkit:
```bash
vot initialize vots2026/main --workspace workspace2026
```

## Running

```bash
# motion gate, B+ backbone
python run_eval_170.py --tracker v3g  --backbone B --out dam4sam_v3g
# motion gate + identity gating (D2), Large backbone
python run_eval_170.py --tracker v7d2 --backbone L --out dam4sam_l_d2
```

## Acknowledgements
- **DAM4SAM** — Videnović, Lukežič, Kristan. <https://github.com/jovanavidenovic/DAM4SAM> · arXiv:2411.17576
- **SAM 2** — Ravi et al., Meta AI. <https://github.com/facebookresearch/sam2> · arXiv:2408.00714

This is derivative research code; the licenses of the respective upstream components apply.
