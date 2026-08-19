# DAM4SAM-VOTS2026

Training-free extensions of [DAM4SAM](https://github.com/jovanavidenovic/DAM4SAM) for the
[VOTS2026](https://www.votchallenge.net/vots2026/) visual object tracking & segmentation
challenge (ECCV 2026), on top of SAM2.1.

Six trackers from this repository were evaluated in the main VOTS2026 sub-challenge. All of them
exceeded the co-authorship baseline, and the results are part of the official challenge paper
(*The Fourth Visual Object Tracking Segmentation VOTS2026 Challenge Results*, Kristan et al.,
ECCV 2026 Workshops — [OpenReview](https://openreview.net/forum?id=zGOan36xpF)).

> **Post-challenge finding (Aug 2026).** A per-component ablation scored on the *sequestered*
> evaluation set showed that the motion gate these trackers were built around **never executed**:
> removing the entire motion pathway leaves the output masks byte-identical on all 375
> target-sequences. The measured gain comes from a bounded sliding window over SAM2's
> non-conditioning memory — a change originally made for GPU-memory reasons. The full diagnosis
> is in [docs/ABLATION.md](docs/ABLATION.md); the knock-out variants are in
> [`dam4sam/dam4sam_tracker_knockouts.py`](dam4sam/dam4sam_tracker_knockouts.py).

## Official results (VOTS2026 dataset, 120 sequences, sequestered ground truth)

| Tracker | Backbone | Added logic | Q | Rank |
|---|---|---|---|---|
| `dam4sam_l_cloud` | SAM2.1-Large | — (backbone scaling) | **0.639** | **9 / 25** |
| `dam4sam_l_d2` | SAM2.1-Large | motion gate + identity admission | 0.630 | 12 |
| `dam4sam_v3g` | SAM2.1-B+ | motion gate | 0.614 | 15 |
| `dam4sam_v7d2` | SAM2.1-B+ | motion gate + identity admission | 0.608 | 16 |
| `dam4sam_sam3` | SAM3 | DAM4SAM memory on SAM3 | 0.599 | 18 |
| `dam4sam_b` | SAM2.1-B+ | — (baseline) | 0.598 | 19 |

Context: challenge winner VLM-SAFEDAM Q=0.733; committee baselines SAM3 Q=0.645,
**DAM4SAM Q=0.626** (outperformed by `dam4sam_l_cloud`), MSAOT Q=0.439 (entry criterion).

Highlights:

- **`dam4sam_l_cloud` (Q=0.639) outperforms the committee's own DAM4SAM entry (0.626)** —
  achieved by scaling the image encoder to Hiera-L and engineering the full-dataset evaluation
  on preemptible cloud GPUs (see [docs/CLOUD_SETUP.md](docs/CLOUD_SETUP.md)).
- **`dam4sam_v3g` posts NRE=0.035 — the 3rd best not-reported error among all 25 entries**, and the
  best of any entry that keeps absence detection functional (the two lower NRE scores come from
  trackers with ADQ ≤ 0.374 that simply under-report absence). It improves Q by +0.016 over the
  same-backbone baseline.
- **The mechanism behind that number is not the one we designed for.** A 7-variant knock-out
  ablation, scored on the sequestered set and verified byte-for-byte against the stored
  trajectories, attributes the gain to bounded non-conditioning memory and shows the gated motion
  prior to be inert — and actively harmful when forced to run. See
  [docs/ABLATION.md](docs/ABLATION.md).

## Method (original contributions)

All variants are pure subclasses of the DAM4SAM baseline; with flags disabled each is
bit-identical to it (verified per change).

- **Bounded non-conditioning memory** (`dam4sam/dam4sam_tracker_v3.py`, `MEM_WINDOW`) — SAM2 retains
  its own past predictions for every processed frame; on sequences of thousands of frames this
  history grows unbounded, so evidence produced while the tracker was confused about a distractor
  stays available to memory attention indefinitely. A sliding window evicts non-conditioning
  outputs older than `W = 300` frames while leaving conditioning memory (promoted anchors,
  including frame 0) untouched. **This is what the measured improvement is attributable to** — see
  the ablation.
- **Scale-invariant motion gate** (`dam4sam/dam4sam_tracker_v3g.py`) — a constant-velocity Kalman
  filter used only as a *gated tiebreaker*: when SAM2.1's mask selection is ambiguous, candidates
  are re-ranked by motion consistency, but only if the object's size-normalized velocity exceeds a
  threshold and the filter state is healthy. Gate closed ⇒ exact baseline behavior. *In practice
  the gate is unreachable* (`predict()` is only called inside the gated branch, so the velocity it
  tests never leaves zero); the ablation documents this and quantifies what happens when the
  pathway is forced to run.
- **Knock-out ablation harness** (`dam4sam/dam4sam_tracker_knockouts.py`) — seven constant-only
  subclasses that each disable exactly one component, plus a byte-level comparison of stored
  trajectories that distinguishes *"contributes nothing"* from *"never executed"*.
- **Identity-aware memory admission (D2)** (`dam4sam/dam4sam_tracker_v7.py`, `USE_D2`) — admission
  into DAM4SAM's distractor-resolving memory is gated on cosine similarity between the chosen
  mask's pooled memory-encoder embedding and a prototype bank (immutable init prototype + rolling
  last-confident slot + strictly-admitted prototypes), keeping distractor-contaminated frames out
  of long-term memory.
- **Evaluation & infrastructure** — resumable full-dataset runner
  ([scripts/run_eval_170.py](scripts/run_eval_170.py)), strict pre-submission integrity check
  ([scripts/check_results_integrity.py](scripts/check_results_integrity.py)), and multi-GPU cloud
  tooling with an incremental rescue puller ([scripts/cloud/](scripts/cloud)).

## Repository layout

```
dam4sam/            tracker code (DAM4SAM upstream + variant subclasses + VOT wrappers)
scripts/            evaluation runner, integrity check, ablation runner, trajectory diff
scripts/cloud/      cloud GPU setup, run scripts, rescue puller
workspace2026/      VOT toolkit workspace configuration (trackers.ini, stack config)
docs/               method & ablation writeup, raw ablation scores, cloud workflow
```

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

The SAM3 variant (`dam4sam_sam3`) additionally needs a clone of the
[SAM3 repository](https://github.com/facebookresearch/sam3) and its checkpoint; point `SAM3_ROOT`
(and optionally `SAM3_CHECKPOINT`) at it, or place the clone next to this repository as
`../sam3_upstream`.

The VOTS2026 dataset is obtained via the VOT toolkit:
```bash
vot initialize vots2026/main --workspace workspace2026
```

## Running

```bash
# motion gate, B+ backbone
python scripts/run_eval_170.py --tracker v3g  --backbone B --out dam4sam_v3g
# motion gate + identity admission (D2), Large backbone
python scripts/run_eval_170.py --tracker v7d2 --backbone L --out dam4sam_l_d2

# verify before packing a submission
python scripts/check_results_integrity.py --workspace workspace2026 --tracker dam4sam_v3g
```

The official challenge runs used the VOT toolkit directly (`vot evaluate` / `vot pack`) with the
registry in [workspace2026/trackers.ini](workspace2026/trackers.ini) (adjust the absolute paths to
your clone). For full-dataset runs on rented GPUs see [docs/CLOUD_SETUP.md](docs/CLOUD_SETUP.md).

## Reproducing the ablation

The seven knock-out variants are constants-only subclasses of the full model, so each one differs
from it by exactly one component
([`dam4sam/dam4sam_tracker_knockouts.py`](dam4sam/dam4sam_tracker_knockouts.py)):

```bash
# all seven variants: environment control, per-variant smoke test, full run, integrity check
scripts/run_knockouts.sh
VARIANTS="ko_motion ko_chassis" scripts/run_knockouts.sh   # or a subset

# the byte-level comparison the conclusions rest on (no GPU, no ground truth needed)
python scripts/compare_trajectories.py --reference dam4sam_v3g --variants 'dam4sam_ko_*'
```

The comparison is what separates *"this component contributes nothing"* from *"this component
never executed"*: three variants reproduce the full model's masks byte for byte on all 375
target-sequences, while the variants that do bypass the speed gate differ on 361-375 of them.
Benchmark scores for every variant are in
[docs/knockout_scores.txt](docs/knockout_scores.txt); the analysis is in
[docs/ABLATION.md](docs/ABLATION.md).

## Citation

If you use these trackers or results, please cite the VOTS2026 results paper:

```bibtex
@inproceedings{Kristan2026VOTS,
  title     = {The Fourth Visual Object Tracking Segmentation VOTS2026 Challenge Results},
  author    = {Kristan, Matej and Matas, Ji\v{r}\'i and Tokmakov, Pavel and Harley, Adam and
               Felsberg, Michael and \v{C}ehovin Zajc, Luka and Luke\v{z}i\v{c}, Alan and others},
  booktitle = {Computer Vision -- ECCV 2026 Workshops},
  year      = {2026}
}
```

## Acknowledgements & license

- **DAM4SAM** — Videnović, Lukežič, Kristan (CVPR 2025, IJCV 2026).
  <https://github.com/jovanavidenovic/DAM4SAM> · arXiv:2411.17576
- **SAM 2** — Ravi et al., Meta AI (Apache-2.0). <https://github.com/facebookresearch/sam2> ·
  arXiv:2408.00714

This is derivative research code built for challenge participation. The original contributions
(the tracker variants, the ablation harness, the evaluation and cloud tooling, and the
documentation) are by Alexandru-Marian Galea and are released under the Apache License 2.0. The
vendored DAM4SAM and SAM2 components are not covered by it and remain under their upstream terms —
[NOTICE](NOTICE) states exactly which files fall on which side. Model weights are not
distributed here.
