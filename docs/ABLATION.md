# Method & Ablation Study

This document summarizes the tracker variants developed for VOTS2026, the evaluation
methodology, and the full ablation — including the negative results, which turned out to be
the most instructive part of the project.

## Setting

All variants are **training-free** extensions of [DAM4SAM](https://github.com/jovanavidenovic/DAM4SAM)
(Distractor-Aware Memory for SAM2.1). Each variant is a pure subclass of the baseline tracker;
with its flags disabled it is bit-identical to DAM4SAM (verified per change via MD5 on output
masks). The guiding principle: *never intervene unless there is evidence the intervention helps
on that frame.*

Server feedback on the baseline (`dam4sam_b`, SAM2.1-B+) showed the dominant error mode was
**drift** (DRE = 0.09) rather than missed detections (NRE = 0.05), so both original components
target identity/drift failures:

### Component 1 — Scale-invariant motion gate (V3 → V3g)

A per-object constant-velocity Kalman filter provides a motion prediction. When SAM2.1's mask
selection is **ambiguous** (multiple candidate masks with close scores), candidates are re-ranked
by consistency with the motion prediction. Plain V3 applied this unconditionally and regressed on
slow/deformable sequences; **V3g adds a speed gate** — re-ranking activates only when the object's
inter-frame displacement, *normalized by object size*, exceeds a threshold and the filter state is
healthy. With the gate closed the tracker is the baseline, by construction.

### Component 2 — Identity-aware memory admission (D2)

DAM4SAM's distractor-resolving memory (DRM) is only as good as the frames admitted into it.
D2 gates `add_to_drm`: a frame is admitted only if the cosine similarity between the chosen mask's
pooled memory-encoder embedding and a **prototype bank** (immutable init prototype P0 + a rolling
last-confident slot + strictly-admitted prototypes, FIFO-capped) exceeds a threshold. The goal is
to keep distractor-contaminated frames out of long-term memory.

## Evaluation methodology

The VOTS2026 test ground truth is sequestered; scored server submissions were limited (~10 per
tracker, 24h apart). The evaluation ladder used to spend them carefully:

1. **Local proxy metrics** on the full sequence set (mask-emptiness / permanence robustness proxies) —
   free, but domain-biased; a proxy win on a subset does not guarantee a real-Q win overall.
2. **VOST-val calibration** (70 kitchen sequences with public GT, official `vot analysis`) — free
   real-Q signal, but a different domain (slow, deformable motion): useful to prove V3 *hurts* on
   slow sequences (which motivated the V3g gate), not to predict fast-motion gains.
3. **Targeted stress tests** (e.g. LaSOT drift subsets) for component design decisions.
4. **Server submissions** as the final arbiter, one per carefully chosen candidate.

Guardrails that caught real bugs: a mandatory single-sequence **smoke test** before every long
run, and a strict **pre-submission integrity check** (`scripts/check_results_integrity.py`) — it
caught a silent 0/7-masks failure on a 7-object sequence and an `rsync --partial` truncation that
would have corrupted a submission.

## Ablation results

### Challenge-period feedback (VOTS2025 split, 170 sequences)

During the challenge the server scored submissions on the VOTS2025 split only. This gave a clean
2×2 ablation (backbone × components):

| Variant | Backbone | Components | Q (VOTS2025 split) |
|---|---|---|---|
| `dam4sam_b` | B+ | — | 0.69 |
| `dam4sam_v3g` | B+ | motion gate | 0.69 |
| `dam4sam_v7d2` | B+ | motion gate + D2 | 0.69 |
| `dam4sam_l_cloud` | Large | — | **0.72** |
| `dam4sam_l_d2` | Large | motion gate + D2 | 0.71 |

Verdict on this split: **backbone scale was the only lever** (+0.03); both hand-crafted
components were Q-neutral or slightly negative. Notably, D2 was neutral even *in-calibration*
(same B+ backbone and native thresholds as the baseline), which rules out "just mis-calibrated
for Large": the identity signal itself did not separate target from distractor reliably on real
multi-object data — a domain-extrapolation trap, since it had looked strong on single-object
LaSOT stress tests.

### Final public results (VOTS2026 dataset, 120 sequences — official challenge split)

Released with the [VOTS2026 results paper](https://www.votchallenge.net/vots2026/); the new
dataset is harder and emphasizes drift/identity failures.

| Variant | Q | Acc | Rob | NRE | DRE | Rank |
|---|---|---|---|---|---|---|
| `dam4sam_l_cloud` | **0.639** | 0.789 | 0.811 | 0.054 | 0.135 | **9 / 25** |
| `dam4sam_l_d2` | 0.630 | 0.791 | 0.805 | 0.052 | 0.144 | 12 |
| `dam4sam_v3g` | 0.614 | 0.769 | 0.817 | **0.035** | 0.148 | 15 |
| `dam4sam_v7d2` | 0.608 | 0.766 | 0.809 | 0.038 | 0.154 | 16 |
| `dam4sam_sam3` | 0.599 | 0.714 | 0.793 | 0.070 | 0.137 | 18 |
| `dam4sam_b` | 0.598 | 0.716 | 0.796 | 0.055 | 0.149 | 19 |
| *DAM4SAM (committee baseline)* | *0.626* | | | | | |
| *MSAOT (entry baseline)* | *0.439* | | | | | |

Reference points: challenge winner VLM-SAFEDAM Q=0.733; SAM3 committee baseline Q=0.645.

Two findings survive, one gets revised:

- **Backbone scale remains the main lever**: B+ → Large is +0.041 on this dataset, and
  `dam4sam_l_cloud` outperforms the committee's own DAM4SAM entry (0.639 vs 0.626).
- **D2 remains slightly negative on Large** (0.630 vs 0.639), confirming the challenge-period
  verdict on the identity signal.
- **Revised — the motion gate is *not* a no-op on hard data**: on the drift-heavy VOTS2026 split,
  `dam4sam_v3g` gains **+0.016 Q** over the B+ baseline (Acc +0.053, Rob +0.021) and posts
  NRE = 0.035, the **3rd best not-reported error of all 25 entries**. The earlier "no-op" verdict
  was an artifact of the milder VOTS2025 split, where gains on fast sequences and losses on slow
  ones cancelled out. The gate behaves as designed: inactive where motion is uninformative,
  helpful where it is.

## Lessons

1. **A proxy metric is faithful only on its own domain.** A win on a hand-picked hard subset does
   not transfer to the full distribution; only the sequestered set settles it.
2. **Cross-domain extrapolation is the main trap.** Single-object LaSOT signals (D2's 4× drift
   reduction) did not survive contact with real multi-object sequences.
3. **Verify "no effect" on more than one distribution.** The same component can be a wash on one
   dataset split and a clear win on a harder one (V3g).
4. **Infrastructure rigor pays**: smoke tests before long runs, strict integrity checks before
   packing, resumable runners, and an incremental rescue puller for preemptible cloud GPUs each
   prevented at least one lost run or corrupted submission.
