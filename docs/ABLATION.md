# Method & Ablation Study

This document summarizes the tracker variants developed for VOTS2026, the evaluation
methodology, and the full ablation — including the negative results, which turned out to be the
most instructive part of the project.

**Summary of the outcome.** Six trackers were submitted to VOTS2026 and all cleared the
co-authorship baseline. The best (`dam4sam_l_cloud`, Q=0.639, 9/25) beat the committee's own
DAM4SAM entry. The most interesting (`dam4sam_v3g`) posted **NRE=0.035, the 3rd lowest
not-reported error of all 25 entries**. A post-challenge per-component ablation, scored on the
same sequestered set, then showed that the component we credited for that number **never ran**,
and that the gain belongs to a bounded memory window introduced for unrelated engineering
reasons. Both halves are reported below.

## Setting

All variants are **training-free** extensions of
[DAM4SAM](https://github.com/jovanavidenovic/DAM4SAM) (Distractor-Aware Memory for SAM2.1). Each
variant is a pure subclass of the baseline tracker; with its flags disabled it is bit-identical
to DAM4SAM (verified per change via MD5 on output masks). The guiding principle: *never intervene
unless there is evidence the intervention helps on that frame.*

Server feedback on the baseline (`dam4sam_b`) showed drift (DRE=0.09) dominating missed
detections (NRE=0.05), so the designed components targeted identity/drift failures.

### Component 1 — Bounded non-conditioning memory (`MEM_WINDOW`)

SAM2 keeps two stores: *conditioning* memory (a small set of promoted anchor frames — DAM4SAM's
recent-appearance and distractor-resolving banks, capped and admission-gated) and
*non-conditioning* memory, which holds the model's own outputs for previously processed frames
and is consulted by memory attention for temporal consistency. The baseline retains
non-conditioning outputs for **every** frame; on VOTS2026 sequences (mean ≈2,000 frames, max
10,699) that history is unbounded.

Our variants evict non-conditioning entries older than `W = 300` frames. Conditioning memory is
untouched, so long-term re-detection anchors survive. `W` was set by bisection on a development
subset: below ~100 re-acquisition after long absences breaks; above a few hundred there is no
further benefit and memory exhaustion looms. The window comfortably exceeds SAM2's attention span
(`num_maskmem × stride ≈ 35` frames).

Intended mechanism: non-conditioning memory is populated by the tracker's *own* predictions, which
during a confusion episode are predictions of a distractor. Retained indefinitely, that evidence
biases the model when the true target reappears. Bounding the window gives contamination a finite
lifetime.

*Introduced as an OOM fix; turned out to be the only component that moves the metrics.*

### Component 2 — Scale-invariant motion gate (V3 → V3g)

A per-object constant-velocity Kalman filter provides a motion prediction. When SAM2.1's mask
selection is **ambiguous** (top-two predicted-IoU within a tie margin), candidates are re-ranked
by consistency with that prediction. Plain V3 applied this whenever SAM was ambiguous and
regressed on slow/deformable sequences; **V3g adds a speed gate** — re-ranking activates only when
the object's displacement, normalized by object size, exceeds a threshold and the filter state is
healthy. With the gate closed the tracker is the baseline, by construction.

**This component never executes.** See [Diagnosis](#diagnosis-a-gate-that-cannot-open).

### Component 3 — Identity-aware memory admission (D2)

DAM4SAM's distractor-resolving memory (DRM) is only as good as the frames admitted into it. D2
gates `add_to_drm`: a frame is admitted only if the cosine similarity between the chosen mask's
pooled memory-encoder embedding and a **prototype bank** (immutable init prototype P0 + rolling
last-confident slot + strictly-admitted prototypes, FIFO-capped) exceeds a threshold.

## Evaluation methodology

The VOTS2026 test ground truth is sequestered; scored submissions were limited (~10 per tracker,
24h apart). The ladder used to spend them carefully:

1. **Local proxy metrics** on the full sequence set (mask-emptiness / permanence proxies) — free,
   but domain-biased.
2. **VOST-val calibration** (70 kitchen sequences with public GT, official `vot analysis`) — free
   real-Q signal, but a different domain (slow, deformable motion).
3. **Targeted stress tests** (LaSOT drift subsets) for component design decisions.
4. **Server submissions** as the final arbiter, one per carefully chosen candidate.
5. **Post-challenge benchmark** — per-component knock-outs on the sequestered set.

Guardrails that caught real bugs: a mandatory single-sequence **smoke test** before every long
run, and a strict **pre-submission integrity check**
([`scripts/check_results_integrity.py`](../scripts/check_results_integrity.py)) — it caught a
silent 0/7-masks failure on a 7-object sequence and an `rsync --partial` truncation that would
have corrupted a submission.

## Official results (VOTS2026, 120 sequences, sequestered)

| Variant | Q | Acc | Rob | NRE | DRE | ADQ | Rank |
|---|---|---|---|---|---|---|---|
| `dam4sam_l_cloud` | **0.639** | 0.789 | 0.811 | 0.054 | 0.135 | 0.676 | **9 / 25** |
| `dam4sam_l_d2` | 0.630 | 0.791 | 0.805 | 0.052 | 0.144 | 0.581 | 12 |
| `dam4sam_v3g` | 0.614 | 0.769 | 0.817 | **0.035** | 0.148 | 0.542 | 15 |
| `dam4sam_v7d2` | 0.608 | 0.766 | 0.809 | 0.038 | 0.154 | 0.561 | 16 |
| `dam4sam_sam3` | 0.599 | 0.714 | 0.793 | 0.070 | 0.137 | 0.604 | 18 |
| `dam4sam_b` | 0.598 | 0.716 | 0.796 | 0.055 | 0.149 | 0.571 | 19 |
| *DAM4SAM (committee)* | *0.626* | | | *0.062* | | | *14* |
| *MSAOT (entry baseline)* | *0.439* | | | | | | |

Reference points: winner VLM-SAFEDAM Q=0.733; SAM3 committee baseline Q=0.645.

**On the NRE result.** NRE in isolation is gameable — a tracker that never declares absence scores
near-zero trivially. The challenge data contains the reductio: SAMBA-D4 has the lowest NRE
(0.019) with ADQ=0.000 and Q=0.034; the second lowest (SENTRY-SR, 0.032) pays with ADQ=0.374.
Ours is the **lowest NRE of any entry with functional absence detection (ADQ > 0.5)**, while
simultaneously improving Q, Acc and Rob over its baseline — which rules out presence bias.

**Orthogonality.** Backbone scaling (B+ → Large) moves Q by +0.041 but leaves NRE essentially
unchanged (0.055 → 0.054). The memory window leaves DRE unchanged (0.149 → 0.148) but collapses
NRE. The two levers act on disjoint error axes and compose.

## Per-component knock-outs (sequestered set)

Each variant disables exactly one component; everything else is bit-identical. Rows marked ≡
produced **byte-identical output masks** to the full model on all 375 target-sequences — verified
directly on the stored trajectories, not merely equal metrics.

| Variant | Q | Acc | Rob | NRE | DRE | ADQ | vs full |
|---|---|---|---|---|---|---|---|
| Baseline (DAM4SAM, B+) | 0.598 | 0.716 | 0.796 | 0.055 | 0.149 | 0.571 | −0.016 |
| **Full (`dam4sam_v3g`)** | 0.614 | 0.769 | 0.817 | 0.035 | 0.148 | 0.542 | — |
| − filter lifecycle ≡ | 0.614 | 0.769 | 0.817 | 0.035 | 0.148 | 0.542 | 0.000 |
| − ambiguity gate ≡ | 0.614 | 0.769 | 0.817 | 0.035 | 0.148 | 0.542 | 0.000 |
| − **entire motion pathway** ≡ | 0.614 | 0.769 | 0.817 | 0.035 | 0.148 | 0.542 | 0.000 |
| − conditional filtering | **0.615** | 0.769 | 0.819 | 0.035 | 0.146 | 0.542 | +0.001 |
| − speed gate (motion forced on) | 0.604 | 0.765 | 0.807 | 0.038 | 0.155 | 0.524 | −0.010 |
| Unconditional SAMURAI-style blend | 0.608 | 0.761 | 0.816 | 0.037 | 0.148 | 0.559 | −0.006 |
| **Memory bound alone** (no motion, no filtering) | 0.615 | 0.769 | 0.819 | 0.035 | 0.146 | 0.542 | +0.001 |

Byte-level verification against the full model's 375 trajectory files
([`scripts/compare_trajectories.py`](../scripts/compare_trajectories.py); raw scores and the
reproduction command are in [knockout_scores.txt](knockout_scores.txt)):

| Variant | identical | differ |
|---|---|---|
| − filter lifecycle | **375** | 0 |
| − ambiguity gate | **375** | 0 |
| − entire motion pathway | **375** | 0 |
| − conditional filtering | 14 | 361 |
| − speed gate | 1 | 374 |
| unconditional blend | 0 | 375 |
| memory bound alone | 14 | 361 · *and 375/375 identical to “− conditional filtering”* |

Readings:

1. **The motion pathway contributes exactly nothing.** Its nullity is exact, not statistical.
2. **When it does execute, it hurts.** Forcing the speed gate open costs −0.010 Q with every
   secondary measure degrading; the unconditional blend loses −0.006. Note the forced-gate variant
   loses *more* than the blend despite discarding less machinery — its motion weight is 0.5 vs 0.2.
   Aggressive motion on ambiguous frames is worse than mild motion everywhere.
3. **Conditional post-processing is mildly harmful** (+0.001 Q when removed), and only touches
   mask-shape metrics — NRE and ADQ are untouched, a coherent signature.
4. **The ablation harness is sound** — knock-outs of live components change 361–375 sequences.
5. **The memory bound carries the entire gain — and this was predicted, then confirmed.** Before
   running it, we predicted that a variant keeping *only* the memory bound (motion pathway *and*
   conditional filtering both removed) would be indistinguishable from the "− conditional
   filtering" row, since the extra thing it drops was never running anyway. It is: byte-identical
   on all 375 sequences. A tracker that is the baseline plus one bounded memory window recovers
   the whole published improvement. The DRM admission rules, promotion thresholds and bank cap
   (`DRM_MAX_FRAMES=30`) are identical between baseline and variants — verified in source.

## A control where the memory bound provably does nothing (VOST-val)

The mechanism above makes a sharp prediction. If the window acts by truncating SAM2's *unbounded
backward search* for a frame in which the target was last visible, then where the target never
disappears the search never has to travel far, the window never truncates a lookup that would
otherwise have succeeded, and the bound must be a **bit-exact no-op** — however long the sequence.

[VOST-val](https://www.vostdataset.org/) supplies that regime: 70 sequences of slow, kitchen-scale
manipulation with public ground truth, in which targets are transformed and divided rather than
leaving the frame. Comparing `ko_postproc` (the memory bound with the conditional filter removed
and the motion pathway inert — i.e. baseline + `MEM_WINDOW` only) against the baseline:

```bash
python scripts/compare_trajectories.py --workspace workspace_vost_val \
       --reference dam4sam_b --variants dam4sam_ko_postproc
#   dam4sam_ko_postproc   107 identical / 0 differ   <- byte-identical on all files
```

**107 / 107 target-sequences byte-identical**, across the 69 sequences both trackers complete —
including all **13** sequences longer than `W`, on which the window evicts up to **155** frames of
history (`4030_cut_broccoli`, 455 frames). Not one output bit changes. The bound is inert there not
because nothing was pruned, but because nothing that was pruned would ever have been read.

Two things follow. The first is the scope condition: the window buys nothing on video without
absences, which is consistent with the gain being confined to NRE with DRE unchanged. The second
is a stronger form of the elimination argument above — if any *other* live behavioural difference
existed between the V3 chassis and the baseline, this run could not have come out bit-exact. The
"verified in source" claim of reading 5 is therefore also a measured one.

*(The one excluded sequence, `1203_cut_mango`, has seven concurrent targets, on which the
concurrent multi-object wrapper used for these VOST-val runs emits no output. The failure predates
the ablation, is independent of the memory bound, and does not affect the VOTS2026 runs, which
score objects sequentially via [`scripts/run_eval_170.py`](../scripts/run_eval_170.py).)*

## Diagnosis: a gate that cannot open

Provable inertness demands an explanation. Three individually reasonable facts compose into a
deadlock:

1. The speed gate tests the filter's velocity estimate, initialised to zero.
2. That estimate can only become non-zero through the filter's **prediction** step. The initial
   covariance is block-diagonal, so the Kalman gain's velocity rows are identically zero and the
   *update* step cannot move velocity; position–velocity correlation is introduced only by
   `P ← F P Fᵀ + Q` inside prediction. Empirically: driving the filter with a target moving at
   8 px/frame and calling `update()` alone leaves velocity at exactly `0.0` after 15 steps, whereas
   the standard predict–update cycle converges to 8.00 and opens the gate.
3. `predict()` is invoked at exactly one site — inside the branch guarded by the gates
   ([`dam4sam_tracker_v3.py:144`](../dam4sam/dam4sam_tracker_v3.py#L144)).

Gate needs velocity → velocity needs `predict()` → `predict()` needs the gate. The branch is
unreachable by construction, and no input distribution can revive it. Facts 1 and 2 are
reproducible on the filter alone, with no GPU and no dataset:

```bash
python scripts/test_gate_deadlock.py
#   update() only     : vx=0.0000000000  rho=0.0000  gate=False
#   predict()+update(): vx=7.9999532934  rho=0.0800  gate=True
#   max |K[4:, :]|    : 0.0e+00   (velocity rows of the Kalman gain)
```

This explains every row of the table: knock-outs of the lifecycle and ambiguity gate modify code
that never runs; the two variants that differ are precisely the two that bypass the speed gate.
The development-time evidence that had persuaded us the module worked came from a proxy emptiness
metric on a handful of sequences, where the differences we attributed to gated motion are equally
consistent with the memory window introduced in the same revision.

The failure mode is general: **any design that gates a prior on a statistic only that prior can
estimate risks the same circularity**, and it is invisible to unit tests of the components in
isolation — the filter is correct, the gate is correct, and the composition is unreachable.

## Lessons

1. **Metric-level ablation cannot distinguish *inert* from *absent*.** Three identical rows read
   naturally as "these components are redundant." Byte-level comparison of stored predictions says
   something categorically stronger. The check costs one pass over the output files and needs no
   ground truth.
2. **Instrument the gate, not only the output.** A counter of activations would have exposed the
   deadlock immediately, at the cost of one integer. We validated the module's *behaviour* without
   ever confirming that it *ran*.
3. **A held-out win does not license a mechanistic claim.** The result is real and was obtained on
   sequestered data; the advertised mechanism was still false.
4. **A proxy metric is faithful only on its own domain.** A win on a hand-picked hard subset does
   not transfer to the full distribution.
5. **Cross-domain extrapolation is the main trap.** Single-object LaSOT signals (D2's 4× drift
   reduction) did not survive contact with real multi-object sequences.
6. **Infrastructure rigor pays.** Smoke tests before long runs, strict integrity checks before
   packing, resumable runners and an incremental rescue puller for preemptible cloud GPUs each
   prevented at least one lost run or corrupted submission.
