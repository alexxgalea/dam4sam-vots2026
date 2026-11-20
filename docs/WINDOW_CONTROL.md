# The memory window, re-tested with a single-constant control (Nov 2026)

[ABLATION.md](ABLATION.md) concluded that the bounded non-conditioning memory window
(`MEM_WINDOW = 300`) accounts for the whole improvement of `dam4sam_v3g` over the baseline
(Q 0.598 → 0.614, NRE 0.055 → 0.035). This note reports a control that isolates the window, and
corrects three statements made there and in the README.

## What changes

| Earlier statement | Status |
|---|---|
| The unbounded backward search for a visible frame is SAM2's. | **Wrong.** It is DAM4SAM's recent-appearance (RAM) selection. Vanilla SAM2 reads memory at fixed frame offsets and has no such search. SAMURAI has its own unbounded backward loop. |
| The baseline plus the window recovers the whole published improvement. | **Not established.** The baseline and the window variants were produced by different pipelines (below), so their difference is not the window's alone. |
| The window reduces missed detections. | **Not supported as a general effect.** In a clean comparison it changes the output in both directions and the net change in empty predictions is close to zero. |

What stands: the motion pathway never executed, the knock-outs that are byte-identical to the full
model are byte-identical, and the window acts through the path described, exactly where predicted.

## Why the original comparison does not isolate the window

- `dam4sam_b` (the scored baseline) was run through the VOT toolkit with
  `vot_wrapper_multiobject.py`, which runs under `torch.cuda.amp.autocast()`.
- `dam4sam_v3g` and all seven knock-outs were run with `scripts/run_eval_170.py`, which until now
  had no autocast, so they ran in fp32.

The stored trajectories show it: `dam4sam_b` and `dam4sam_ko_chassis` (the window alone) differ in
mask content on all 375 target-sequences, with the first differing frame at a median of frame 1 and
never later than frame 23. The window cannot evict anything before frame 301. On `animal`
(100 frames) the two differ from frame 2.

The VOST-val control in ABLATION.md is unaffected: there both trackers went through the same
wrapper, and they are byte-identical.

Two further facts matter for any re-run: outputs are not bit-reproducible across machines (a fresh
RTX 3060 run of a variant differs from its stored RTX 3080 run), but are deterministic on one machine.

## The control

`ko_nowindow` is `ko_chassis` with `MEM_WINDOW = 10**9`: same class, same runner, same precision,
same CPU offload, one constant different. Both were run on one RTX 3060 under autocast, on the 30
sequences in which the window was active at least once in the stored `dam4sam_ko_chassis` run
(85 target-sequences, 236,970 object-frames). Each run logs which frame filled each RAM slot
(`--log-selection`), so the first frame at which the unbounded search reaches past the window is
known exactly rather than inferred.

Checks made first, on short sequences: the upstream baseline tracker, `ko_nowindow` and `ko_chassis`
are byte-identical to each other in this pipeline; a repeated run is byte-identical; the selection
log does not change the fp32 output.

## Results

**The window acts where the account says, and nowhere else.**

| | Target-sequences | Outcome |
|---|---|---|
| Search never reaches past the window | 49 | 49 byte-identical |
| Search reaches past the window | 36 | slot selection first differs at exactly the predicted frame on 36; masks never differ before it; 8 have identical output anyway |

**Its effect on the output is not one-directional.** Counting frames on which the tracker reports
the target absent, in the 36 target-sequences where the window acts:

| With the window, compared to without | Target-sequences | Empty frames |
|---|---|---|
| fewer empty frames | 12 | −2,402 |
| unchanged | 15 | 0 |
| more empty frames | 9 | +3,185 |

Net over all 85: 43,472 → 44,255 empty frames (+1.8%). The largest gains (`racing-16` obj4 −590,
`motorcycle-1` obj4 −554, `seagull` −485) are on the objects where the stored challenge runs also
differed most; the largest loss (`bus` obj2, +1,936) outweighs any single gain. Per-trajectory
numbers are in [window_control_ac3060.csv](window_control_ac3060.csv).

An empty frame is an error only if the target is visible, and the ground truth is sequestered, so
these counts say that the window is not a consistent reduction in "target absent" predictions; they
do not say whether it helps or hurts NRE.

**An observation, not a finding.** In three of the largest gains, a long empty run under the
unbounded tracker ends after roughly `W` frames under the bounded one (`racing-16` obj4: 891 → 297
frames; `seagull`: 1,017 → 399; `tank-16` obj4: 357 → 305). That would fit the window acting as a
timeout: after about 300 empty frames the RAM slots are empty, only conditioning memory is left,
and the tracker starts reporting a mask again. Across all 36 target-sequences the excess of empty
runs ending at 290–320 frames is small (9 against 4), so this is a hypothesis.

## Not done

- **No score for the clean pair.** A server submission of `ko_nowindow` and `ko_chassis` over all
  170 sequences would give the window's actual effect on Q, NRE, DRE and ADQ. It needs `ko_nowindow`
  on the remaining 140 sequences (about 18 GPU-hours on an RTX 3060 under autocast) and `ko_chassis`
  only where the selection log shows the window acting.
- The effect was measured under autocast on one GPU. The published knock-out table is fp32.
- No second dataset. LVOS (6 fps, 574 frames per video on average) rarely contains 300 consecutive
  empty frames, so `W = 300` would be inert there almost everywhere.

## Reproduce

```bash
python scripts/download_sequences.py                       # VOTS2026 sequences, restarts on stalls
python scripts/run_window_control.py --seq-file <list> --tag ac3060
python scripts/compare_window_control.py --tag ac3060 --csv docs/window_control_ac3060.csv

# the comparison of the stored challenge trajectories (no GPU):
python scripts/compare_trajectories.py --reference dam4sam_b --variants dam4sam_ko_chassis
python scripts/predict_divergence.py --selftest
```

`run_window_control.py` passes three flags to `run_eval_170.py` that matter:

- `--strict` — without it a tracking error is recorded as an empty mask and the run continues, so
  an out-of-memory error becomes thousands of silent "target absent" predictions.
- `--blocking-copies` — under WSL2, non-blocking CPU↔GPU copies of CPU-stored frame outputs leak
  host-memory registrations until a copy fails with `CUDA error: out of memory` at under 1 GB of
  VRAM (kernel log: `dxgkio_create_allocation` / `GPADL` failures). Output is byte-identical with
  the flag.
- `--autocast` — the precision mode of the toolkit wrappers; 3.4× faster than fp32 on an RTX 3060.

`ko_nowindow` keeps every frame output on the CPU, about 1.5 MB per frame: the longest sequence
(10,699 frames) needs about 17 GB of host RAM.
