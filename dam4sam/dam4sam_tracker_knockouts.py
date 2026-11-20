"""
Per-component knock-out variants used for the ablation in docs/ABLATION.md.

Design: each variant is the full V3g tracker with EXACTLY ONE component
disabled; everything else (memory window, CPU offload, DRM banks, checkpoint,
seeds) is unchanged. Every knock-out is a *constants-and-hooks only* subclass of
DAM4SAMTrackerV3g -- no new code runs inside the tracking loop -- so any
behavioural difference is attributable to the single knob being flipped.

Because the variants differ from the full model by constants alone, a variant
whose stored masks are byte-identical to the full model's on every sequence is
not "a component that contributes nothing": it is a component that never
executed. That distinction is the point of the harness, and it is what the
comparison in scripts/compare_trajectories.py measures.

Components and their knock-outs (measured Q / NRE on the sequestered VOTS2026
set are in docs/knockout_scores.txt; full model: Q=0.614, NRE=0.035):

  ko_tie        full - ambiguity gate.
                TIE_MARGIN=1.0 makes every frame "ambiguous", so the motion
                blend (w=0.5) is applied to all candidates whenever the speed
                gate and filter health allow -- including when SAM is certain.
                Measured: byte-identical to the full model on 375/375
                trajectories. Inert, because the speed gate never opens.

  ko_speed      full - speed gate (algorithmically V3, run on the same chassis
                as V3g). Motion re-ranking fires on every ambiguous frame,
                including slow and deformable objects.
                Measured: Q 0.604 (-0.010), NRE 0.038. The only gate that is
                load-bearing -- and it is load-bearing by staying shut.

  ko_lifecycle  full - Kalman lifecycle hygiene. The filter is updated on any
                non-empty mask (no confidence threshold, no size band) and is
                never re-initialised after a loss, so a stale velocity persists.
                Measured: byte-identical on 375/375. Inert.

  ko_postproc   full - conditional connected-component filtering
                (USE_POSTPROC=False, i.e. no keep_largest at all, as in the
                baseline).
                Measured: Q 0.615 (+0.001) -- removing it helps slightly;
                361/375 trajectories differ, so the knock-out did take effect.

  ko_blend      the historical V2 behaviour reconstructed on the modern
                chassis: global blend with w=0.2 over all candidates, no gates,
                no lifecycle, unconditional keep_largest.
                Measured: Q 0.608 (-0.006), all 375 trajectories differ. Mild
                motion everywhere costs less than aggressive motion on
                ambiguous frames (see ko_speed).

  ko_motion     full - the ENTIRE motion pathway (no Kalman filter is even
                instantiated), keeping the V3 chassis: MEM_WINDOW=300 sliding
                window over non-conditioning memory + CPU offload of frame
                states.
                Measured: byte-identical on 375/375. This is the experiment
                that settled attribution: the baseline has no memory pruning,
                every V3-derived variant has it, and removing all motion code
                changes nothing -- so the gain belongs to the chassis.

  ko_chassis    memory bound alone: no motion, no conditional filtering.
                Predicted before it was run to be byte-identical to ko_postproc
                (the only thing it additionally removes is motion, which never
                ran); confirmed exactly, 375/375. Baseline + one bounded memory
                window recovers the entire published improvement.

Sanity property: none of these classes touches V3g itself, so a control rerun
of v3g stays bit-identical to the submitted results (scripts/run_knockouts.sh
performs that control before spending GPU-days).
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from dam4sam_tracker_v3g import DAM4SAMTrackerV3g


class KONoTieGate(DAM4SAMTrackerV3g):
    """full - ambiguity gate: motion re-ranking considered on every frame."""
    TIE_MARGIN = 1.0          # srt[0]-srt[1] <= 1.0 always holds for IoU in [0,1]


class KONoSpeedGate(DAM4SAMTrackerV3g):
    """full - speed gate: algorithmically identical to V3."""
    def _motion_gate_open(self):
        return True


class KONoLifecycle(DAM4SAMTrackerV3g):
    """full - filter hygiene: update on any non-empty mask, never re-init."""
    UPDATE_IOU     = -1.0     # any predicted IoU passes
    KF_GATE_PIXELS = 1        # any non-empty mask yields an update box
    SIZE_LO, SIZE_HI = 0.0, float("inf")   # no size band
    MAX_COAST      = 10**9    # _need_reinit is never set


class KONoPostproc(DAM4SAMTrackerV3g):
    """full - conditional connected-component filtering."""
    USE_POSTPROC = False


class KOBlendV2Style(DAM4SAMTrackerV3g):
    """Historical V2 reconstructed on the V3g chassis: global blend, no gates."""
    TIE_MARGIN     = 1.0      # blend over all candidates, always
    MOTION_WEIGHT  = 0.20     # the historical V2 weight (SAMURAI-style)
    UPDATE_IOU     = -1.0
    KF_GATE_PIXELS = 1
    SIZE_LO, SIZE_HI = 0.0, float("inf")
    MAX_COAST      = 10**9
    SECOND_FRAC    = 1.01     # second < 1.01*largest always holds => unconditional keep_largest

    def _motion_gate_open(self):
        return True


class KONoMotion(DAM4SAMTrackerV3g):
    """full - the entire motion pathway, chassis kept (MEM_WINDOW + offload)."""
    USE_MOTION = False


class KOChassisOnly(DAM4SAMTrackerV3g):
    """Chassis only: no motion, no conditional component filtering."""
    USE_MOTION = False
    USE_POSTPROC = False


class KOChassisNoWindow(KOChassisOnly):
    """Chassis only, memory bound switched off: the single-constant control.

    Differs from KOChassisOnly by MEM_WINDOW alone (same runner, same precision,
    same CPU offload), so any difference in output between the two is the
    window's doing. The cutoff is never positive, so nothing is ever evicted.
    """
    MEM_WINDOW = 10**9


KNOCKOUTS = {
    "ko_tie":       KONoTieGate,
    "ko_speed":     KONoSpeedGate,
    "ko_lifecycle": KONoLifecycle,
    "ko_postproc":  KONoPostproc,
    "ko_blend":     KOBlendV2Style,
    "ko_motion":    KONoMotion,
    "ko_chassis":   KOChassisOnly,
    "ko_nowindow":  KOChassisNoWindow,
}
