#!/usr/bin/env bash
# ============================================================================
# Knock-out ablation runner (see docs/ABLATION.md).
#
# Runs the knock-out variants sequentially over every sequence in
# workspace2026, with:
#   0. environment control: rerun v3g on the smoke sequence and byte-compare it
#      against the stored submission results (warns if the environment or the
#      checkpoint has drifted since the official run -- otherwise the knock-outs
#      would not be comparable to the published numbers)
#   1. a per-variant smoke test (one multi-object + one short sequence) before
#      committing GPU-days
#   2. the full run (resume-safe: run_eval_170.py skips completed sequences)
#   3. a strict integrity check before a variant is declared done
#
# IMPORTANT: no --max-regression / --max-catastrophes here. For knock-outs the
# regressions *are* the measurement; early stopping would truncate the data.
#
# Usage:
#   nohup scripts/run_knockouts.sh > logs/knockouts/runner.log 2>&1 &
#   VARIANTS="ko_lifecycle" scripts/run_knockouts.sh     # a single variant
#
# Cost: ~20 h per variant on an RTX 3080, i.e. several days for the full set.
# ============================================================================
set -uo pipefail

cd "$(dirname "$0")/.."                      # repo root
PY="dam4sam/.venv/bin/python3"
RUN="scripts/run_eval_170.py"
CHECK="scripts/check_results_integrity.py"
LOGDIR="logs/knockouts"
mkdir -p "$LOGDIR"

VARIANTS=(${VARIANTS:-ko_lifecycle ko_speed ko_tie ko_postproc ko_blend ko_motion ko_chassis})

log() { echo "[$(date '+%F %T')] $*"; }

# ── Pick the smoke sequences dynamically ───────────────────────────────────
# SMOKE_MULTI = the sequence with the most objects (shortest one on a tie)
#               -- catches "0 of 7 masks written" style failures
# SMOKE_SHORT = the shortest single-object sequence -- a fast end-to-end chain
read -r SMOKE_MULTI SMOKE_SHORT <<<"$("$PY" - <<'EOF'
from pathlib import Path
seqs = []
root = Path("workspace2026/sequences")
for d in sorted(root.iterdir()):
    if not d.is_dir():
        continue
    n_obj = max(1, len(list(d.glob("groundtruth_obj*.txt"))))
    n_frm = len(list((d / "color").glob("*.jpg"))) or len(list((d / "color").glob("*.png")))
    if n_frm:
        seqs.append((d.name, n_obj, n_frm))
multi = min(seqs, key=lambda s: (-s[1], s[2]))[0]
short = min((s for s in seqs if s[1] == 1), key=lambda s: s[2])[0]
print(multi, short)
EOF
)"
log "smoke sequences: multi-object=$SMOKE_MULTI short=$SMOKE_SHORT"

# ── Step 0: environment control -- v3g must still reproduce bit-identically ──
CONTROL_REF=""
for cand in workspace2026/results/dam4sam_v3g/baseline workspace2026/results/v3g/baseline; do
    [ -d "$cand/$SMOKE_SHORT" ] && CONTROL_REF="$cand" && break
done
if [ -n "$CONTROL_REF" ]; then
    log "control: rerun v3g on $SMOKE_SHORT, byte-compare against $CONTROL_REF"
    rm -rf "workspace2026/results/_ko_control/baseline/$SMOKE_SHORT"
    "$PY" "$RUN" --tracker v3g --out _ko_control --seqs "$SMOKE_SHORT" \
        >"$LOGDIR/control_v3g.log" 2>&1
    ok=1
    for f in "$CONTROL_REF/$SMOKE_SHORT"/*.bin; do
        new="workspace2026/results/_ko_control/baseline/$SMOKE_SHORT/$(basename "$f")"
        if [ ! -f "$new" ] || ! cmp -s "$f" "$new"; then ok=0; break; fi
    done
    if [ "$ok" = 1 ]; then
        log "control OK: v3g is bit-identical to the submitted run -- environment unchanged"
    else
        log "!!! CONTROL WARNING: v3g does NOT reproduce the stored results on $SMOKE_SHORT."
        log "!!! The environment or the checkpoint changed since the official run. The"
        log "!!! knock-outs stay valid RELATIVE TO EACH OTHER, but comparing them against"
        log "!!! the published v3g score is no longer clean. Investigate before reporting."
    fi
else
    log "control: no local v3g reference results -- skipping the byte comparison"
fi

# ── Run the variants ────────────────────────────────────────────────────────
for v in "${VARIANTS[@]}"; do
    OUT="dam4sam_${v}"                        # = the identifier in trackers.ini (for vot pack)
    LOG="$LOGDIR/${v}.log"
    log "================ $v -> results/$OUT ================"

    # 1. Smoke: multi-object + short. They land in the same output directory,
    #    so the full run skips them afterwards -- no work is thrown away.
    log "[$v] smoke: $SMOKE_MULTI,$SMOKE_SHORT"
    "$PY" "$RUN" --tracker "$v" --out "$OUT" --seqs "$SMOKE_MULTI,$SMOKE_SHORT" \
        >>"$LOG" 2>&1
    smoke_fail=0
    for s in "$SMOKE_MULTI" "$SMOKE_SHORT"; do
        n_expected="$("$PY" -c "
from pathlib import Path
d = Path('workspace2026/sequences/$s')
print(max(1, len(list(d.glob('groundtruth_obj*.txt')))))")"
        n_found="$(find "workspace2026/results/$OUT/baseline/$s" -name '*.bin' -size +0c 2>/dev/null | wc -l)"
        if [ "$n_found" -lt "$n_expected" ]; then
            log "[$v] SMOKE FAIL on $s: $n_found/$n_expected non-empty .bin files"
            smoke_fail=1
        fi
    done
    if [ "$smoke_fail" = 1 ]; then
        log "[$v] variant ABANDONED (see $LOG). Moving on to the next one."
        continue
    fi
    log "[$v] smoke OK"

    # 2. Full run -- no greedy early stop, the regressions are the data
    log "[$v] full run start (~20 h on an RTX 3080; resumable after a crash)"
    "$PY" "$RUN" --tracker "$v" --out "$OUT" >>"$LOG" 2>&1
    log "[$v] full run finished (exit=$?)"

    # 3. Strict integrity check -- the "ready to pack" condition
    if "$PY" "$CHECK" --workspace workspace2026 --tracker "$OUT" >>"$LOG" 2>&1; then
        log "[$v] INTEGRITY OK -- ready to pack:"
        log "[$v]   vot pack --workspace workspace2026 $OUT"
    else
        log "[$v] !!! INTEGRITY FAIL -- do not submit. Details in $LOG"
    fi
done

log "runner finished. Logs in $LOGDIR/."
log "Byte-level comparison against the full model:"
log "  python scripts/compare_trajectories.py --reference dam4sam_v3g --variants 'dam4sam_ko_*'"
