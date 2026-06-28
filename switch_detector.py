"""
Switch-detector §4.3 (offline, pe seriile LaSOT) + evaluarea portii COMBINATE D1:
    veto = suspect[t] (switch-detector)  AND  identitate (cos_hiera_last cu margine)

Detector (training-free, doar din trecutul traiectoriei alese):
    jump[t]  = |c_t - c_{t-1}| / sqrt(max(area_{t-1}, A_MIN))   (salt centroid scale-norm)
    aratio[t] = max(a_t, a_{t-1}) / max(min(a_t, a_{t-1}), A_MIN)
    suspect[t] = (jump > T_JUMP) sau (aratio > T_AREA), tinut aprins HOLD cadre.

Evaluare pe v7probe npz (au si semnalele de identitate):
  - recall pe onset-urile de drift GT (suspect aprins in [onset-2, onset+RESCUE_WIN])
  - rata de cadre suspecte pe cadrele curate (cat de des se aprinde degeaba)
  - poarta combinata: salvari (din onset-urile salvabile) vs fals-pozitive reale
    (suspect ∧ rival>ales+m pe cadre curate) -> celula de operare pt D1.
"""
import itertools
from pathlib import Path

import numpy as np

OUT = Path("/home/alexx/Projects/VOTSTracker/lasot_drift_out/v7probe")
PERSIST, RESCUE_WIN, A_MIN = 5, 5, 25.0
SIG = "probe_cos_hiera_last"


def centers(boxes):
    return np.stack([boxes[..., 0] + boxes[..., 2] / 2,
                     boxes[..., 1] + boxes[..., 3] / 2], axis=-1)


def load_seq(f):
    d = np.load(f)
    frames = d["probe_frame"].astype(int)
    cb = d["probe_cand_boxes"]
    chosen = d["probe_chosen"].astype(int)
    gt, invalid = d["gt"], d["invalid"].astype(bool)
    N, M, _ = cb.shape
    cc = centers(cb)
    chosen_boxes = np.array([cb[i, chosen[i]] for i in range(N)])
    chosen_c = centers(chosen_boxes)
    areas = chosen_boxes[:, 2] * chosen_boxes[:, 3]
    valid = np.zeros(N, dtype=bool)
    out = np.ones((N, M), dtype=bool)
    for i in range(N):
        fr = frames[i]
        if fr >= len(gt) or invalid[fr] or gt[fr, 2] <= 0:
            continue
        valid[i] = True
        bx, by, bw, bh = gt[fr]
        for j in range(M):
            x, y = cc[i, j]
            if not np.isnan(x):
                out[i, j] = not (bx <= x <= bx + bw and by <= y <= by + bh)
    chosen_out = np.array([out[i, chosen[i]] for i in range(N)])
    return d, N, M, chosen, chosen_c, areas, valid, out, chosen_out


def suspect_flags(chosen_c, areas, T_JUMP, T_AREA, HOLD):
    N = len(areas)
    s = np.zeros(N, dtype=bool)
    hold = 0
    for t in range(1, N):
        a_prev = max(areas[t - 1] if not np.isnan(areas[t - 1]) else 0.0, A_MIN)
        fired = False
        if not (np.isnan(chosen_c[t]).any() or np.isnan(chosen_c[t - 1]).any()):
            jump = np.linalg.norm(chosen_c[t] - chosen_c[t - 1]) / np.sqrt(a_prev)
            if jump > T_JUMP:
                fired = True
        if not (np.isnan(areas[t]) or np.isnan(areas[t - 1])):
            lo = max(min(areas[t], areas[t - 1]), A_MIN)
            hi = max(areas[t], areas[t - 1], A_MIN)
            if hi / lo > T_AREA:
                fired = True
        hold = HOLD if fired else max(hold - 1, 0)
        s[t] = hold > 0
    return s


def onsets_of(valid, chosen_out):
    res, run, i, N = [], 0, 0, len(valid)
    while i < N:
        if not valid[i]:
            i += 1; continue
        if chosen_out[i]:
            run += 1
            if run == PERSIST:
                res.append(i - PERSIST + 1)
                while i < N and (not valid[i] or chosen_out[i]):
                    i += 1
                run = 0
                continue
        else:
            run = 0
        i += 1
    return res


def evaluate(T_JUMP, T_AREA, HOLD, margin, verbose=False):
    tot_on, det_on, saves, fp, fp_harm, clean, susp_clean = 0, 0, 0, 0, 0, 0, 0
    for f in sorted(OUT.glob("*.npz")):
        d, N, M, chosen, chosen_c, areas, valid, out, chosen_out = load_seq(f)
        sig = d[SIG]
        s = suspect_flags(chosen_c, areas, T_JUMP, T_AREA, HOLD)
        ons = onsets_of(valid, chosen_out)
        tot_on += len(ons)
        for o in ons:
            win = range(max(o - 2, 0), min(o + RESCUE_WIN + 1, N))
            if any(s[k] for k in win):
                det_on += 1
                if any(valid[k] and s[k] and j != chosen[k] and not out[k, j]
                       and sig[k][j] > sig[k][chosen[k]] + margin
                       for k in win for j in range(M)):
                    saves += 1
        cl = valid & ~chosen_out
        clean += int(cl.sum())
        susp_clean += int((cl & s).sum())
        for i in np.nonzero(cl & s)[0]:
            v = sig[i]
            rivals = [(v[j], j) for j in range(M) if j != chosen[i]
                      and not np.isnan(v[j])]
            if not rivals:
                continue
            rv, rj = max(rivals)
            if rv > v[chosen[i]] + margin:
                fp += 1
                if out[i, rj]:          # override spre candidat OUT-of-box = nociv
                    fp_harm += 1
    if verbose:
        print(f"T_JUMP={T_JUMP} T_AREA={T_AREA} HOLD={HOLD} m={margin}: "
              f"detector recall {det_on}/{tot_on}; suspect pe curate "
              f"{susp_clean}/{clean} ({susp_clean / max(clean, 1):.1%}); "
              f"POARTA COMBINATA: {saves} salvari / {fp} fp ({fp_harm} nocive)")
    return det_on, tot_on, saves, fp, fp_harm, clean, susp_clean


if __name__ == "__main__":
    print(f"semnal identitate: {SIG}\n— sweep —")
    best = None
    for T_JUMP, T_AREA, HOLD, m in itertools.product(
            [0.5, 0.8, 1.2], [2.0, 3.0], [3, 8], [0.02, 0.05]):
        det, tot, sv, fp, fph, cl, sc = evaluate(T_JUMP, T_AREA, HOLD, m)
        score = sv - fph * 0.5          # un fp ~ strica o jumatate de salvare (conservator)
        print(f"  jump>{T_JUMP} area>{T_AREA} hold={HOLD} m={m}: "
              f"recall={det}/{tot} salvari={sv} fp={fp} nocive={fph} (suspect-curate={sc}/{cl})")
        if best is None or score > best[0]:
            best = (score, (T_JUMP, T_AREA, HOLD, m))
    print("\n— config aleasa —")
    evaluate(*best[1], verbose=True)
