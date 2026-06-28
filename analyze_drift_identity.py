"""
Analiza identitate-la-drift (PLAN_TONIGHT_OPUS §3.1 / PLAN_WIN_VOTS2026 gate §4.1).

Pe rularile v7probe LaSOT (GT real, box):
  1. Gaseste onset-urile de drift ale candidatului ALES (centroid in afara gt-box,
     persistenta >= PERSIST cadre valide).
  2. "Salvabil" la onset (fereastra [onset, onset+RESCUE_WIN]): exista alt candidat
     cu centroidul IN gt-box -> plafonul teoretic al unui veto de identitate (D1).
  3. Pe cadrele salvabile: cat de des fiecare semnal (cos_ptr_last, cos_hiera_last,
     si insusi sam_iou) prefera candidatul corect (in-box) fata de cel ales (out-box)?
  4. Siguranta veto-ului: pe cadrele valide NE-drift (alesul in-box), cat de des un
     veto cu margine m ar surclasa gresit alesul? -> alege marginea la care
     fals-pozitivele ~ 0 si raporteaza cate onset-uri tot s-ar salva cu acea margine.
"""
import json
from pathlib import Path

import numpy as np

OUT = Path("/home/alexx/Projects/VOTSTracker/lasot_drift_out/v7probe")
PERSIST = 5
RESCUE_WIN = 5
MARGINS = [0.0, 0.02, 0.05, 0.10, 0.15]
SIGNALS = ["probe_ious", "probe_cos_ptr_last", "probe_cos_hiera_last",
           "probe_cos_ptr_p0", "probe_cos_hiera_p0"]


def centers(boxes):
    return np.stack([boxes[..., 0] + boxes[..., 2] / 2,
                     boxes[..., 1] + boxes[..., 3] / 2], axis=-1)


def in_box(pt, box):
    return (box[0] <= pt[0] <= box[0] + box[2]) and (box[1] <= pt[1] <= box[1] + box[3])


def main():
    onsets_total, rescuable_total = 0, 0
    # per semnal: [corect pe salvabile, total salvabile]
    pick = {s: [0, 0] for s in SIGNALS}
    # veto safety: per (semnal, margine): [fals-pozitive, cadre ne-drift], [salvari, onset-uri]
    fp = {(s, m): 0 for s in SIGNALS for m in MARGINS}
    saves = {(s, m): 0 for s in SIGNALS for m in MARGINS}
    n_clean = 0

    for f in sorted(OUT.glob("*.npz")):
        d = np.load(f)
        if "probe_cand_boxes" not in d:
            continue
        frames = d["probe_frame"].astype(int)
        cb = d["probe_cand_boxes"]            # (N, M, 4)
        chosen = d["probe_chosen"].astype(int)
        gt, invalid = d["gt"], d["invalid"].astype(bool)
        N, M, _ = cb.shape
        cc = centers(cb)                      # (N, M, 2)

        # out-of-box pentru fiecare candidat, per cadru (aliniat la frames[i])
        out = np.ones((N, M), dtype=bool)
        valid = np.zeros(N, dtype=bool)
        for i in range(N):
            fr = frames[i]
            if fr >= len(gt) or invalid[fr] or gt[fr, 2] <= 0:
                continue
            valid[i] = True
            for j in range(M):
                if not np.isnan(cc[i, j]).any():
                    out[i, j] = not in_box(cc[i, j], gt[fr])

        chosen_out = np.array([out[i, chosen[i]] for i in range(N)])

        # cadre curate (alesul in-box) -> baza pt fals-pozitive de veto
        for i in np.nonzero(valid & ~chosen_out)[0]:
            n_clean += 1
            for s in SIGNALS:
                v = d[s][i]
                rival = max((v[j] for j in range(M) if j != chosen[i]
                             and not np.isnan(v[j])), default=-np.inf)
                for m in MARGINS:
                    # veto gresit: un rival OUT-of-box sau egal surclaseaza alesul corect
                    if rival > v[chosen[i]] + m:
                        fp[(s, m)] += 1

        # onset-uri de drift (PERSIST cadre valide consecutive cu alesul out)
        run = 0
        i = 0
        while i < N:
            if not valid[i]:
                i += 1; continue
            if chosen_out[i]:
                run += 1
                if run == PERSIST:
                    onset = i - PERSIST + 1
                    onsets_total += 1
                    # fereastra de salvare
                    win = [k for k in range(onset, min(onset + RESCUE_WIN + 1, N)) if valid[k]]
                    resc_frames = [(k, j) for k in win for j in range(M)
                                   if j != chosen[k] and not out[k, j]]
                    if resc_frames:
                        rescuable_total += 1
                        for s in SIGNALS:
                            pick[s][1] += 1
                            ok = any(d[s][k][j] > d[s][k][chosen[k]]
                                     for k, j in resc_frames)
                            if ok:
                                pick[s][0] += 1
                            for m in MARGINS:
                                if any(d[s][k][j] > d[s][k][chosen[k]] + m
                                       for k, j in resc_frames):
                                    saves[(s, m)] += 1
                    # sari peste restul evenimentului
                    while i < N and (not valid[i] or chosen_out[i]):
                        i += 1
                    run = 0
                    continue
            else:
                run = 0
            i += 1

    print(f"onset-uri drift: {onsets_total} | salvabile (alt candidat in-box, "
          f"fereastra {RESCUE_WIN}): {rescuable_total} "
          f"({rescuable_total / max(onsets_total, 1):.0%}) | cadre curate: {n_clean}")
    print(f"\n{'semnal':<22} {'alege corect pe salvabile':>26}")
    for s in SIGNALS:
        k, n = pick[s]
        print(f"{s:<22} {k}/{n} ({k / max(n, 1):.0%})")
    print(f"\nVeto cu margine — salvari / fals-pozitive (din {n_clean} cadre curate):")
    print(f"{'semnal':<22} | " + " | ".join(f"m={m:.2f}" for m in MARGINS))
    for s in SIGNALS:
        row = f"{s:<22} | "
        for m in MARGINS:
            row += f"{saves[(s, m)]}sv/{fp[(s, m)]}fp".rjust(9) + " | "
        print(row)
    print("\nCitire: vrem o celula cu salvari cat mai multe si fp≈0. "
          "fp pe cadre curate = override care STRICA un cadru bun (cost direct in Q).")


if __name__ == "__main__":
    main()
