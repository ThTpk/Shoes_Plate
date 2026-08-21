#!/usr/bin/env python3
"""Screen every participant's barefoot pressure data and propose an insole spec.

For each participant this measures, on the instant of peak loading of each
usable footstep:

  A        contact area                              (cm^2)
  F        total force                               (N)
  p_max    highest local pressure                    (kPa)
  p*       F / A, the pressure if perfectly spread   (kPa)
  PPR      p_max / p*, how concentrated the load is  (-)
  AI       arch index, midfoot area / footprint area, toes excluded
  zone     where p_max sits: toes / forefoot / midfoot / heel

The median over that participant's footsteps becomes their profile, which maps
to an insole specification through the rule table in `classify`.

This is an engineering screen over pressure data, not a clinical prescription.
"""

import argparse
import pathlib
import sys

import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / 'FRDR_dataset' / 'example_code' / 'python'))
from utils import load_footsteps, load_metadata  # noqa: E402

PX_MM = 5.0
PIX_AREA_M2 = (PX_MM / 1000) ** 2
PIX_AREA_CM2 = (PX_MM / 10) ** 2
CONTACT_KPA = 20.0   # load-bearing contact, used for force and contact area
PRINT_KPA = 5.0      # footprint outline, used for the arch index and foot geometry.
                     # The classic ink-print arch index registers very light contact,
                     # so a 20 kPa cut-off erases real midfoot contact and biases the
                     # index low (median 0.168 at 20 kPa vs 0.218 at 5 kPa).

# arch index bands (Cavanagh & Rodgers 1987)
AI_HIGH, AI_FLAT = 0.21, 0.26
# how concentrated the peak is, relative to a perfectly spread load
PPR_MILD, PPR_SEVERE = 4.0, 6.0


def toe_cut(mask):
    """Row index separating the toes from the sole body.

    In a barefoot print the toes are joined to the forefoot by a narrow neck.
    Look for the narrowest row in the upper third and cut there; if no clear
    neck exists, fall back to dropping the top 15%.
    """
    width = mask.sum(1)
    rows = np.flatnonzero(width)
    r0, r1 = rows[0], rows[-1]
    h = r1 - r0 + 1
    lo, hi = r0 + int(0.10 * h), r0 + int(0.35 * h)
    if hi <= lo + 1:
        return r0 + int(0.15 * h)
    band = width[lo:hi]
    neck = lo + int(np.argmin(band))
    # a real neck is clearly narrower than the forefoot just below it
    below = width[neck:r1 + 1]
    if len(below) and width[neck] < 0.75 * np.median(below[below > 0]):
        return neck
    return r0 + int(0.15 * h)


def measure_step(ppi, frame):
    """Measurements for one footstep.

    Footprint shape (arch index, zone of the peak) comes from `ppi`, the peak
    over the whole stance - that is the footprint the arch index is defined on.
    Force and pressure come from `frame`, the single instant of peak load, so
    that F = integral(p dA) is a physically consistent snapshot.
    """
    mask = ppi > PRINT_KPA
    if mask.sum() < 50:
        return None
    p = np.where(mask, ppi, 0.0)
    inst = np.where(frame > CONTACT_KPA, frame, 0.0)
    rows = np.flatnonzero(mask.any(1))
    r0, r1 = rows[0], rows[-1]
    h = r1 - r0 + 1

    A_print = (ppi > CONTACT_KPA).sum() * PIX_AREA_CM2       # load-bearing footprint
    A = max((inst > 0).sum() * PIX_AREA_CM2, 1e-6)            # contact at the peak instant
    F = float(inst.sum() * 1000.0 * PIX_AREA_M2)
    p_star = F / (A / 1e4) / 1000.0
    p_max = float(inst.max())

    # arch index on the footprint with toes removed
    cut = toe_cut(mask)
    sole = mask[cut:r1 + 1]
    hs = sole.shape[0]
    third = hs / 3.0
    fore = sole[:int(third)].sum()
    mid = sole[int(third):int(2 * third)].sum()
    hind = sole[int(2 * third):].sum()
    total = fore + mid + hind
    AI = mid / total if total else np.nan

    # which zone carries the peak
    ry = np.unravel_index(np.argmax(inst), inst.shape)[0]
    f = (ry - r0) / max(h - 1, 1)
    zone = 'toes' if f < 0.20 else 'forefoot' if f < 0.45 else 'midfoot' if f < 0.70 else 'heel'

    return dict(A=A, A_print=A_print, F=F, p_star=p_star, p_max=p_max,
                PPR=p_max / p_star if p_star else np.nan,
                AI=AI, zone=zone, length_cm=h * PX_MM / 10,
                width_cm=mask.any(0).sum() * PX_MM / 10)


def classify(row, k0, max_carve):
    """Map a participant profile onto an insole specification."""
    ai, ppr, zone = row.AI, row.PPR, row.zone

    foot = 'high arch' if ai < AI_HIGH else 'flat' if ai > AI_FLAT else 'normal'
    sev = 'mild' if ppr < PPR_MILD else 'severe' if ppr > PPR_SEVERE else 'moderate'

    if foot == 'high arch':
        base = 'cushioned full-contact'
        why = 'small midfoot contact concentrates load on heel and forefoot'
    elif foot == 'flat':
        base = 'supportive with medial arch post'
        why = 'midfoot already loaded; support rather than fill'
    else:
        base = 'semi-rigid contoured'
        why = 'balanced loading'

    add = {'forefoot': 'metatarsal pad proximal to the hot spot',
           'toes': 'forefoot extension with toe relief',
           'heel': 'heel cup with cushioning insert',
           'midfoot': 'arch conformity only'}[zone]

    carve_ideal = row.p_max / k0
    carve = min(carve_ideal, max_carve)
    # residual peak once the carve is capped, before the force rebalances
    resid = max(row.p_star, row.p_max - k0 * carve)
    thickness = 10.0 if carve_ideal <= 2.0 else 15.0 if carve_ideal <= 3.0 else 20.0

    return pd.Series(dict(
        foot_type=foot, severity=sev, base=base, addition=add, rationale=why,
        carve_ideal_mm=round(carve_ideal, 2), carve_applied_mm=round(carve, 2),
        base_thickness_mm=thickness,
        predicted_peak_kPa=round(resid, 0),
        predicted_reduction_pct=round(100 * (1 - resid / row.p_max), 0),
        priority=round(ppr, 2)))


def main():
    ap = argparse.ArgumentParser(description='Propose an insole spec for every participant.')
    ap.add_argument('--participants', default='1-150')
    ap.add_argument('--footwear', default='BF')
    ap.add_argument('--speed', default='W1', help='trial to profile (default: W1, normal walking)')
    ap.add_argument('--modulus', type=float, default=2.0, help='foam E in MPa')
    ap.add_argument('--thickness', type=float, default=10.0, help='reference foam thickness in mm')
    ap.add_argument('--max-carve', type=float, default=2.0, help='deepest contour cut in mm')
    ap.add_argument('--out', default='insole_plan.csv')
    ap.add_argument('--steps-out', default=None, help='optional per-footstep CSV')
    args = ap.parse_args()

    a, b = (args.participants.split('-') + [None])[:2]
    ids = range(int(a), int(b) + 1) if b else [int(a)]
    k0 = args.modulus * 1000.0 / args.thickness

    per_step = []
    for n, pid in enumerate(ids, 1):
        try:
            steps = load_footsteps(pid, args.footwear, args.speed, pipeline=1)
            md = load_metadata(pid, args.footwear, args.speed)
        except FileNotFoundError:
            print(f'  P{pid:03d}: no data, skipped')
            continue
        for i in md.index[md.Exclude == 0]:
            f = steps[i]
            m = measure_step(f.max(0), f[f.sum((1, 2)).argmax()])
            if m:
                m.update(participant=pid, side=str(md.Side[i])[0], footstep=int(md.FootstepID[i]))
                per_step.append(m)
        if n % 25 == 0:
            print(f'  {n}/{len(list(ids))} participants', flush=True)

    S = pd.DataFrame(per_step)
    if args.steps_out:
        S.to_csv(args.steps_out, index=False)

    num = S.groupby('participant')[['A', 'A_print', 'F', 'p_star', 'p_max', 'PPR', 'AI',
                                    'length_cm', 'width_cm']].median()
    zone = S.groupby('participant').zone.agg(lambda s: s.mode().iat[0])
    prof = num.join(zone).join(S.groupby('participant').size().rename('n_steps'))
    plan = prof.join(prof.apply(classify, axis=1, k0=k0, max_carve=args.max_carve))
    plan = plan.sort_values('priority', ascending=False)
    plan.round(3).to_csv(args.out)

    print(f'\n{len(plan)} participants profiled from {len(S)} footsteps '
          f'({args.footwear} {args.speed})\n')
    print('foot type      ', plan.foot_type.value_counts().to_dict())
    print('severity       ', plan.severity.value_counts().to_dict())
    print('peak zone      ', plan.zone.value_counts().to_dict())
    print('\nrecommended base:')
    for k, v in plan.base.value_counts().items():
        print(f'  {v:4d}  {k}')
    print('\nadditions:')
    for k, v in plan.addition.value_counts().items():
        print(f'  {v:4d}  {k}')
    print(f'\narch index AI: median {plan.AI.median():.3f} '
          f'(range {plan.AI.min():.3f}-{plan.AI.max():.3f})')
    print(f'PPR: median {plan.PPR.median():.2f} (range {plan.PPR.min():.2f}-{plan.PPR.max():.2f})')
    print(f'predicted peak reduction: median {plan.predicted_reduction_pct.median():.0f}%')
    print(f'\ntop 10 priority (most concentrated loading):')
    cols = ['foot_type', 'severity', 'zone', 'p_max', 'PPR', 'carve_ideal_mm',
            'predicted_reduction_pct']
    print(plan[cols].head(10).round(2).to_string())
    print(f'\nfull plan: {pathlib.Path(args.out).resolve()}')


if __name__ == '__main__':
    main()
