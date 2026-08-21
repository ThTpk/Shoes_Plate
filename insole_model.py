#!/usr/bin/env python3
"""Mathematical model of a contoured insole designed from a flat-plate pressure map.

The physics is a Winkler elastic foundation: the insole behaves as a bed of
independent springs of stiffness k0 = E / h0 (pressure per unit compression).

On a FLAT plate the foot sinks by u0 and the measured pressure is

    p_flat(x,y) = k0 * max(0, u0 - g(x,y))                                  (1)

where g(x,y) is the height of the sole surface above its lowest point. Equation
(1) inverts to give the sole's shape directly from the pressure map:

    g(x,y) = u0 - p_flat(x,y) / k0                                          (2)

so a high-pressure spot is simply a part of the foot that sticks out further.

Carving the insole surface down by c(x,y) makes the new pressure

    p_new(x,y) = max(0, p_flat(x,y) - k0 * c(x,y) + k0 * D)                 (3)

with D a rigid-body sink chosen so that the total force is unchanged. Setting
c = p_flat / k0 makes (3) constant: a perfectly conforming insole equalises
pressure. Real foam cannot be carved arbitrarily deep, so c is clipped at
c_max and (3) is solved for D by bisection.

The alternative lever is stiffness rather than shape. Keeping the surface flat
and varying the local modulus gives uniform pressure when

    k(x,y) = k0 * p_target / p_flat(x,y)                                    (4)

i.e. the foam must be SOFTER under the hot spots.
"""

import argparse
import pathlib
import sys

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / 'FRDR_dataset' / 'example_code' / 'python'))

PX_MM = 5.0                      # sensor pitch: 0.5 cm
PIX_AREA_M2 = (PX_MM / 1000) ** 2
CONTACT_KPA = 20.0               # below this a pixel is not in contact


def load_peak_image(participant, footwear, speed, step, source):
    """Return the pressure map to design against.

    'ppi'        peak over the whole stance - the clinical standard, but the
                 peaks happen at different instants so its integral exceeds
                 body weight;
    'peak-frame' the single instant of largest total force - a physically
                 consistent snapshot, integrating to roughly body weight.
    """
    from utils import load_footsteps, load_metadata
    steps = load_footsteps(participant, footwear, speed, pipeline=1)
    md = load_metadata(participant, footwear, speed)
    rows = md.index[md.Exclude == 0]
    if step >= len(rows):
        raise SystemExit(f'step {step} out of range (trial has {len(rows)} usable footsteps)')
    f = steps[rows[step]]
    img = f.max(0) if source == 'ppi' else f[f.sum((1, 2)).argmax()]
    return img, md.iloc[rows[step]]


def force_N(p_kpa):
    """Total force carried by a pressure map, in newtons."""
    return float(p_kpa.sum() * 1000.0 * PIX_AREA_M2)


def simulate(p_flat, carve_mm, k0, mask):
    """Pressure after carving, with the rigid-body sink D fixed by force balance."""
    target_F = force_N(p_flat)

    def total(D):
        return force_N(np.where(mask, np.maximum(0.0, p_flat - k0 * carve_mm + k0 * D), 0.0))

    lo, hi = -50.0, 50.0
    for _ in range(200):
        mid = (lo + hi) / 2
        if total(mid) < target_F:
            lo = mid
        else:
            hi = mid
    D = (lo + hi) / 2
    return np.where(mask, np.maximum(0.0, p_flat - k0 * carve_mm + k0 * D), 0.0), D


def main():
    ap = argparse.ArgumentParser(description='Design a contoured insole from a pressure map.')
    ap.add_argument('--participant', type=int, default=9)
    ap.add_argument('--footwear', default='BF')
    ap.add_argument('--speed', default='W1')
    ap.add_argument('--step', type=int, default=10, help='index among usable footsteps')
    ap.add_argument('--modulus', type=float, default=2.0, help='foam Young modulus E in MPa')
    ap.add_argument('--thickness', type=float, default=10.0, help='foam thickness h0 in mm')
    ap.add_argument('--max-carve', type=float, default=4.0, help='deepest contour cut in mm')
    ap.add_argument('--max-strain', type=float, default=0.30,
                    help='strain beyond which the linear foam model is not trusted')
    ap.add_argument('--source', choices=('ppi', 'peak-frame'), default='ppi',
                    help="'ppi' = peak-pressure image (clinical standard); "
                         "'peak-frame' = the instant of largest total force")
    ap.add_argument('--figure', default=None, help='write a comparison figure to this path')
    args = ap.parse_args()

    p, meta = load_peak_image(args.participant, args.footwear, args.speed, args.step, args.source)
    mask = p > CONTACT_KPA
    p = np.where(mask, p, 0.0)

    E_kpa = args.modulus * 1000.0
    k0 = E_kpa / args.thickness                     # kPa per mm
    A_cm2 = mask.sum() * (PX_MM / 10) ** 2
    F = force_N(p)
    p_uniform = F / (A_cm2 / 1e4) / 1000.0          # kPa

    print(f'footstep: P{args.participant:03d} {args.footwear} {args.speed} '
          f'step {int(meta.FootstepID)} ({meta.Side})')
    print(f'foam: E = {args.modulus} MPa, h0 = {args.thickness} mm  ->  '
          f'k0 = {k0:.0f} kPa/mm')
    print()
    print(f'contact area      A = {A_cm2:.1f} cm^2')
    print(f'equivalent load   F = {F:.0f} N   (source: {args.source})')
    print(f'measured peak     p_max = {p.max():.0f} kPa')
    print(f'measured mean     p_avg = {p[mask].mean():.0f} kPa')
    print(f'ideal uniform     p* = F/A = {p_uniform:.0f} kPa'
          f'   -> best possible reduction {100 * (1 - p_uniform / p.max()):.0f}%')
    print()

    # ---- design A: contoured surface ------------------------------------
    carve_ideal = p / k0
    carve = np.clip(carve_ideal, 0, args.max_carve)
    p_new, D = simulate(p, carve, k0, mask)
    strain = (carve + D) / args.thickness

    print('design A - contoured surface  c(x,y) = p_flat / k0, clipped')
    print(f'  ideal carve depth  max {carve_ideal.max():.2f} mm '
          f'(clipped at {args.max_carve} mm), mean {carve_ideal[mask].mean():.2f} mm')
    print(f'  rigid-body sink    D = {D:.2f} mm')
    print(f'  new peak pressure  {p_new.max():.0f} kPa  '
          f'(was {p.max():.0f})  ->  reduction {100 * (1 - p_new.max() / p.max()):.0f}%')
    print(f'  new mean pressure  {p_new[mask].mean():.0f} kPa')
    print(f'  force check        {force_N(p_new):.0f} N vs {F:.0f} N')
    print(f'  peak strain        {strain.max():.0%}'
          + ('  <-- beyond the linear range, treat as indicative'
             if strain.max() > args.max_strain else '  (within linear range)'))

    # ---- design B: graded stiffness -------------------------------------
    with np.errstate(divide='ignore', invalid='ignore'):
        k_needed = np.where(mask, k0 * p_uniform / np.maximum(p, 1e-9), np.nan)
    kn = k_needed[mask]
    print()
    print('design B - graded stiffness  k(x,y) = k0 * p* / p_flat')
    print(f'  required k range   {np.nanmin(kn):.0f} - {np.nanmax(kn):.0f} kPa/mm '
          f'(ratio {np.nanmax(kn) / np.nanmin(kn):.0f}x)')
    print(f'  as modulus         E = {np.nanmin(kn) * args.thickness / 1000:.2f} - '
          f'{np.nanmax(kn) * args.thickness / 1000:.2f} MPa')
    print('  softest foam goes under the hot spots')

    if args.figure:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        jet = matplotlib.colormaps['jet'](np.linspace(0, 1, 256))
        jet[0] = [0, 0, 0, 1]
        cmap = matplotlib.colors.ListedColormap(jet)
        vmax = p.max()
        fig, ax = plt.subplots(1, 4, figsize=(13, 5))
        for a in ax:
            a.axis('off')
        im0 = ax[0].imshow(p, cmap=cmap, vmin=0, vmax=vmax)
        ax[0].set_title(f'measured p_flat\npeak {p.max():.0f} kPa')
        fig.colorbar(im0, ax=ax[0], shrink=.7, label='kPa')
        im1 = ax[1].imshow(np.where(mask, carve, np.nan), cmap='viridis')
        ax[1].set_title(f'insole carve depth\nmax {carve.max():.2f} mm')
        fig.colorbar(im1, ax=ax[1], shrink=.7, label='mm')
        im2 = ax[2].imshow(p_new, cmap=cmap, vmin=0, vmax=vmax)
        ax[2].set_title(f'pressure on insole\npeak {p_new.max():.0f} kPa')
        fig.colorbar(im2, ax=ax[2], shrink=.7, label='kPa')
        ax[3].axis('on')
        bins = np.linspace(0, vmax, 41)
        ax[3].hist(p[mask], bins=bins, alpha=.6, label='flat plate')
        ax[3].hist(p_new[p_new > 0], bins=bins, alpha=.6, label='on insole')
        ax[3].axvline(p_uniform, color='k', ls='--', lw=1, label='ideal p*')
        ax[3].set_xlabel('pressure (kPa)')
        ax[3].set_ylabel('pixels')
        ax[3].legend()
        ax[3].set_title('pressure distribution')
        fig.suptitle(f'Insole model - P{args.participant:03d} {args.footwear} {args.speed} '
                     f'step {int(meta.FootstepID)}   (E={args.modulus} MPa, '
                     f'h0={args.thickness} mm, max carve {args.max_carve} mm)')
        fig.tight_layout()
        fig.savefig(args.figure, dpi=130)
        print(f'\nfigure: {pathlib.Path(args.figure).resolve()}')


if __name__ == '__main__':
    main()
