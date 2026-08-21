#!/usr/bin/env python3
"""Turn a pressure map into an insole surface: an explicit function, a 3D view, an STL.

The carve depth from the Winkler model is a pixel map, which is not something a
machinist or a CAD package can use. This fits it with two closed-form surfaces:

  polynomial   c(u,v) = sum_ij a_ij u^i v^j            compact, no anatomy in it
  Gaussians    c(u,v) = sum_k A_k exp(-(u-u_k)^2/2s_k^2 - (v-v_k)^2/2t_k^2)

with u across the foot and v along it, both normalised to [-1, 1]. The Gaussian
form is the interesting one: each term is a physical feature of the insole - a
heel cup, an arch fill, a metatarsal relief - so its parameters can be read and
adjusted by hand.

The insole top surface is  z(x,y) = h0 - c(x,y),  which is what gets rendered
and written to STL.
"""

import argparse
import pathlib
import struct
import sys

import numpy as np
from scipy.ndimage import gaussian_filter, binary_dilation, binary_fill_holes
from scipy.optimize import curve_fit

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / 'FRDR_dataset' / 'example_code' / 'python'))

PX_MM = 5.0
CONTACT_KPA = 20.0


# --------------------------------------------------------------------------
# surface models
# --------------------------------------------------------------------------
def poly_design(u, v, degree):
    """Design matrix for a bivariate polynomial, terms ordered by total degree."""
    cols, names = [], []
    for total in range(degree + 1):
        for i in range(total + 1):
            j = total - i
            cols.append((u ** i) * (v ** j))
            names.append((i, j))
    return np.column_stack(cols), names


def fit_poly(u, v, c, degree):
    A, names = poly_design(u, v, degree)
    coef, *_ = np.linalg.lstsq(A, c, rcond=None)
    pred = A @ coef
    return coef, names, pred


def gauss_sum(uv, *p):
    u, v = uv
    out = np.zeros_like(u)
    for k in range(len(p) // 5):
        A, u0, v0, su, sv = p[5 * k:5 * k + 5]
        out = out + A * np.exp(-((u - u0) ** 2 / (2 * su ** 2) + (v - v0) ** 2 / (2 * sv ** 2)))
    return out


def fit_gaussians(u, v, c, k):
    """Seed one bump per anatomical band, then let least squares move them."""
    seeds = np.linspace(0.65, -0.75, k)          # v: toes at +1, heel at -1
    p0, lo, hi = [], [], []
    for s in seeds:
        p0 += [float(c.max()) * 0.7, 0.0, float(s), 0.45, 0.30]
        lo += [0.0, -1.5, -1.5, 0.05, 0.05]
        hi += [float(c.max()) * 3 + 1e-6, 1.5, 1.5, 3.0, 3.0]
    popt, _ = curve_fit(gauss_sum, (u, v), c, p0=p0, bounds=(lo, hi), maxfev=200000)
    return popt, gauss_sum((u, v), *popt)


def r2(c, pred):
    return 1 - np.sum((c - pred) ** 2) / np.sum((c - c.mean()) ** 2)


# --------------------------------------------------------------------------
# geometry
# --------------------------------------------------------------------------
def insole_outline(mask, pad):
    """Grow the contact area into a plausible insole outline."""
    out = binary_dilation(mask, np.ones((3, 3), bool), iterations=pad)
    return binary_fill_holes(out)


def write_stl(path, X, Y, Z, Zb, inside):
    """Watertight binary STL of the solid between the Zb and Z faces."""
    tris = []

    def quad(a, b, c, d):
        tris.append((a, b, c))
        tris.append((a, c, d))

    ny, nx = Z.shape
    cell = inside[:-1, :-1] & inside[1:, :-1] & inside[:-1, 1:] & inside[1:, 1:]
    for i in range(ny - 1):
        for j in range(nx - 1):
            if not cell[i, j]:
                continue
            p00 = (X[i, j], Y[i, j], Z[i, j])
            p10 = (X[i, j + 1], Y[i, j + 1], Z[i, j + 1])
            p11 = (X[i + 1, j + 1], Y[i + 1, j + 1], Z[i + 1, j + 1])
            p01 = (X[i + 1, j], Y[i + 1, j], Z[i + 1, j])
            quad(p00, p10, p11, p01)                                   # top
            b00 = (X[i, j], Y[i, j], Zb[i, j])
            b10 = (X[i, j + 1], Y[i, j + 1], Zb[i, j + 1])
            b11 = (X[i + 1, j + 1], Y[i + 1, j + 1], Zb[i + 1, j + 1])
            b01 = (X[i + 1, j], Y[i + 1, j], Zb[i + 1, j])
            quad(b01, b11, b10, b00)                                   # bottom
            # a wall wherever the neighbouring cell is outside the outline
            if i == 0 or not cell[i - 1, j]:
                quad(b00, b10, p10, p00)
            if i == ny - 2 or not cell[i + 1, j]:
                quad(b11, b01, p01, p11)
            if j == 0 or not cell[i, j - 1]:
                quad(b01, b00, p00, p01)
            if j == nx - 2 or not cell[i, j + 1]:
                quad(b10, b11, p11, p10)

    with open(path, 'wb') as f:
        f.write(b'\0' * 80)
        f.write(struct.pack('<I', len(tris)))
        for a, b, c in tris:
            n = np.cross(np.subtract(b, a), np.subtract(c, a))
            ln = np.linalg.norm(n)
            n = n / ln if ln else np.zeros(3)
            f.write(struct.pack('<12fH', *n, *a, *b, *c, 0))
    return len(tris)


def main():
    ap = argparse.ArgumentParser(description='Fit and render an insole surface.')
    ap.add_argument('--participant', type=int, default=83)
    ap.add_argument('--footwear', default='BF')
    ap.add_argument('--speed', default='W1')
    ap.add_argument('--step', type=int, default=5)
    ap.add_argument('--modulus', type=float, default=2.0)
    ap.add_argument('--thickness', type=float, default=10.0)
    ap.add_argument('--max-carve', type=float, default=2.0)
    ap.add_argument('--smooth', type=float, default=1.2,
                    help='Gaussian blur sigma in pixels applied to the carve map')
    ap.add_argument('--pad', type=int, default=2, help='dilations that grow the insole outline')
    ap.add_argument('--degree', type=int, default=5, help='polynomial degree')
    ap.add_argument('--gaussians', type=int, default=3)
    ap.add_argument('--figure', default='insole_3d.png')
    ap.add_argument('--hero', default=None, help='one large 3D render of the insole')
    ap.add_argument('--gif', default=None, help='rotating turntable animation')
    ap.add_argument('--gif-frames', type=int, default=180)
    ap.add_argument('--gif-fps', type=int, default=12,
                    help='frames per second; frames/fps = seconds per revolution')
    ap.add_argument('--exaggerate', type=float, default=5.0,
                    help='z-axis stretch for the 3D views; 0 = true proportions')
    ap.add_argument('--profile', choices=('cushion', 'split', 'shell', 'carve'),
                    default='cushion',
                    help="'cushion' = thickest under the hot spots, top bulges up and "
                         "bottom bulges down (softer where the load is high); "
                         "'split' = the same mirroring but thinnest under the hot spots; "
                         "'shell' = constant-thickness formed shell; 'carve' = flat "
                         "blank cut on the top face only")
    ap.add_argument('--shell', type=float, default=6.0,
                    help='shell wall thickness in mm, used by --profile shell')
    ap.add_argument('--stl', default=None)
    args = ap.parse_args()

    from utils import load_footsteps, load_metadata
    steps = load_footsteps(args.participant, args.footwear, args.speed, pipeline=1)
    md = load_metadata(args.participant, args.footwear, args.speed)
    rows = md.index[md.Exclude == 0]
    f = steps[rows[args.step]]
    p = f[f.sum((1, 2)).argmax()]
    p = np.where(p > CONTACT_KPA, p, 0.0)

    k0 = args.modulus * 1000.0 / args.thickness
    carve = np.clip(p / k0, 0, args.max_carve)
    carve = gaussian_filter(carve, args.smooth)
    outline = insole_outline(p > CONTACT_KPA, args.pad)
    carve = np.where(outline, carve, 0.0)

    # crop to the outline so the fit is not dominated by empty space
    ys, xs = np.where(outline)
    y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max()
    carve = carve[y0:y1 + 1, x0:x1 + 1]
    outline = outline[y0:y1 + 1, x0:x1 + 1]
    pressure = p[y0:y1 + 1, x0:x1 + 1]
    ny, nx = carve.shape

    # v runs heel(-1) -> toes(+1); the pipeline stores toes at row 0, so flip
    vv, uu = np.mgrid[0:ny, 0:nx]
    U = 2 * uu / (nx - 1) - 1
    V = 1 - 2 * vv / (ny - 1)
    X = uu * PX_MM
    Y = (ny - 1 - vv) * PX_MM

    def build(cmap_):
        """Top, mid and bottom surfaces for a carve map, per --profile.

        cushion  material is ADDED where the pressure is high, so the pad is
                 thickest under the hot spots: the top face bulges up and the
                 bottom face bulges down by the same amount. Because the local
                 stiffness of a foam layer is k = E / h, a thicker patch is a
                 SOFTER patch - this is equation (4), k proportional to 1/p,
                 built as geometry instead of as a change of material.
        carve    a flat blank with the contour cut into the top face only
        shell    a formed shell of constant thickness: the contour becomes the
                 MID-surface and both faces carry the same relief
        split    thickness h0 - c, mirrored about a flat mid-plane; thinnest
                 under the hot spots
        """
        if args.profile == 'cushion':
            t = args.thickness + cmap_
            top = t / 2
            bot = -t / 2
        elif args.profile == 'carve':
            top = args.thickness - cmap_
            bot = np.zeros_like(top)
        elif args.profile == 'shell':
            # centre the contour on its own mean so the mid-surface sits level
            mid = args.thickness / 2 - (cmap_ - cmap_[outline].mean())
            top = mid + args.shell / 2
            bot = mid - args.shell / 2
        else:                                       # split
            # two mirrored faces about a flat mid-plane at z = 0, separated by
            # exactly the surface height z = h0 - c, so top = -bot everywhere
            # and (top - bot) == z(x,y)
            z = args.thickness - cmap_
            top = z / 2
            bot = -z / 2
        return top, bot, (top + bot) / 2

    Ztop, Zbot, Zmid = build(carve)
    Z = Ztop

    m = outline
    u, v, c = U[m], V[m], carve[m]

    coef, names, pred_p = fit_poly(u, v, c, args.degree)
    r2_poly = r2(c, pred_p)
    popt, pred_g = fit_gaussians(u, v, c, args.gaussians)
    r2_gauss = r2(c, pred_g)

    print(f'insole for P{args.participant:03d} {args.footwear} {args.speed} '
          f'step {int(md.FootstepID[rows[args.step]])} ({md.Side[rows[args.step]]})')
    print(f'grid {nx} x {ny} px = {nx * PX_MM:.0f} x {ny * PX_MM:.0f} mm, '
          f'{m.sum()} points inside the outline')
    print(f'foam E = {args.modulus} MPa, h0 = {args.thickness} mm -> k0 = {k0:.0f} kPa/mm')
    print(f'carve depth: max {carve.max():.2f} mm, mean {c[c > 0].mean():.2f} mm\n')

    print(f'polynomial degree {args.degree}: {len(coef)} terms, R2 = {r2_poly:.4f}')
    order = np.argsort(-np.abs(coef))[:8]
    print('  largest terms:  ' + '  '.join(
        f'{coef[i]:+.3f} u^{names[i][0]} v^{names[i][1]}' for i in order))

    print(f'\n{args.gaussians} Gaussians: {len(popt)} parameters, R2 = {r2_gauss:.4f}')
    print(f'  {"k":>2} {"A (mm)":>8} {"u0":>7} {"v0":>7} {"su":>6} {"sv":>6}   position')
    for k in range(args.gaussians):
        A, u0, v0, su, sv = popt[5 * k:5 * k + 5]
        where = ('toes' if v0 > 0.55 else 'forefoot' if v0 > 0.05
                 else 'midfoot' if v0 > -0.45 else 'heel')
        # +u is the right-hand side of the image. Measured over 240 footsteps, the
        # hallux sits at u = +0.61 on left feet and u = -0.60 on right feet, so
        # +u is medial on a LEFT foot and lateral on a RIGHT foot.
        foot_is_left = str(md.Side[rows[args.step]]).lower().startswith('l')
        medial_is_positive = foot_is_left
        if abs(u0) < 0.2:
            side = 'centred'
        else:
            side = 'medial' if (u0 > 0) == medial_is_positive else 'lateral'
        print(f'  {k:>2} {A:>8.3f} {u0:>7.3f} {v0:>7.3f} {su:>6.3f} {sv:>6.3f}   {where} ({side})')

    # ---------------- figure ----------------
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    Zg = np.where(outline, args.thickness - gauss_sum((U, V), *popt), np.nan)

    # the fitted model, written out so the figure carries its own equation
    K = args.gaussians
    equation = (r'$c(u,v)\;=\;\sum_{k=1}^{' + str(K) + r'} A_k\,'
                r'\exp\left[-\frac{(u-u_k)^2}{2\sigma_{u,k}^2}'
                r'-\frac{(v-v_k)^2}{2\sigma_{v,k}^2}\right]$')

    def term_lines():
        out = [f'{"k":>2}  {"A (mm)":>7}  {"u_k":>6}  {"v_k":>6}  {"su":>5}  {"sv":>5}   part']
        for k in range(K):
            A, u0, v0, su, sv = popt[5 * k:5 * k + 5]
            part = ('toe pad' if v0 > 0.55 else 'met pad' if v0 > 0.05
                    else 'arch' if v0 > -0.45 else 'heel cup')
            out.append(f'{k + 1:>2}  {A:>7.3f}  {u0:>+6.2f}  {v0:>+6.2f}  '
                       f'{su:>5.2f}  {sv:>5.2f}   {part}')
        return '\n'.join(out)

    caption = (f'{args.profile} profile'
               + (f', wall {args.shell:.0f} mm' if args.profile == 'shell' else '')
               + f'      R2 = {r2_gauss:.3f}')

    carve_g = np.where(outline, gauss_sum((U, V), *popt), 0.0)
    Gtop, Gbot, Gmid = build(carve_g)
    nan = lambda a: np.where(outline, a, np.nan)
    Zm, Zb = nan(Ztop), nan(Zbot)
    Zg, Zgb = nan(Gtop), nan(Gbot)

    zlo = min(np.nanmin(Zb), np.nanmin(Zgb)) - 0.3
    zhi = max(np.nanmax(Zm), np.nanmax(Zg)) + 0.3

    # presentation figure: barefoot pressure -> carve depth -> insole surface
    fig = plt.figure(figsize=(15, 6.6))
    jet = matplotlib.colormaps['jet'](np.linspace(0, 1, 256))
    jet[0] = [0, 0, 0, 1]
    jet_cmap = matplotlib.colors.ListedColormap(jet)

    ax = fig.add_axes((0.03, 0.06, 0.24, 0.74))
    im = ax.imshow(pressure, cmap=jet_cmap, origin='upper')
    ax.set_title('barefoot pressure   $p(x,y)$', fontsize=12, pad=8)
    ax.axis('off')
    fig.colorbar(im, ax=ax, shrink=.85, label='kPa')

    ax = fig.add_axes((0.36, 0.06, 0.24, 0.74))
    im = ax.imshow(np.where(outline, carve, np.nan), cmap='magma', origin='upper')
    ax.set_title('carve depth   $c = p/k_0$', fontsize=12, pad=8)
    ax.axis('off')
    fig.colorbar(im, ax=ax, shrink=.85, label='mm')

    ax = fig.add_axes((0.63, 0.00, 0.36, 0.82), projection='3d')
    ax.plot_surface(X, Y, Zm, cmap='viridis', linewidth=0.15, edgecolor='k',
                    antialiased=True, rcount=ny, ccount=nx,
                    vmin=np.nanmin(Zm), vmax=np.nanmax(Zm))
    ax.plot_surface(X, Y, Zb, cmap='autumn', linewidth=0.15, edgecolor='k',
                    antialiased=True, rcount=ny, ccount=nx,
                    vmin=np.nanmin(Zb), vmax=np.nanmax(Zb))
    ax.set_zlim(zlo, zhi)
    ax.set_box_aspect((nx, ny, args.exaggerate or ny * (zhi - zlo) / (ny * PX_MM)))
    ax.view_init(elev=30, azim=-62)
    ax.set_xlabel('x (mm)', fontsize=9, labelpad=-2)
    ax.set_ylabel('y (mm)', fontsize=9, labelpad=2)
    ax.set_zlabel('z (mm)', fontsize=9, labelpad=-3)
    ax.tick_params(labelsize=7, pad=0)
    ax.zaxis.set_major_locator(plt.MaxNLocator(4))
    ax.set_title('insole surface, from above', fontsize=12, pad=6)

    fig.suptitle(f'P{args.participant:03d} {args.footwear} {args.speed}   '
                 f'{args.profile} profile, E={args.modulus} MPa, h0={args.thickness} mm'
                 + ('   (z axis exaggerated)' if args.exaggerate else ''), fontsize=13)
    fig.savefig(args.figure, dpi=120)
    print(f'\nfigure: {pathlib.Path(args.figure).resolve()}')

    # ---------------- one large render ----------------
    def draw(ax, elev, azim, faces=None):
        up, dn = faces if faces else (Zm, Zb)
        ax.plot_surface(X, Y, up, cmap='viridis', linewidth=0.2, edgecolor='#00000030',
                        antialiased=True, rcount=ny, ccount=nx,
                        vmin=np.nanmin(up), vmax=np.nanmax(up), shade=True)
        ax.plot_surface(X, Y, dn, cmap='autumn', linewidth=0.2, edgecolor='#00000030',
                        antialiased=True, rcount=ny, ccount=nx,
                        vmin=np.nanmin(dn), vmax=np.nanmax(dn), shade=True)
        ax.set_zlim(zlo, zhi)
        # exaggerate=0 -> real proportions: z spans (zhi - zlo) mm against
        # a footprint of nx*PX_MM by ny*PX_MM mm
        ax.set_box_aspect((nx, ny, args.exaggerate or ny * (zhi - zlo) / (ny * PX_MM)))
        ax.view_init(elev=elev, azim=azim)
        ax.set_xlabel('x (mm)', fontsize=9)
        ax.set_ylabel('y (mm)', fontsize=9)
        ax.set_zlabel('z (mm)', fontsize=9)
        ax.tick_params(labelsize=7)

    if args.hero:
        # the two faces are mirror images, so one viewpoint can only ever show
        # one of them: render the pair side by side
        f2 = plt.figure(figsize=(14, 8.5))
        for col, (elev, azim, lab) in enumerate(
                [(32, -58, 'top face'), (-30, -58, 'bottom face')]):
            a2 = f2.add_axes((0.02 + 0.49 * col, 0.01, 0.48, 0.70), projection='3d')
            draw(a2, elev, azim)
            a2.set_title(lab, fontsize=11, pad=2)
        f2.text(0.5, 0.965, f'Insole surface   P{args.participant:03d} {args.footwear} '
                            f'{args.speed}   deepest carve {carve.max():.2f} mm'
                            + ('   (z axis exaggerated)' if args.exaggerate else ''),
                ha='center', fontsize=13)
        f2.text(0.5, 0.900, equation, ha='center', fontsize=15)
        f2.text(0.5, 0.858, caption, ha='center', fontsize=10.5, color='0.3')
        f2.text(0.5, 0.842, term_lines(), ha='center', va='top', fontsize=9,
                family='monospace',
                bbox=dict(boxstyle='round,pad=0.5', fc='#f4f4f4', ec='0.7'))
        f2.savefig(args.hero, dpi=130)
        print(f'hero:   {pathlib.Path(args.hero).resolve()}')

    if args.gif:
        from matplotlib.animation import FuncAnimation, PillowWriter
        f3 = plt.figure(figsize=(8, 7.6))
        a3 = f3.add_axes((-0.05, -0.06, 1.10, 0.82), projection='3d')
        f3.text(0.5, 0.955, f'P{args.participant:03d} insole - deepest carve '
                            f'{carve.max():.2f} mm', ha='center', fontsize=12)
        f3.text(0.5, 0.875, equation, ha='center', fontsize=14)
        f3.text(0.5, 0.815, term_lines(), ha='center', va='top', fontsize=8,
                family='monospace',
                bbox=dict(boxstyle='round,pad=0.4', fc='#f4f4f4', ec='0.7'))

        def frame(i):
            a3.clear()
            draw(a3, 14 + 26 * np.sin(2 * np.pi * i / args.gif_frames),
                 -180 + 360 * i / args.gif_frames)
            return []

        anim = FuncAnimation(f3, frame, frames=args.gif_frames, blit=False)
        anim.save(args.gif, writer=PillowWriter(fps=args.gif_fps), dpi=85)
        print(f'gif:    {pathlib.Path(args.gif).resolve()}  '
              f'({args.gif_frames} frames @ {args.gif_fps} fps = '
              f'{args.gif_frames / args.gif_fps:.1f} s per revolution)')

    if args.stl:
        n = write_stl(args.stl, X, Y, Ztop, Zbot, outline)
        print(f'STL:    {pathlib.Path(args.stl).resolve()}  ({n} triangles)')


if __name__ == '__main__':
    main()
