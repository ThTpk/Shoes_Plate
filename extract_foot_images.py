#!/usr/bin/env python3
"""Extract per-footstep plantar pressure images from the StepUP-P150 dataset.

Reads the preprocessed footstep tensors (pipeline_1.npz / pipeline_2.npz) and
writes one PNG per footstep into an output folder, plus an index.csv manifest.

Examples
--------
  # one trial, ~74 images
  python extract_foot_images.py --participants 56 --footwear P1 --speed W1

  # first 10 participants, barefoot, all walking speeds, upscaled 4x
  python extract_foot_images.py --participants 1-10 --footwear BF --upscale 4

  # count what a full run would produce, without writing anything
  python extract_foot_images.py --dry-run
"""

import argparse
import csv
import pathlib
import sys
import time

import numpy as np
import matplotlib as mpl
from PIL import Image

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / 'FRDR_dataset' / 'example_code' / 'python'))
from utils import dataset_folder, load_metadata, load_footsteps  # noqa: E402

FOOTWEAR = ['BF', 'ST', 'P1', 'P2']
SPEEDS = ['W1', 'W2', 'W3', 'W4']
CONFIRM_THRESHOLD = 5000


def parse_ids(spec, lo=1, hi=150):
    """'all' | '56' | '1-10' | '1,5,9-12' -> sorted list of ints."""
    if spec.strip().lower() == 'all':
        return list(range(lo, hi + 1))
    out = set()
    for part in spec.split(','):
        part = part.strip()
        if not part:
            continue
        if '-' in part:
            a, b = part.split('-', 1)
            out.update(range(int(a), int(b) + 1))
        else:
            out.add(int(part))
    bad = [i for i in out if not lo <= i <= hi]
    if bad:
        raise argparse.ArgumentTypeError(f'participant id out of range {lo}-{hi}: {sorted(bad)}')
    return sorted(out)


def parse_choices(spec, valid, label):
    if spec.strip().lower() == 'all':
        return list(valid)
    picked = [p.strip().upper() for p in spec.split(',') if p.strip()]
    bad = [p for p in picked if p not in valid]
    if bad:
        raise argparse.ArgumentTypeError(f'unknown {label}: {bad} (valid: {valid})')
    return picked


def build_lut(name):
    """256-entry RGB lookup table; index 0 forced to black for the jet map."""
    colors = mpl.colormaps[name](np.linspace(0, 1, 256))
    if name == 'jet':
        colors[0] = [0, 0, 0, 1]
    return (colors[:, :3] * 255).astype(np.uint8)


def render(img, vmax, lut, upscale, gray):
    """float pressure image -> uint8 PNG-ready array, scaled to [0, vmax]."""
    vmax = float(vmax)
    if not np.isfinite(vmax) or vmax <= 0:
        vmax = 1.0
    idx = np.clip(img / vmax, 0, 1)
    idx = (idx * 255).astype(np.uint8)
    out = idx if gray else lut[idx]
    if upscale > 1:
        out = np.repeat(np.repeat(out, upscale, axis=0), upscale, axis=1)
    return out


def trial_list(participants, footwear, speeds):
    """(pid, fw, sp) triples that actually exist on disk."""
    trials, missing = [], 0
    for pid in participants:
        for fw in footwear:
            for sp in speeds:
                if (dataset_folder / f'{pid:03}' / fw / sp).is_dir():
                    trials.append((pid, fw, sp))
                else:
                    missing += 1
    return trials, missing


def main():
    ap = argparse.ArgumentParser(
        description='Extract per-footstep pressure images from StepUP-P150.')
    ap.add_argument('--participants', default='all',
                    help="'all', '56', '1-10', '1,5,9-12' (default: all)")
    ap.add_argument('--footwear', default='all', help='BF,ST,P1,P2 or all (default: all)')
    ap.add_argument('--speed', default='all', help='W1,W2,W3,W4 or all (default: all)')
    ap.add_argument('--pipeline', type=int, choices=(1, 2), default=1,
                    help='1 = original kPa units, 2 = resized + amplitude-normalized (default: 1)')
    ap.add_argument('--mode', choices=('peak', 'frames'), default='peak',
                    help="'peak' = one image per footstep; 'frames' = every frame (default: peak)")
    ap.add_argument('--frame-step', type=int, default=1,
                    help='keep every Nth frame in frames mode (default: 1)')
    ap.add_argument('--out', default='Images', help='output folder (default: Images)')
    ap.add_argument('--layout', choices=('flat', 'nested'), default='flat',
                    help='flat = all PNGs in one folder; nested = one subfolder per participant')
    ap.add_argument('--cmap', default='jet',
                    help="matplotlib colormap, or 'gray' for 8-bit grayscale (default: jet)")
    ap.add_argument('--scale', choices=('image', 'trial', 'fixed'), default='image',
                    help="pressure-to-colour scaling reference: 'image' matches the notebook's "
                         "per-image contrast, 'trial'/'fixed' keep brightness comparable "
                         '(default: image)')
    ap.add_argument('--vmax', type=float, default=None, help='upper pressure bound for --scale fixed')
    ap.add_argument('--upscale', type=int, default=1,
                    help='nearest-neighbour pixel magnification (default: 1)')
    ap.add_argument('--include-excluded', action='store_true',
                    help='also export footsteps flagged Exclude (outliers / partial steps)')
    ap.add_argument('--limit-per-trial', type=int, default=None,
                    help='cap footsteps taken from each trial')
    ap.add_argument('--overwrite', action='store_true', help='rewrite images that already exist')
    ap.add_argument('--dry-run', action='store_true', help='report the plan, write nothing')
    ap.add_argument('--yes', action='store_true', help='skip the confirmation prompt for large runs')
    args = ap.parse_args()

    if args.scale == 'fixed' and args.vmax is None:
        ap.error('--scale fixed requires --vmax (e.g. --vmax 1000 for pipeline 1)')
    if args.upscale < 1:
        ap.error('--upscale must be >= 1')
    if args.frame_step < 1:
        ap.error('--frame-step must be >= 1')

    if not dataset_folder.is_dir():
        sys.exit(f'dataset folder not found: {dataset_folder}\n'
                 'set STEPUP_DATA to the folder that contains py/')

    try:
        participants = parse_ids(args.participants)
        footwear = parse_choices(args.footwear, FOOTWEAR, 'footwear')
        speeds = parse_choices(args.speed, SPEEDS, 'speed')
    except (argparse.ArgumentTypeError, ValueError) as e:
        ap.error(str(e))
    gray = args.cmap == 'gray'
    lut = None if gray else build_lut(args.cmap)

    trials, missing = trial_list(participants, footwear, speeds)
    if not trials:
        sys.exit('no matching trials found on disk')

    # planning pass: metadata only (cheap) so the size of the job is known up front
    print(f'scanning {len(trials)} trials ...', flush=True)
    plan, n_steps = [], 0
    for pid, fw, sp in trials:
        md = load_metadata(pid, fw, sp)
        keep = md if args.include_excluded else md[md['Exclude'] == 0]
        if args.limit_per_trial:
            keep = keep.head(args.limit_per_trial)
        plan.append((pid, fw, sp, list(keep.index)))
        n_steps += len(keep)

    per_step = 1 if args.mode == 'peak' else len(range(0, 101, args.frame_step))
    n_images = n_steps * per_step
    print(f'trials: {len(trials)} found, {missing} missing | '
          f'footsteps: {n_steps} | images: {n_images}')
    if args.dry_run:
        return

    if n_images > CONFIRM_THRESHOLD and not args.yes:
        if args.layout == 'flat':
            print(f'note: {n_images} files in one folder - consider --layout nested')
        if input(f'write {n_images} images to {args.out}/? [y/N] ').strip().lower() not in ('y', 'yes'):
            sys.exit('aborted')

    out_root = pathlib.Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    manifest = out_root / 'index.csv'
    new_manifest = args.overwrite or not manifest.exists()

    written = skipped = 0
    t0 = time.time()
    with open(manifest, 'w' if new_manifest else 'a', newline='', encoding='utf-8') as fh:
        w = csv.writer(fh)
        if new_manifest:
            w.writerow(['filename', 'participant', 'footwear', 'speed', 'footstep_id', 'pass_id',
                        'side', 'exclude', 'foot_length_px', 'foot_width_px', 'peak_value',
                        'vmax_used', 'pipeline', 'mode'])

        for n, (pid, fw, sp, rows) in enumerate(plan, 1):
            if not rows:
                continue
            md = load_metadata(pid, fw, sp)
            steps = load_footsteps(pid, fw, sp, pipeline=args.pipeline)
            trial_max = float(steps[rows].max()) if args.scale == 'trial' else None

            d = out_root / f'P{pid:03d}' if args.layout == 'nested' else out_root
            d.mkdir(parents=True, exist_ok=True)

            for r in rows:
                m = md.iloc[r]
                side = str(m['Side'])[0].upper()
                stem = f"P{pid:03d}_{fw}_{sp}_step{int(m['FootstepID']):03d}_{side}"
                peak = steps[r].max(0)
                peak_val = float(peak.max())

                if args.scale == 'image':
                    vmax = peak_val
                elif args.scale == 'trial':
                    vmax = trial_max
                else:
                    vmax = args.vmax

                if args.mode == 'peak':
                    targets = [(d / f'{stem}.png', peak)]
                else:
                    sub = d / stem
                    sub.mkdir(exist_ok=True)
                    targets = [(sub / f'frame{f:03d}.png', steps[r][f])
                               for f in range(0, steps.shape[1], args.frame_step)]

                for path, img in targets:
                    if path.exists() and not args.overwrite:
                        skipped += 1
                        continue
                    arr = render(img, vmax, lut, args.upscale, gray)
                    Image.fromarray(arr, mode='L' if gray else 'RGB').save(path, optimize=True)
                    written += 1
                    w.writerow([path.relative_to(out_root).as_posix(), pid, fw, sp,
                                int(m['FootstepID']), int(m['PassID']), side, int(m['Exclude']),
                                m['FootLength'], m['FootWidth'], round(peak_val, 3),
                                round(float(vmax), 6), args.pipeline, args.mode])

            print(f'[{n}/{len(plan)}] P{pid:03d} {fw} {sp}: {len(rows)} footsteps '
                  f'-> {written} written, {skipped} skipped', flush=True)

    dt = max(time.time() - t0, 1e-6)
    print(f'\ndone: {written} images written, {skipped} skipped, '
          f'{dt:.1f}s ({written / dt:.0f} img/s)')
    print(f'output:   {out_root.resolve()}')
    print(f'manifest: {manifest.resolve()}')


if __name__ == '__main__':
    main()
