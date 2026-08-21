#!/usr/bin/env python3
"""Select anatomically complete footstep images produced by extract_foot_images.py.

A barefoot print is "complete" when the whole foot landed on the sensor: heel,
midfoot, forefoot and toes all present, at the size that participant normally
prints. Absolute pixel thresholds would punish small-footed participants, so
every measurement is compared against that participant's own median for that
foot side.

Measured per image (on the de-noised contact mask):
  h_r      bounding-box height / participant median   - truncated prints
  area_r   contact area / participant median          - faint or partial contact
  heel_r   heel-region area / participant median      - forefoot-only prints
  toe_r    toe-region area / participant median       - toes off the mat
  border   contact touching the image edge            - clipped print
  Rscore   registration error from the dataset metadata (lower is better)

Example
-------
  python select_good_images.py
  python select_good_images.py --min-h-ratio 0.95 --max-rscore 0.8
  python select_good_images.py --dry-run
"""

import argparse
import pathlib
import shutil
import sys

import numpy as np
import pandas as pd
from PIL import Image
from scipy import ndimage

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / 'FRDR_dataset' / 'example_code' / 'python'))

# region splits as a fraction of the print's bounding-box height (0 = toes, 1 = heel)
TOE_END, FORE_START, FORE_END, HEEL_START = 0.20, 0.20, 0.55, 0.75


def contact_mask(path, upscale, min_component):
    """Non-black pixels of the peak-pressure PNG, at native resolution, de-noised."""
    rgb = np.asarray(Image.open(path).convert('RGB'))
    if upscale > 1:
        rgb = rgb[::upscale, ::upscale]
    mask = rgb.astype(np.int32).sum(2) > 0
    if min_component > 1 and mask.any():
        lab, n = ndimage.label(mask, structure=np.ones((3, 3), int))
        if n:
            sizes = np.bincount(lab.ravel())
            sizes[0] = 0
            mask = np.isin(lab, np.flatnonzero(sizes >= min_component))
    return mask


def measure(mask):
    """Bounding box, contact area and per-region areas of one print."""
    rows = np.flatnonzero(mask.any(1))
    cols = np.flatnonzero(mask.any(0))
    if len(rows) == 0:
        return None
    r0, r1, c0, c1 = rows[0], rows[-1], cols[0], cols[-1]
    h = int(r1 - r0 + 1)
    box = mask[r0:r1 + 1, c0:c1 + 1]
    return dict(
        h=h, w=int(c1 - c0 + 1), area=int(mask.sum()),
        toe=int(box[:max(1, int(TOE_END * h))].sum()),
        fore=int(box[int(FORE_START * h):int(FORE_END * h)].sum()),
        heel=int(box[int(HEEL_START * h):].sum()),
        border=bool(mask[0].any() or mask[-1].any() or mask[:, 0].any() or mask[:, -1].any()),
    )


def load_rscore(index):
    """Join Rscore from the dataset metadata; returns None if the dataset is absent."""
    try:
        from utils import dataset_folder, load_metadata
    except ImportError:
        return None
    if not dataset_folder.is_dir():
        return None
    keys = index[['participant', 'footwear', 'speed']].drop_duplicates()
    frames = []
    for pid, fw, sp in keys.itertuples(index=False):
        md = load_metadata(int(pid), fw, sp)
        frames.append(md[['ParticipantID', 'Footwear', 'Speed', 'FootstepID', 'Rscore']])
    md = pd.concat(frames, ignore_index=True)
    return md.rename(columns={'ParticipantID': 'participant', 'Footwear': 'footwear',
                              'Speed': 'speed', 'FootstepID': 'footstep_id'})


def main():
    ap = argparse.ArgumentParser(description='Copy anatomically complete footstep images '
                                             'into a separate folder.')
    ap.add_argument('--src', default='Images', help='folder written by extract_foot_images.py')
    ap.add_argument('--out', default='good_images', help='destination folder (default: good_images)')
    ap.add_argument('--upscale', type=int, default=4,
                    help='the --upscale used when the source images were written (default: 4)')
    ap.add_argument('--min-component', type=int, default=3,
                    help='drop contact blobs smaller than this many pixels as sensor noise')
    ap.add_argument('--min-h-ratio', type=float, default=0.90)
    ap.add_argument('--min-area-ratio', type=float, default=0.80)
    ap.add_argument('--min-heel-ratio', type=float, default=0.70)
    ap.add_argument('--min-toe-ratio', type=float, default=0.50)
    ap.add_argument('--max-rscore', type=float, default=0.0,
                    help='registration-error ceiling; off by default because a high Rscore '
                         'usually means an atypically shaped foot, not an incomplete print')
    ap.add_argument('--allow-border', action='store_true',
                    help='keep prints that touch the image edge')
    ap.add_argument('--clean', action='store_true',
                    help='delete PNGs already in the destination first, so a re-run with '
                         'different thresholds does not leave stale files behind')
    ap.add_argument('--dry-run', action='store_true', help='report the funnel, copy nothing')
    args = ap.parse_args()

    src = pathlib.Path(args.src)
    manifest = src / 'index.csv'
    if not manifest.exists():
        sys.exit(f'{manifest} not found - run extract_foot_images.py first')

    index = pd.read_csv(manifest)
    if 'mode' in index and (index['mode'] != 'peak').any():
        index = index[index['mode'] == 'peak']
        print('note: only peak-mode images can be scored; frames-mode rows ignored')
    print(f'scoring {len(index)} images from {src}/ ...', flush=True)

    recs = []
    for i, fn in enumerate(index.filename):
        m = measure(contact_mask(src / fn, args.upscale, args.min_component))
        if m is None:
            m = dict(h=0, w=0, area=0, toe=0, fore=0, heel=0, border=False)
        m['filename'] = fn
        recs.append(m)
        if i and i % 10000 == 0:
            print(f'  {i}/{len(index)}', flush=True)

    d = index.merge(pd.DataFrame(recs), on='filename')

    rs = load_rscore(index)
    if rs is None:
        print('note: dataset metadata unavailable - skipping the Rscore check')
        d['Rscore'] = np.nan
    else:
        d = d.merge(rs, on=['participant', 'footwear', 'speed', 'footstep_id'], how='left')

    # every measurement is judged against the participant's own median for that side
    g = d.groupby(['participant', 'footwear', 'side'])
    for c in ['h', 'area', 'heel', 'toe']:
        med = g[c].transform('median')
        d[c + '_r'] = np.where(med > 0, d[c] / med.replace(0, np.nan), 0.0)

    checks = {
        'empty': d.area == 0,
        'short_print': d.h_r < args.min_h_ratio,
        'small_area': d.area_r < args.min_area_ratio,
        'missing_heel': d.heel_r < args.min_heel_ratio,
        'missing_toes': d.toe_r < args.min_toe_ratio,
    }
    if not args.allow_border:
        checks['clipped_at_edge'] = d.border
    if args.max_rscore > 0:
        checks['high_rscore'] = d.Rscore.notna() & (d.Rscore > args.max_rscore)

    fail = pd.DataFrame(checks)
    d['reasons'] = [';'.join(fail.columns[r]) for r in fail.to_numpy()]
    keep = d[d.reasons == '']
    drop = d[d.reasons != '']

    print(f'\ntotal: {len(d)}')
    for name, col in checks.items():
        print(f'  fails {name:16} {int(col.sum()):6}')
    print(f'\nkept: {len(keep)} ({100 * len(keep) / len(d):.1f}%)   rejected: {len(drop)}')
    if args.dry_run:
        return

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    existing = list(out.rglob('*.png'))
    if existing:
        if args.clean:
            for p in existing:
                p.unlink()
            for p in sorted((q for q in out.rglob('*') if q.is_dir()), reverse=True):
                if not any(p.iterdir()):
                    p.rmdir()
            print(f'cleaned {len(existing)} images out of {out}/')
        else:
            stale = {p.relative_to(out).as_posix() for p in existing} - set(keep.filename)
            if stale:
                print(f'warning: {len(stale)} images already in {out}/ are not in this '
                      f'selection and were left in place - re-run with --clean to remove them')

    cols = [c for c in d.columns if c != 'reasons']
    for n, fn in enumerate(keep.filename, 1):
        dst = out / fn
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src / fn, dst)
        if n % 10000 == 0:
            print(f'  copied {n}/{len(keep)}', flush=True)

    keep[cols].to_csv(out / 'index.csv', index=False)
    drop[cols + ['reasons']].to_csv(out / 'rejected.csv', index=False)
    print(f'\ncopied {len(keep)} images to {out.resolve()}')
    print(f'manifest: {(out / "index.csv").resolve()}')
    print(f'rejects:  {(out / "rejected.csv").resolve()}')


if __name__ == '__main__':
    main()
