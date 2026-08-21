#!/usr/bin/env python3
"""Render the insole presentation figure and turntable GIF for a random sample
of the screened barefoot footsteps.

Each sampled row of good_images/index.csv identifies a footstep by
(participant, footwear, speed, footstep_id). insole_3d.py addresses footsteps by
their position among the trial's usable steps, so the id is translated here and
insole_3d.py is then run unchanged - that keeps every output identical to the
single-footstep command rather than re-implementing the rendering.
"""

import argparse
import pathlib
import subprocess
import sys
import time

import pandas as pd

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / 'FRDR_dataset' / 'example_code' / 'python'))
from utils import load_metadata  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description='Batch-render insole figures and GIFs.')
    ap.add_argument('--index', default='good_images/index.csv')
    ap.add_argument('--out', default='shoeplate_image')
    ap.add_argument('--count', type=int, default=100)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--exaggerate', type=float, default=18.0)
    ap.add_argument('--profile', default='cushion')
    ap.add_argument('--gif-frames', type=int, default=180)
    ap.add_argument('--no-gif', action='store_true', help='figures only')
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()

    idx = pd.read_csv(args.index)
    sample = idx.sample(n=min(args.count, len(idx)), random_state=args.seed)
    sample = sample.sort_values(['participant', 'speed', 'footstep_id'])
    print(f'sampled {len(sample)} of {len(idx)} footsteps (seed {args.seed})')

    # footstep_id -> position among the trial's usable steps, one metadata read per trial
    order = {}
    for (pid, fw, sp), _ in sample.groupby(['participant', 'footwear', 'speed']):
        md = load_metadata(int(pid), fw, sp)
        keep = md.index[md.Exclude == 0]
        order[(pid, fw, sp)] = {int(md.FootstepID[i]): n for n, i in enumerate(keep)}

    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    jobs = []
    for r in sample.itertuples(index=False):
        step = order[(r.participant, r.footwear, r.speed)].get(int(r.footstep_id))
        if step is None:
            print(f'  skip P{r.participant:03d} {r.speed} step {r.footstep_id}: not in trial')
            continue
        stem = f'P{r.participant:03d}_{r.footwear}_{r.speed}_step{int(r.footstep_id):03d}'
        jobs.append((r, step, stem))

    print(f'{len(jobs)} to render -> {out.resolve()}')
    if args.dry_run:
        return

    t0 = time.time()
    done = skipped = failed = 0
    for n, (r, step, stem) in enumerate(jobs, 1):
        fig = out / f'{stem}.png'
        gif = out / f'{stem}.gif'
        if fig.exists() and (args.no_gif or gif.exists()):
            skipped += 1
            continue
        cmd = [sys.executable, str(HERE / 'insole_3d.py'),
               '--participant', str(int(r.participant)),
               '--footwear', str(r.footwear), '--speed', str(r.speed),
               '--step', str(step), '--profile', args.profile,
               '--exaggerate', str(args.exaggerate),
               '--figure', str(fig)]
        if not args.no_gif:
            cmd += ['--gif', str(gif), '--gif-frames', str(args.gif_frames)]
        res = subprocess.run(cmd, capture_output=True, text=True)
        if res.returncode:
            failed += 1
            print(f'  [{n}/{len(jobs)}] {stem} FAILED\n{res.stderr.strip()[-400:]}', flush=True)
            continue
        done += 1
        el = time.time() - t0
        eta = el / done * (len(jobs) - n)
        print(f'  [{n}/{len(jobs)}] {stem}  ({el / 60:.1f} min elapsed, '
              f'~{eta / 60:.0f} min left)', flush=True)

    size = sum(f.stat().st_size for f in out.iterdir() if f.is_file())
    print(f'\ndone: {done} rendered, {skipped} already present, {failed} failed')
    print(f'{out.resolve()}  ({size / 1e6:.0f} MB)')


if __name__ == '__main__':
    main()
