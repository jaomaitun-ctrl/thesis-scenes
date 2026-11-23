"""
Learning progress of a run from its Monitor CSVs (no SOFA needed):

    python3 rl/summarize.py rl/runs/ppo_p44_v1 [bin_steps]

Prints, per bin of environment steps: episodes, mean return, success rate, final insertion
depth, tip error at the end of the episode and wall-exit rate.
"""
import csv
import glob
import os
import sys


def main(run_dir, bin_steps=10_000):
    eps = []
    for fn in glob.glob(os.path.join(run_dir, 'monitor_*.monitor.csv')):
        with open(fn) as f:
            next(f)                                    # JSON header line
            rows = list(csv.DictReader(f))
        steps = 0
        for r in rows:                                 # per-env cumulative steps
            steps += int(r['l'])
            eps.append((steps, r))
    if not eps:
        print('no episodes yet'); return
    n_env = len(glob.glob(os.path.join(run_dir, 'monitor_*.monitor.csv')))
    eps.sort(key=lambda e: e[0])
    print(f'{len(eps)} episodes from {n_env} envs')
    print(f'{"steps":>9} {"eps":>4} {"return":>7} {"success":>7} {"depth mm":>8} {"tip err":>7} {"shape":>6} {"outside":>7}')
    b, cur = bin_steps, []
    for st, r in eps + [(float('inf'), None)]:
        if st * n_env > b or r is None:
            if cur:
                f = lambda k: sum(float(x[k]) for x in cur) / len(cur)
                succ = sum(x['success'] in ('True', '1', '1.0') for x in cur) / len(cur)
                out = sum(float(x['outside']) > 0 for x in cur) / len(cur)
                print(f'{b:9d} {len(cur):4d} {f("r"):7.2f} {succ:7.0%} {f("depth_mm"):8.1f} '
                      f'{f("tip_err_mm"):7.1f} {f("shape_err_mm"):6.1f} {out:7.0%}')
            cur = []
            while r is not None and st * n_env > b:
                b += bin_steps
        if r is not None:
            cur.append(r)


if __name__ == '__main__':
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 10_000)
