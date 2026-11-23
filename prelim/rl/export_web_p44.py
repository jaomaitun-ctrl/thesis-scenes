"""
Build a self-contained web viewer (one HTML file, three.js from a CDN) that replays recorded
insertions of patient 44 (rl/rollout.py --record-rod), for viewing in any browser.

    python3 rl/export_web_p44.py <out.html> <kocher_dir> <ventricle.stl> name=rollout.npz ...

Frame: LPS millimetres are shown as three.js (x, y, z) = (-L, S, P) = (R, S, P), right-handed,
superior up.  numpy only.
"""
import json
import os
import struct
import sys

import numpy as np

RUNS = {  # key -> (label, colour, shown)
    'v2': ('PPO v2 · tip + shape reward', '#ff7a1a', True),
    'v1': ('PPO v1 · tip reward', '#3b94ff', True),
    'baseline': ('Fitted schedule · no policy', '#9aa0a8', False),
}


def to3(p):
    p = np.asarray(p, float)
    return np.stack([-p[..., 0], p[..., 2], p[..., 1]], axis=-1)


def read_stl(fn):
    b = open(fn, 'rb').read()
    n = struct.unpack('<I', b[80:84])[0]
    a = np.frombuffer(b[84:84 + 50 * n], dtype=np.dtype([('n', '<f4', 3), ('v', '<f4', (3, 3)), ('a', '<u2')]))
    return a['v'].reshape(-1, 3).astype(float)


def read_fcsv(fn):
    rows = [l for l in open(fn).read().splitlines() if l and not l.startswith('#')]
    return np.array([[float(v) for v in r.split(',')[1:4]] for r in rows])


def r2(a, k=2):
    return np.round(np.asarray(a, float), k).tolist()


def main(out, kdir, stl, runs):
    tri = read_stl(stl)
    key = np.round(tri, 3)
    uniq, inv = np.unique(key, axis=0, return_inverse=True)
    faces = inv.reshape(-1, 3)
    # (x, y, z) -> (-x, z, y) has determinant +1, so the STL winding is kept
    oc = {f[:-5]: read_fcsv(os.path.join(kdir, f)) for f in os.listdir(kdir) if f.startswith('OC')}
    fs = sorted([read_fcsv(os.path.join(kdir, f))[-1] for f in os.listdir(kdir) if f.startswith('F')],
                key=lambda p: p[0])
    K = sorted([v[-1] for v in oc.values()], key=lambda p: p[0])[0]

    data = {'runs': []}
    ref = None
    for name, fn in runs:
        z = np.load(fn)
        ref = ref if ref is not None else z
        R, dep = z['rods_0'], z['depth_0']
        goal = z['goal']
        tipgoal = np.linalg.norm(R[:, -1] - goal, axis=1)
        s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(R[-1], axis=0), axis=1))]
        ins = R[-1][s >= s[-1] - dep[-1] + 1e-6]
        dev = np.min(np.linalg.norm(ins[:, None] - z['path'][None], axis=2), axis=1).mean()
        label, colour, shown = RUNS.get(name, (name, '#cccccc', True))
        data['runs'].append(dict(key=name, label=label, colour=colour, shown=shown,
                                 frames=r2(to3(R), 2), depth=r2(dep, 1), tipGoal=r2(tipgoal, 2),
                                 goalErr=round(float(z['goal_err_0']), 2), rodDev=round(float(dev), 2)))
    F, d = ref['F'], ref['d']
    v = K - F
    data.update(
        mesh=dict(vertices=r2(to3(uniq), 2), faces=faces.ravel().tolist()),
        path=r2(to3(ref['path']), 2), K=r2(to3(K)), F=r2(to3(F)), goal=r2(to3(ref['goal'])),
        sheathBase=r2(to3(F - d * float(ref['L_rigid_mm']))), sheathRadius=4.5, robotRadius=float(ref['r_mm']),
        sutures=[r2(to3(p)) for k, p in sorted(oc.items()) if k in ('OC_1', 'OC_2')],
        kocherToF=round(float(-np.dot(v, d)), 1), kocherOffAxis=round(float(np.linalg.norm(v - np.dot(v, d) * d)), 2))
    html = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'web_p44_template.html')).read()
    open(out, 'w').write(html.replace('/*__DATA__*/null', json.dumps(data, separators=(',', ':'))))
    print(f'wrote {out}: {len(uniq)} vertices, {len(faces)} faces, '
          + ', '.join(f'{r["key"]} {len(r["frames"])} frames' for r in data['runs'])
          + f', {os.path.getsize(out) / 1e6:.2f} MB')


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], sys.argv[3], [a.split('=', 1) for a in sys.argv[4:]])
