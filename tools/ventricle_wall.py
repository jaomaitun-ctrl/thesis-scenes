"""
Prepare a patient ventricle mesh as the collision wall for the SOFA scene, and check
how a robot centreline sits inside it (numpy only).

  * reads the binary/ASCII STL (mm, LPS - same frame as the planned paths/markups)
  * reports closedness, the entry point E where the Kocher->F axis crosses the wall,
    and the clearance (distance to wall minus robot radius) along given centrelines
  * writes a copy with an ENTRY HOLE: triangles closer than HOLE_MARGIN + radius to the
    insertion segment K->F are removed, so the robot can pass through the wall it was
    inserted through (the ventriculostomy) without colliding with it.

Run:  python tools/ventricle_wall.py <design_case.json> <patients_dataset> <out_wall.stl>
"""
import json
import os
import struct
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from fit_robot_design import kocher_axes  # noqa: E402

HOLE_MARGIN_MM = 1.5
SHEATH_CLEARANCE_MM = 5.0   # sheath axis must stay this far from the wall (r + alarm + margin)


def read_stl(fn):
    data = open(fn, 'rb').read()
    n = struct.unpack('<I', data[80:84])[0]
    if 84 + 50 * n == len(data):
        rec = np.frombuffer(data[84:], dtype=np.dtype([('n', '<f4', 3), ('v', '<f4', (3, 3)), ('a', '<u2')]))
        tri = rec['v'].astype(float)
    else:
        v = [list(map(float, l.split()[1:4])) for l in data.decode().splitlines() if l.strip().startswith('vertex')]
        tri = np.array(v).reshape(-1, 3, 3)
    return tri


def write_stl(fn, tri):
    n = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
    rec = np.zeros(len(tri), dtype=np.dtype([('n', '<f4', 3), ('v', '<f4', (3, 3)), ('a', '<u2')]))
    rec['n'], rec['v'] = n, tri
    with open(fn, 'wb') as f:
        f.write(b'ventricle wall with entry hole'.ljust(80, b' '))
        f.write(struct.pack('<I', len(tri)))
        f.write(rec.tobytes())


def closest_points_on_triangles(p, tri):
    """Closest point on each triangle to point p (Ericson, vectorised over triangles)."""
    a, b, c = tri[:, 0], tri[:, 1], tri[:, 2]
    ab, ac, ap = b - a, c - a, p - a
    d1, d2 = np.einsum('ij,ij->i', ab, ap), np.einsum('ij,ij->i', ac, ap)
    bp = p - b
    d3, d4 = np.einsum('ij,ij->i', ab, bp), np.einsum('ij,ij->i', ac, bp)
    cp = p - c
    d5, d6 = np.einsum('ij,ij->i', ab, cp), np.einsum('ij,ij->i', ac, cp)
    va, vb, vc = d3 * d6 - d5 * d4, d5 * d2 - d1 * d6, d1 * d4 - d3 * d2
    denom = np.where(va + vb + vc == 0, 1e-30, va + vb + vc)
    v, w = vb / denom, vc / denom
    out = a + ab * v[:, None] + ac * w[:, None]                       # face region
    m = (d1 <= 0) & (d2 <= 0); out[m] = a[m]
    m = (d3 >= 0) & (d4 <= d3); out[m] = b[m]
    m = (d6 >= 0) & (d5 <= d6); out[m] = c[m]
    m = (vc <= 0) & (d1 >= 0) & (d3 <= 0)
    t = d1 / np.where(d1 - d3 == 0, 1e-30, d1 - d3); out[m] = (a + ab * t[:, None])[m]
    m = (vb <= 0) & (d2 >= 0) & (d6 <= 0)
    t = d2 / np.where(d2 - d6 == 0, 1e-30, d2 - d6); out[m] = (a + ac * t[:, None])[m]
    m = (va <= 0) & ((d4 - d3) >= 0) & ((d5 - d6) >= 0)
    t = (d4 - d3) / np.where((d4 - d3) + (d5 - d6) == 0, 1e-30, (d4 - d3) + (d5 - d6))
    out[m] = (b + (c - b) * t[:, None])[m]
    return out


def distance_to_mesh(pts, tri):
    return np.array([np.linalg.norm(closest_points_on_triangles(p, tri) - p, axis=1).min() for p in pts])


def inside(pts, tri, direction=(0.5773, 0.5774, 0.5773)):
    """Ray-parity inside test (Moller-Trumbore), one ray direction per point."""
    dvec = np.array(direction) / np.linalg.norm(direction)
    e1, e2 = tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]
    h = np.cross(dvec, e2)
    det = np.einsum('ij,ij->i', e1, h)
    ok = np.abs(det) > 1e-12
    inv = np.where(ok, 1 / np.where(ok, det, 1), 0)
    res = []
    for p in pts:
        s = p - tri[:, 0]
        u = np.einsum('ij,ij->i', s, h) * inv
        q = np.cross(s, e1)
        v = (q @ dvec) * inv
        t = np.einsum('ij,ij->i', e2, q) * inv
        hit = ok & (u >= 0) & (v >= 0) & (u + v <= 1) & (t > 1e-9)
        res.append(hit.sum() % 2 == 1)
    return np.array(res)


def segment_distance(points, a, b):
    ab = b - a
    t = np.clip(((points - a) @ ab) / (ab @ ab), 0, 1)
    return np.linalg.norm(points - (a + t[:, None] * ab), axis=1)


def main(case_json, ds_dir, out_stl):
    case = json.load(open(case_json))
    name = case['patient_path']
    pid, side = name.rsplit('_', 1)
    K, F = kocher_axes(os.path.join(ds_dir, 'kochers', pid))[side]
    tri = read_stl(os.path.join(ds_dir, 'ventricles', f'{name}.stl'))
    r_mm = 1e3 * case['sections'][0].get('radius_m', 6e-3)

    v = tri.reshape(-1, 3)
    _, inv_idx = np.unique(np.round(v, 5), axis=0, return_inverse=True)
    f = inv_idx.reshape(-1, 3)
    edges = np.sort(np.vstack([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]]), axis=1)
    _, cnt = np.unique(edges, axis=0, return_counts=True)
    print(f'{name}: {len(tri)} triangles, bounds {v.min(0).round(1)} .. {v.max(0).round(1)} mm, '
          f'edges used once {np.sum(cnt == 1)}, more than twice {np.sum(cnt > 2)} (0/0 = closed)')

    d = (F - K) / np.linalg.norm(F - K)
    line = K + np.outer(np.linspace(0, np.linalg.norm(F - K) * 1.2, 600), d)
    ins = inside(line, tri)
    first = int(np.argmax(ins)) if ins.any() else None
    print(f'F inside ventricle: {inside(F[None], tri)[0]}, wall clearance at F '
          f'{distance_to_mesh(F[None], tri)[0]:.2f} mm')
    if first is not None:
        E = line[first]
        print(f'entry point E (axis enters the ventricle) {E.round(2)}, {np.linalg.norm(E - K):.1f} mm from K, '
              f'{np.linalg.norm(F - E):.1f} mm before F')

    curves = {'patient path': np.loadtxt(case['ref_path_file'], delimiter=','),
              'PCC fit (robot)': np.array(case['pcc_fit_centerline_m']) * 1e3}
    refs = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(case_json))), 'matlab_refs')
    tag = os.path.basename(case_json)[:-5]
    for suffix in ['', '_sofa']:
        fn = os.path.join(refs, tag + suffix + '.csv')
        if os.path.exists(fn):
            curves[f'{"SOFA" if suffix else "MATLAB"} equilibrium'] = np.loadtxt(fn, delimiter=',') * 1e3
    for label, c in curves.items():
        dist = distance_to_mesh(c, tri)
        inn = inside(c, tri)
        clr = dist - r_mm
        print(f'  {label:22s}: {len(c)} pts, outside lumen {np.sum(~inn)}, min wall distance {dist.min():.2f} mm, '
              f'min clearance (r = {r_mm:.2f}) {clr.min():+.2f} mm, points with clearance < 0: {np.sum(clr < 0)}')

    # Entry hole: grow it until the sheath axis (80 mm behind F .. F) keeps
    # SHEATH_CLEARANCE_MM from the wall, so spheres still held in the sheath never touch it
    cent = tri.mean(axis=1)
    axis = F[None] + np.outer(np.linspace(-80, 0, 161), d)
    margin = HOLE_MARGIN_MM
    while True:
        hole = segment_distance(cent, F - 80 * d, F) < r_mm + margin
        clear = distance_to_mesh(axis, tri[~hole]).min()
        if clear >= SHEATH_CLEARANCE_MM or margin > 10:
            break
        margin += 0.5
    write_stl(out_stl, tri[~hole])
    print(f'entry hole: removed {hole.sum()} triangles within {r_mm + margin:.1f} mm of the sheath axis; '
          f'sheath-axis clearance {clear:.2f} mm -> {out_stl}')


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], sys.argv[3])
