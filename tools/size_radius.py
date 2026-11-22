"""
Radius sizing for the resized robot (25 + 55 mm), from the per-path fitted
curvatures in poses/robot_design_fit.json.

The cross-section is scaled self-similarly from the MATLAB design (chamber offset
and chamber radius keep their ratio to the outer radius), with the same material.
Then, for a required curvature kappa:
  fibre strain   eps = kappa * r                        (grows with r)
  pressure       p   from the Yeoh moment law M(kappa)  (~ E * kappa * r: grows with r)
  wall thickness r - offset - chamber radius            (shrinks with r)
So a smaller radius lowers strain and pressure, but thins the walls.  Pressure also
scales with C10, which the MATLAB script marks as a placeholder until tested.

Run:  python tools/size_radius.py
"""
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
FIT = json.load(open(os.path.join(os.path.dirname(HERE), 'poses', 'robot_design_fit.json')))

# MATLAB design, section 1 / section 2 (radius, offset/r, chamber r / r, Yeoh C)
SEC = [dict(r=6.0e-3, off=3.5 / 6.0, rc=2.0 / 6.0, C=[58.3e3, 5e3, 0.5e3]),
       dict(r=5.5e-3, off=3.2 / 5.5, rc=1.8 / 5.5, C=[41.7e3, 4e3, 0.4e3])]
RADII_MM = [2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0]


def yeoh_moment(r, C, k):
    nz = 800
    dz = 2 * r / nz
    z = -r + dz * (np.arange(1, nz + 1) - 0.5)
    w = 2 * np.sqrt(r ** 2 - z ** 2) * dz
    lam = 1 + k * z
    I1 = lam ** 2 + 2 / lam
    P = (2 * lam - 2 / lam ** 2) * (C[0] + 2 * C[1] * (I1 - 3) + 3 * C[2] * (I1 - 3) ** 2)
    return np.sum(P * z * w)


def max_chamber_pressure(r, off, rc, C, k):
    """Largest chamber pressure [Pa] from pressureFromBend, worst case over bending planes."""
    if k * r >= 0.9:
        return np.nan
    M = yeoh_moment(r, C, k)
    a = 2 * np.pi * np.arange(3) / 3
    A = np.vstack([off * np.sin(a), -off * np.cos(a)])
    worst = 0.0
    for phi in np.linspace(0, 2 * np.pi / 3, 13):       # 3-fold symmetric
        tau = np.linalg.pinv(A) @ (M * np.array([-np.sin(phi), np.cos(phi)]))
        worst = max(worst, ((tau - tau.min()) / (np.pi * rc ** 2)).max())
    return worst


def main():
    per = FIT['per_path'].values()
    k1 = np.array([v['kappa1'] for v in per]) * 1e3     # 1/m
    k2 = np.array([v['kappa2'] for v in per]) * 1e3
    q = {'p50': 50, 'p90': 90, 'max': 100}
    print(f"fitted curvature over {len(k1)} paths [1/mm]: sec1 p50 {np.percentile(k1, 50) / 1e3:.3f} "
          f"p90 {np.percentile(k1, 90) / 1e3:.3f} max {k1.max() / 1e3:.3f} | sec2 p50 "
          f"{np.percentile(k2, 50) / 1e3:.3f} p90 {np.percentile(k2, 90) / 1e3:.3f} max {k2.max() / 1e3:.3f}")
    scale2 = SEC[1]['r'] / SEC[0]['r']                  # keep sec2 slightly thinner, as in MATLAB
    print(f"{'r1 [mm]':>7} {'OD':>5} | {'strain1 p50/p90/max':>21} | {'strain2 p90':>11} | "
          f"{'p1 [kPa] p50/p90':>17} | {'p2 [kPa] p90':>12} | {'wall1 [mm]':>10}")
    rows = []
    for r_mm in RADII_MM:
        r1, r2 = r_mm * 1e-3, r_mm * 1e-3 * scale2
        e1 = [np.percentile(k1, q[x]) * r1 for x in ('p50', 'p90', 'max')]
        e2 = np.percentile(k2, 90) * r2
        s1, s2 = SEC
        p1 = [max_chamber_pressure(r1, s1['off'] * r1, s1['rc'] * r1, s1['C'], np.percentile(k1, x)) / 1e3
              for x in (50, 90)]
        p2 = max_chamber_pressure(r2, s2['off'] * r2, s2['rc'] * r2, s2['C'], np.percentile(k2, 90)) / 1e3
        wall = (r1 - s1['off'] * r1 - s1['rc'] * r1) * 1e3
        rows.append(dict(r_mm=r_mm, strain1=e1, strain2_p90=e2, p1=p1, p2_p90=p2, wall1_mm=wall))
        print(f"{r_mm:7.1f} {2 * r_mm:5.1f} | {100 * e1[0]:5.0f}% {100 * e1[1]:5.0f}% {100 * e1[2]:5.0f}% | "
              f"{100 * e2:10.0f}% | {p1[0]:7.0f} {p1[1]:8.0f} | {p2:12.0f} | {wall:10.2f}")
    return rows


if __name__ == '__main__':
    main()
