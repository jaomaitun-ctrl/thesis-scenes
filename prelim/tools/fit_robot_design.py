"""
Choose the two soft-section lengths of the robot from the patient paths.

Model: piecewise-constant-curvature (PCC) robot, base at the frontal-horn target F,
base axis = that patient's Kocher->F insertion direction (the rigid shaft), two soft
sections of lengths L1 + L2 = L_TOTAL, each with curvature kappa_i <= KAPPA_MAX and
bending-plane angle phi_i (relative to the previous frame).  For every design
(L1, L2) and every patient path, (kappa1, phi1, kappa2, phi2) are fitted to minimise
the mean distance between robot(s) and path(s) for s in [0, min(L_TOTAL, L_path)]
(follow-the-leader shape at full insertion).  The design with the lowest mean error
over all paths is reported.

Inputs (from the senior's dataset, ~/Documents/paths/simulation/patients_dataset):
  paths/<patient>_<side>.csv  (mm, LPS; starts at F)
  kochers/<patient>/OC*.fcsv  (last point of each file; the two lateral ones are the
                               left/right Kocher's points)
  kochers/<patient>/F*.fcsv   (frontal-horn targets; larger x = left in LPS)

Run:  python tools/fit_robot_design.py <patients_dataset> <out.json>
"""
import json
import os
import sys

import numpy as np

L_TOTAL = 80.0                       # [mm] general-path length
L1_GRID = np.arange(10.0, 70.1, 5.0)  # [mm] proximal section length candidates
KAPPA_MAX = 0.15                      # [1/mm] = 0.9 / r for r = 6 mm (Yeoh table limit)
DS = 1.0                              # [mm] comparison spacing
N_STARTS = 6                          # Nelder-Mead restarts (phi1 grid)


def read_fcsv_last(path):
    rows = [l for l in open(path) if l.strip() and not l.startswith('#')]
    return np.array([float(v) for v in rows[-1].split(',')[1:4]])


def kocher_axes(kdir):
    oc = [read_fcsv_last(os.path.join(kdir, f)) for f in os.listdir(kdir) if f.startswith('OC')]
    fs = [read_fcsv_last(os.path.join(kdir, f)) for f in os.listdir(kdir) if f.startswith('F')]
    oc = sorted(oc, key=lambda p: p[0])
    fs = sorted(fs, key=lambda p: p[0])
    # lateral Kocher's points = extreme x of the OC ends; midline end is in between
    return {'right': (oc[0], fs[0]), 'left': (oc[-1], fs[-1])}


def resample(p, s_query):
    s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))]
    return np.column_stack([np.interp(s_query, s, p[:, k]) for k in range(3)]), s[-1]


def rot_z(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def arc(kappa, phi, s):
    """PCC arc along local +z, bending towards local +x rotated by phi.  Returns (pts, R_end)."""
    if kappa < 1e-9:
        pts = np.column_stack([np.zeros_like(s), np.zeros_like(s), s])
        R = np.eye(3)
    else:
        th = kappa * s
        pts = np.column_stack([(1 - np.cos(th)) / kappa, np.zeros_like(s), np.sin(th) / kappa])
        ct, st = np.cos(th[-1]), np.sin(th[-1])
        R = np.array([[ct, 0, st], [0, 1, 0], [-st, 0, ct]])
    Rz = rot_z(phi)
    return pts @ Rz.T, Rz @ R @ Rz.T


def robot_points(params, L1, L2, s_query, R0, p0):
    k1, f1, k2, f2 = params
    s1 = s_query[s_query <= L1]
    s2 = s_query[s_query > L1] - L1
    P1, R1 = arc(k1, f1, np.r_[s1, L1])
    end1 = P1[-1]
    P1 = P1[:-1]
    Rb2 = R1 @ rot_z(0.0)
    P2, _ = arc(k2, f2, s2) if len(s2) else (np.zeros((0, 3)), None)
    local = np.vstack([P1, end1 + P2 @ Rb2.T])
    return p0 + local @ R0.T


def nelder_mead(f, x0, step, iters=250):
    n = len(x0)
    pts = [np.asarray(x0, float)] + [np.asarray(x0, float) + np.eye(n)[i] * step[i] for i in range(n)]
    vals = [f(p) for p in pts]
    for _ in range(iters):
        order = np.argsort(vals)
        pts = [pts[i] for i in order]
        vals = [vals[i] for i in order]
        c = np.mean(pts[:-1], axis=0)
        xr = c + (c - pts[-1]); fr = f(xr)
        if fr < vals[0]:
            xe = c + 2 * (c - pts[-1]); fe = f(xe)
            pts[-1], vals[-1] = (xe, fe) if fe < fr else (xr, fr)
        elif fr < vals[-2]:
            pts[-1], vals[-1] = xr, fr
        else:
            xc = c + 0.5 * (pts[-1] - c); fc = f(xc)
            if fc < vals[-1]:
                pts[-1], vals[-1] = xc, fc
            else:
                pts = [pts[0] + 0.5 * (p - pts[0]) for p in pts]
                vals = [f(p) for p in pts]
    i = int(np.argmin(vals))
    return pts[i], vals[i]


def fit_patient(path, K, F, L1, L2):
    d = (F - K) / np.linalg.norm(F - K)
    a = np.cross(d, [1.0, 0, 0]) if abs(d[0]) < 0.9 else np.cross(d, [0, 1.0, 0])
    a /= np.linalg.norm(a)
    R0 = np.column_stack([np.cross(a, d), a, d])          # local z = insertion axis
    L_path = np.sum(np.linalg.norm(np.diff(path, axis=0), axis=1))
    s_query = np.arange(0, min(L1 + L2, L_path) + 1e-9, DS)
    target, _ = resample(path, s_query)

    def cost(x):
        k1, f1, k2, f2 = x
        pen = 1e3 * (max(0, k1 - KAPPA_MAX) + max(0, -k1) + max(0, k2 - KAPPA_MAX) + max(0, -k2))
        k1c, k2c = np.clip([k1, k2], 0, KAPPA_MAX)
        r = robot_points((k1c, f1, k2c, f2), L1, L2, s_query, R0, path[0])
        return np.mean(np.linalg.norm(r - target, axis=1)) + pen

    best = (None, np.inf)
    for f1 in np.linspace(0, 2 * np.pi, N_STARTS, endpoint=False):
        x, v = nelder_mead(cost, [0.05, f1, 0.03, 0.0], [0.03, 1.0, 0.02, 1.0])
        if v < best[1]:
            best = (x, v)
    x, v = best
    entry_angle = np.degrees(np.arccos(np.clip(d @ (path[min(5, len(path) - 1)] - path[0]) /
                                               np.linalg.norm(path[min(5, len(path) - 1)] - path[0]), -1, 1)))
    return dict(err=float(v), kappa1=float(np.clip(x[0], 0, KAPPA_MAX)), phi1=float(x[1] % (2 * np.pi)),
                kappa2=float(np.clip(x[2], 0, KAPPA_MAX)), phi2=float(x[3] % (2 * np.pi)),
                L_path=float(L_path), entry_vs_path_deg=float(entry_angle),
                tip_err=float(np.linalg.norm(robot_points((np.clip(x[0], 0, KAPPA_MAX), x[1],
                                                         np.clip(x[2], 0, KAPPA_MAX), x[3]),
                                                        L1, L2, s_query[-1:], R0, path[0])[0] - target[-1])))


def main(ds_dir, out_json):
    pdir, kdir = os.path.join(ds_dir, 'paths'), os.path.join(ds_dir, 'kochers')
    cases = []
    for f in sorted(os.listdir(pdir)):
        if not f.endswith('.csv'):
            continue
        pid, side = f[:-4].rsplit('_', 1)
        K, F = kocher_axes(os.path.join(kdir, pid))[side]
        path = np.loadtxt(os.path.join(pdir, f), delimiter=',')
        if np.linalg.norm(path[0] - F) > 2.0:
            print(f'  warning: {f} starts {np.linalg.norm(path[0] - F):.1f} mm from F')
        cases.append((f[:-4], path, K, F))
    print(f'{len(cases)} paths')

    results = {}
    for L1 in L1_GRID:
        L2 = L_TOTAL - L1
        fits = [fit_patient(p, K, F, L1, L2) for _, p, K, F in cases]
        e = np.array([r['err'] for r in fits])
        k1 = np.array([r['kappa1'] for r in fits]); k2 = np.array([r['kappa2'] for r in fits])
        results[float(L1)] = fits
        print(f'L1 = {L1:4.0f}  L2 = {L2:4.0f} mm | mean dev {e.mean():5.2f} mm (median {np.median(e):5.2f}, '
              f'90th {np.percentile(e, 90):5.2f}) | kappa1 {k1.mean():.3f} (at cap {np.mean(k1 > 0.98 * KAPPA_MAX):.0%}) '
              f'kappa2 {k2.mean():.3f} | bend1 {np.degrees(k1.mean() * L1):4.0f} deg bend2 {np.degrees(k2.mean() * L2):4.0f} deg',
              flush=True)

    best_L1 = min(results, key=lambda L: np.mean([r['err'] for r in results[L]]))
    fits = results[best_L1]
    out = dict(L_total_mm=L_TOTAL, kappa_max_1pmm=KAPPA_MAX, best_L1_mm=best_L1, best_L2_mm=L_TOTAL - best_L1,
               mean_dev_by_L1={str(k): float(np.mean([r['err'] for r in v])) for k, v in results.items()},
               per_path={name: r for (name, *_), r in zip(cases, fits)})
    k1 = np.array([r['kappa1'] for r in fits]); k2 = np.array([r['kappa2'] for r in fits])
    out['best_summary'] = dict(
        kappa1_mean=float(k1.mean()), kappa1_p90=float(np.percentile(k1, 90)),
        kappa2_mean=float(k2.mean()), kappa2_p90=float(np.percentile(k2, 90)),
        bend1_deg_mean=float(np.degrees(k1.mean() * best_L1)),
        bend2_deg_mean=float(np.degrees(k2.mean() * (L_TOTAL - best_L1))),
        entry_vs_path_deg_mean=float(np.mean([r['entry_vs_path_deg'] for r in fits])))
    json.dump(out, open(out_json, 'w'), indent=1)
    print('best:', best_L1, L_TOTAL - best_L1, json.dumps(out['best_summary'], indent=1))


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
