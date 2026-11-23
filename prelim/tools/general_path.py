"""
Recompute the "general path" (senior's thesis, Eq. 4.1-4.3, Fig. 4.2) from the
per-patient planned paths, and its curvature / torsion along arc length.

Input : a folder of <patient>_<left|right>.csv paths (mm, LPS, one point per row),
        e.g. ~/Documents/paths/simulation/patients_dataset/paths.  Each path starts at
        the marked frontal-horn target F (see functions/simulationUtils/generatePath.m),
        so s = 0 is at the frontal horn, not at the skull.
Output: <out>/general_path_<side>.csv   columns s_mm, x, y, z, kappa_1pmm, tau_1pmm, n_paths
        <out>/general_path_<side>_per_patient_stats.csv
        <out>/general_path.png

Method (as in the thesis):
  T_i(s) = dr_i/ds on a uniform arc-length grid               (4.1)
  T_hat(s) = mean_i T_i(s)                                    (4.2)  NOT renormalised
  r_hat(s) = integral_0^s T_hat                               (4.3)
Paths have different lengths; at each s the mean runs over the paths that reach s,
and the general path is kept where at least MIN_FRACTION of the paths reach.
Curvature/torsion of r_hat use the general (non-unit-speed) formulas
  kappa = |r' x r''| / |r'|^3,  tau = (r' x r'') . r''' / |r' x r''|^2.

Run:  python3 tools/general_path.py <paths_dir> <out_dir>
"""
import os
import sys
from collections import defaultdict

import numpy as np

DS = 0.5                 # [mm] arc-length grid
MIN_FRACTION = 0.5       # keep s where >= this fraction of paths still exist
SMOOTH_MM = 2.0          # [mm] Gaussian smoothing before differentiating r_hat


def resample(p, ds=DS):
    seg = np.linalg.norm(np.diff(p, axis=0), axis=1)
    keep = np.r_[True, seg > 1e-9]
    p = p[keep]
    s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))]
    sg = np.arange(0, s[-1], ds)
    return sg, np.column_stack([np.interp(sg, s, p[:, k]) for k in range(3)])


def smooth(r, sig):
    """Gaussian smoothing along the rows, edges padded with the end values (numpy only)."""
    h = int(np.ceil(4 * sig))
    k = np.exp(-0.5 * (np.arange(-h, h + 1) / sig) ** 2)
    k /= k.sum()
    rp = np.vstack([np.repeat(r[:1], h, 0), r, np.repeat(r[-1:], h, 0)])
    return np.column_stack([np.convolve(rp[:, j], k, mode='valid') for j in range(r.shape[1])])


def curvature_torsion(r, ds=DS, smooth_mm=SMOOTH_MM):
    rs = smooth(r, smooth_mm / ds)
    d1 = np.gradient(rs, ds, axis=0)
    d2 = np.gradient(d1, ds, axis=0)
    d3 = np.gradient(d2, ds, axis=0)
    c = np.cross(d1, d2)
    cn = np.linalg.norm(c, axis=1)
    kappa = cn / np.linalg.norm(d1, axis=1) ** 3
    tau = np.einsum('ij,ij->i', c, d3) / np.maximum(cn ** 2, 1e-12)
    return kappa, tau


def general_path(paths):
    tangents = []
    for p in paths:
        _, r = resample(p)
        t = np.gradient(r, DS, axis=0)
        t /= np.linalg.norm(t, axis=1, keepdims=True)
        tangents.append(t)
    n_max = max(len(t) for t in tangents)
    T = np.full((len(tangents), n_max, 3), np.nan)
    for i, t in enumerate(tangents):
        T[i, :len(t)] = t
    n = np.sum(~np.isnan(T[:, :, 0]), axis=0)
    keep = n >= MIN_FRACTION * len(tangents)
    last = np.where(keep)[0][-1] + 1
    T_hat = np.nanmean(T[:, :last], axis=0)                 # Eq. 4.2 (not renormalised)
    r_hat = np.vstack([np.zeros(3), np.cumsum(0.5 * (T_hat[1:] + T_hat[:-1]) * DS, axis=0)])
    s = np.arange(last) * DS
    return s, r_hat, T_hat, n[:last]


def main(paths_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    groups = defaultdict(list)
    for f in sorted(os.listdir(paths_dir)):
        if f.endswith('.csv') and ('_left' in f or '_right' in f):
            side = 'left' if '_left' in f else 'right'
            groups[side].append((f[:-4], np.loadtxt(os.path.join(paths_dir, f), delimiter=',')))

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, axs = plt.subplots(2, 3, figsize=(16, 9))
    except ImportError:
        plt = None
        print('(matplotlib not available: skipping general_path.png)')
    for row, side in enumerate(['left', 'right']):
        names, paths = zip(*groups[side])
        lengths, stats = [], []
        for name, p in groups[side]:
            s_i, r_i = resample(p)
            k_i, t_i = curvature_torsion(r_i)
            lengths.append(s_i[-1])
            stats.append((name, s_i[-1], k_i.max(), s_i[k_i.argmax()],
                          np.degrees(np.trapezoid(k_i[s_i < 20], s_i[s_i < 20]))))
        s, r_hat, T_hat, n = general_path(paths)
        kappa, tau = curvature_torsion(r_hat)
        L_hat = np.sum(np.linalg.norm(np.diff(r_hat, axis=0), axis=1))
        ik, it = kappa.argmax(), np.abs(tau).argmax()
        print(f'[{side}] {len(paths)} paths, length {np.mean(lengths):.1f} +- {np.std(lengths):.1f} mm '
              f'(min {np.min(lengths):.1f}, max {np.max(lengths):.1f})')
        print(f'   general path kept to s = {s[-1]:.1f} mm (>= {MIN_FRACTION:.0%} of paths); '
              f'its geometric length {L_hat:.1f} mm (|T_hat| mean {np.linalg.norm(T_hat, axis=1).mean():.3f})')
        print(f'   max curvature {kappa[ik]:.3f} 1/mm (R = {1 / kappa[ik]:.1f} mm) at s = {s[ik]:.1f} mm; '
              f'max |torsion| {tau[it]:+.3f} 1/mm at s = {s[it]:.1f} mm')
        print(f'   turn of general path in first 20 mm: {np.degrees(np.trapezoid(kappa[s < 20], s[s < 20])):.0f} deg, '
              f'whole path: {np.degrees(np.trapezoid(kappa, s)):.0f} deg')
        st = np.array([x[1:] for x in stats], float)
        print(f'   per patient: max curvature {st[:, 1].mean():.3f} +- {st[:, 1].std():.3f} 1/mm at '
              f's = {st[:, 2].mean():.1f} +- {st[:, 2].std():.1f} mm; turn in first 20 mm '
              f'{st[:, 3].mean():.0f} +- {st[:, 3].std():.0f} deg')
        np.savetxt(os.path.join(out_dir, f'general_path_{side}.csv'),
                   np.column_stack([s, r_hat, kappa, tau, n]), delimiter=',',
                   header='s_mm,x_mm,y_mm,z_mm,kappa_1pmm,tau_1pmm,n_paths', comments='')
        with open(os.path.join(out_dir, f'general_path_{side}_per_patient_stats.csv'), 'w') as f:
            f.write('path,length_mm,kappa_max_1pmm,s_at_kappa_max_mm,turn_first_20mm_deg\n')
            for x in stats:
                f.write(','.join([x[0]] + [f'{v:.4f}' for v in x[1:]]) + '\n')

        if plt is None:
            continue
        ax = fig.add_subplot(2, 3, 3 * row + 1, projection='3d')
        axs[row, 0].remove()
        for p in paths:
            q = p - p[0]
            ax.plot(q[:, 0], q[:, 1], q[:, 2], color='#7fb3e6', lw=0.8, ls='--')
        ax.plot(r_hat[:, 0], r_hat[:, 1], r_hat[:, 2], color='red', lw=2.5)
        ax.set_title(f'{side.capitalize()} ventricle paths (n = {len(paths)})')
        ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')
        axs[row, 1].plot(s, kappa, color='#1f3fd8')
        axs[row, 1].set_ylabel('curvature [1/mm]'); axs[row, 1].set_xlabel('s [mm]'); axs[row, 1].grid(alpha=.3)
        axs[row, 2].plot(s, tau, color='#d81f2f')
        axs[row, 2].set_ylabel('torsion [1/mm]'); axs[row, 2].set_xlabel('s [mm]'); axs[row, 2].grid(alpha=.3)
    if plt is not None:
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, 'general_path.png'), dpi=90)
    print('wrote', out_dir)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
