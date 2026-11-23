"""
Turn the fitted robot design (poses/robot_design_fit.json from fit_robot_design.py)
into solver cases for one patient path: poses/design_<path>_<load>.json, read by
both the SOFA scene and matlab/run_cases.m.

Frame mapping (fit -> robot):
  fit PCC frame: local z = insertion axis d, bend towards local x at phi = 0,
                 towards local y at phi = 90 deg; R0 = [cross(a,d), a, d].
  robot body   : x_b = tangent, bend towards y_b at plane = 0, towards z_b at 90 deg.
  => x_b = d, y_b = cross(a,d), z_b = a, and plane_j = phi_j, bend_j = kappa_j * L_j.
  Section 2's plane is relative to section 1's end frame in both (bend without twist).

Run:  python tools/make_design_cases.py <patients_dataset> <path name, e.g. patient44_right> [r1_mm]
      With r1_mm the cross-section is scaled self-similarly from the MATLAB design
      (section 2 keeps its 5.5/6 ratio) and the case names get an _r<r1> suffix.
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from fit_robot_design import kocher_axes, robot_points  # noqa: E402
from kocher_pose import rotm_to_quat_wxyz  # noqa: E402

L_RIGID_MM = 80.0
# MATLAB cross-sections [radius, chamber offset, chamber radius] in m
MATLAB_SEC = [(6.0e-3, 3.5e-3, 2.0e-3), (5.5e-3, 3.2e-3, 1.8e-3)]


def main(ds_dir, name, r1_mm=None):
    root = os.path.dirname(HERE)
    fit = json.load(open(os.path.join(root, 'poses', 'robot_design_fit.json')))
    pose3 = json.load(open(os.path.join(root, 'poses', 'kocher_patient3.json')))
    L1, L2 = fit['best_L1_mm'], fit['best_L2_mm']
    r = fit['per_path'][name]
    pid, side = name.rsplit('_', 1)
    K, F = kocher_axes(os.path.join(ds_dir, 'kochers', pid))[side]
    path_file = os.path.join(ds_dir, 'paths', name + '.csv')
    path = np.loadtxt(path_file, delimiter=',')

    d = (F - K) / np.linalg.norm(F - K)
    a = np.cross(d, [1.0, 0, 0]) if abs(d[0]) < 0.9 else np.cross(d, [0, 1.0, 0])
    a /= np.linalg.norm(a)
    R0 = np.column_stack([np.cross(a, d), a, d])
    R_body = np.column_stack([d, np.cross(a, d), a])
    q = rotm_to_quat_wxyz(R_body)

    s = np.linspace(0, L1 + L2, 161)
    pcc = robot_points((r['kappa1'], r['phi1'], r['kappa2'], r['phi2']), L1, L2, s, R0, path[0])

    base = {
        'patient_path': name,
        'ref_path_file': path_file,                     # senior's data; overlay only if present
        'fit_mean_dev_mm': r['err'],
        'base_position_m': list((path[0] - L_RIGID_MM * d) / 1000),
        'soft_base_m': list(path[0] / 1000),
        'base_quat_wxyz': list(q),
        'base_quat_xyzw': list(np.r_[q[1:], q[0]]),
        'sections': [
            dict(L_m=L1 / 1000, bend_rad=r['kappa1'] * L1, plane_rad=r['phi1']),
            dict(L_m=L2 / 1000, bend_rad=r['kappa2'] * L2, plane_rad=r['phi2']),
        ],
        'pcc_fit_centerline_m': (pcc / 1000).tolist(),
    }
    tag = ''
    if r1_mm is not None:
        f = r1_mm * 1e-3 / MATLAB_SEC[0][0]
        for sec, (rad, off, rc) in zip(base['sections'], MATLAB_SEC):
            sec.update(radius_m=rad * f, chamberOffset_m=off * f, chamberRadius_m=rc * f)
        tag = f'_r{r1_mm:g}'
    for load, g in [('nogravity', [0.0, 0.0, 0.0]), ('csf', pose3['gravity']['supine_head0_csf'])]:
        out = dict(base, gravity=g)
        fn = os.path.join(root, 'poses', f'design_{name}{tag}_{load}.json')
        json.dump(out, open(fn, 'w'), indent=1)
        print('wrote', fn)
    print('sections:', [(round(x['L_m'] * 1e3), round(np.degrees(x['bend_rad'])), round(np.degrees(x['plane_rad'])))
                        for x in base['sections']], '(L mm, bend deg, plane deg)')


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], float(sys.argv[3]) if len(sys.argv) > 3 else None)
