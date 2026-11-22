"""
Export a recorded insertion (rl/rollout.py --record-rod) of patient 44 for the Unity project
(UnityVentricleNavigation, scene built by Patient44SceneBuilder.cs).  numpy only.

    python3 rl/export_unity_p44.py <out_dir> <kocher_dir> <ventricle.stl> name=rollout.npz ...
    e.g. python3 rl/export_unity_p44.py ~/UnityVentricleNavigation/Assets/Data/Patient44 \
             ~/Documents/paths/simulation/patients_dataset/kochers/patient44 \
             ~/Documents/paths/simulation/patients_dataset/ventricles/patient44_right.stl \
             v1=unity_v1.npz v2=unity_v2.npz baseline=unity_baseline.npz

Everything stays in the LPS millimetre frame of the planned paths, the STL and the Slicer
fiducials (Unity converts with its usual (x, y, z) -> (x, z, y) / 1000 mapping).
Writes:
  patient44_right.stl           closed right lateral ventricle (the policy's lumen)
  <fiducial>.fcsv               Kocher fiducials (OC*: coronal suture, F*: ventricle entry)
  reference_path.csv            x,y,z  planned path from the frontal-horn entry F
  rod_<name>.csv                frame,depth_mm,node,x,y,z  every robot node at every RL step
                                (nodes not yet inserted sit on the sheath axis behind F)
  scene.csv                     key,x,y,z / key,value  Kocher point K, F, sheath axis, goal,
                                sheath length, robot radius, per-run final errors
"""
import os
import shutil
import sys

import numpy as np


def read_fcsv_last(fn):
    rows = [l for l in open(fn).read().splitlines() if l and not l.startswith('#')]
    return np.array([float(v) for v in rows[-1].split(',')[1:4]])


def main(out, kdir, stl, runs):
    os.makedirs(out, exist_ok=True)
    shutil.copyfile(stl, os.path.join(out, 'patient44_right.stl'))
    for f in sorted(os.listdir(kdir)):
        if f.endswith('.fcsv'):
            shutil.copyfile(os.path.join(kdir, f), os.path.join(out, f))
    # same choice as tools/fit_robot_design.kocher_axes: right side = smallest-x OC / F end
    oc = sorted([read_fcsv_last(os.path.join(kdir, f)) for f in os.listdir(kdir) if f.startswith('OC')],
                key=lambda p: p[0])
    fs = sorted([read_fcsv_last(os.path.join(kdir, f)) for f in os.listdir(kdir) if f.startswith('F')],
                key=lambda p: p[0])
    K, F_fid = oc[0], fs[0]

    lines, ref = [], None
    for name, fn in runs:
        z = np.load(fn)
        if ref is None:
            ref = z
            np.savetxt(os.path.join(out, 'reference_path.csv'), z['path'], delimiter=',', fmt='%.4f')
        R, depth = z['rods_0'], z['depth_0']
        with open(os.path.join(out, f'rod_{name}.csv'), 'w') as f:
            f.write('frame,depth_mm,node,x,y,z\n')
            for i, (pts, dep) in enumerate(zip(R, depth)):
                for j, p in enumerate(pts):
                    f.write(f'{i},{dep:.3f},{j},{p[0]:.4f},{p[1]:.4f},{p[2]:.4f}\n')
        P = z['path']
        s =np.r_[0, np.cumsum(np.linalg.norm(np.diff(R[-1], axis=0), axis=1))]
        inserted = R[-1][s >= s[-1] - depth[-1] + 1e-6]
        rod_dev = np.min(np.linalg.norm(inserted[:, None] - P[None], axis=2), axis=1).mean()
        lines += [f'goal_err_mm_{name},{float(z["goal_err_0"]):.3f}',
                  f'rod_dev_mm_{name},{rod_dev:.3f}', f'frames_{name},{len(R)}']
        print(f'{name}: {len(R)} frames x {R.shape[1]} nodes, goal err {float(z["goal_err_0"]):.2f} mm, '
              f'inserted rod {rod_dev:.2f} mm from path')

    F, d = ref['F'], ref['d']
    v = K - F
    with open(os.path.join(out, 'scene.csv'), 'w') as f:
        f.write('key,x,y,z\n')
        for k, p in [('kocher_K', K), ('entry_F', F), ('entry_F_fiducial', F_fid), ('sheath_axis_d', d),
                     ('goal', ref['goal'])]:
            f.write(f'{k},{p[0]:.4f},{p[1]:.4f},{p[2]:.4f}\n')
        for l in [f'sheath_length_mm,{float(ref["L_rigid_mm"]):.3f}', f'robot_radius_mm,{float(ref["r_mm"]):.3f}',
                  f'kocher_off_axis_mm,{np.linalg.norm(v - np.dot(v, d) * d):.3f}',
                  f'kocher_to_F_mm,{-np.dot(v, d):.3f}'] + lines:
            f.write(l + '\n')
    print(f'K {K.round(2)} -> F {F.round(2)}: {-np.dot(v, d):.1f} mm along the sheath axis, '
          f'{np.linalg.norm(v - np.dot(v, d) * d):.2f} mm off it')


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], sys.argv[3], [a.split('=', 1) for a in sys.argv[4:]])
