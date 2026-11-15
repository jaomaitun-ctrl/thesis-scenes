"""
Robot base pose and gravity for a Kocher's-point insertion, in the patient3 mesh frame.

Input : centerline.csv from ventricle_render_kit/plan_path.py (mm, mesh frame;
        section 0 = rigid shaft, 1/2 = soft sections).
Output: poses/kocher_patient3.json, read by both the MATLAB reference runner and
        the SOFA scene, so both solve exactly the same configuration.

Frame of the mesh (from plan_path.py): -y = anterior, +z = superior, x = lateral
(the planned path uses the x < 0 ventricle).

Pose convention
- Insertion axis d = direction of the planned rigid shaft (top -> junction).
- Soft base = the planned rigid/soft junction J; the robot's 80 mm rigid part
  extends back from J along -d (the planner draws only 35.5 mm of it).
- Body frame: x_b = d (rod tangent, as in the MATLAB script), y_b = anatomical
  POSTERIOR projected perpendicular to d, z_b = x_b cross y_b (lateral).
  So a bend with bendingPlane = 0 (sec1 in the MATLAB script) bends posteriorly,
  i.e. from the frontal horn towards the ventricular body.

Gravity (patient supine, head of bed raised by beta): unit direction
  g_hat = cos(beta) * posterior + sin(beta) * inferior   (mesh frame).
In CSF the soft sections feel weight minus buoyancy:  g_eff = g (1 - rho_CSF/rho).

Run:  python3 tools/kocher_pose.py <centerline.csv> poses/kocher_patient3.json
"""
import json
import sys

import numpy as np

L_RIGID_MM = 80.0            # rigid insertion length in cosserat_yeoh_quasistatic.m
RHO_ROBOT = 1050.0           # [kg/m^3] both soft sections in the MATLAB script
RHO_CSF = 1007.0             # [kg/m^3] CSF at body temperature (~1.0063-1.0075 g/ml)
G = 9.81
POSTERIOR = np.array([0.0, 1.0, 0.0])
INFERIOR = np.array([0.0, 0.0, -1.0])


def rotm_to_quat_wxyz(R):
    """Rotation matrix -> unit quaternion [w x y z] (Shepperd's method)."""
    t = np.trace(R)
    if t > 0:
        s = 2 * np.sqrt(1 + t)
        q = [0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s]
    else:
        i = int(np.argmax(np.diag(R)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = 2 * np.sqrt(1 + R[i, i] - R[j, j] - R[k, k])
        q = np.zeros(4)
        q[0] = (R[k, j] - R[j, k]) / s
        q[1 + i] = 0.25 * s
        q[1 + j] = (R[j, i] + R[i, j]) / s
        q[1 + k] = (R[k, i] + R[i, k]) / s
    q = np.asarray(q, float)
    return q / np.linalg.norm(q)


def main(csv_path, out_path):
    data = np.loadtxt(csv_path, delimiter=',', skiprows=1)
    x, sec = data[:, :3], data[:, 3]
    rigid = x[sec == 0]
    J = x[sec == 1][0]
    d = rigid[-1] - rigid[0]
    d /= np.linalg.norm(d)

    y_b = POSTERIOR - (POSTERIOR @ d) * d
    y_b /= np.linalg.norm(y_b)
    z_b = np.cross(d, y_b)
    R = np.column_stack([d, y_b, z_b])
    q_wxyz = rotm_to_quat_wxyz(R)

    base_mm = J - L_RIGID_MM * d
    gravity = {}
    for beta in (0, 30):
        b = np.deg2rad(beta)
        g_hat = np.cos(b) * POSTERIOR + np.sin(b) * INFERIOR
        gravity[f'supine_head{beta}_air'] = list(G * g_hat)
        gravity[f'supine_head{beta}_csf'] = list(G * (1 - RHO_CSF / RHO_ROBOT) * g_hat)

    pose = {
        'source': 'ventricle_render_kit/' + csv_path.replace('\\', '/').split('/')[-1],
        'frame': 'patient3 mesh frame, metres; -y anterior, +z superior',
        'insertion_dir': list(d),
        'soft_base_m': list(J / 1000),
        'base_position_m': list(base_mm / 1000),
        'base_quat_wxyz': list(q_wxyz),                                  # MATLAB
        'base_quat_xyzw': list(np.r_[q_wxyz[1:], q_wxyz[0]]),            # SOFA
        'rho_robot': RHO_ROBOT, 'rho_csf': RHO_CSF,
        'gravity': gravity,
        'planned_soft_length_mm': float(data[sec > 0, 4].max() - data[sec > 0, 4].min()),
    }
    with open(out_path, 'w') as f:
        json.dump(pose, f, indent=2)
    print(json.dumps(pose, indent=2))


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
