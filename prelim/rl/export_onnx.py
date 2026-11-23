"""
Export a trained PPO policy (stable-baselines3) to ONNX for Unity (Sentis / Inference Engine),
plus a JSON spec of the observation and action layout the Unity side must reproduce.

    rl/run.sh rl/export_onnx.py rl/runs/ppo_p44_v2/ckpt_50000_steps.zip [out_dir]

Graph: input "obs" float32 [batch, 35] -> output "action" float32 [batch, 5], the
DETERMINISTIC action (Gaussian mean) clipped to [-1, 1].  Opset 15 (supported by Sentis).
Observations are not normalised (VecNormalize was used with norm_obs=False), so Unity feeds
the raw observation exactly as VentricleNavEnv._obs builds it.

Checks: the ONNX output is compared with SB3's model.predict(deterministic=True) on
observations recorded from the real environment (requires SOFA; skipped with --no-env) and
on random observations, using onnxruntime.
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
OPSET = 15


def main():
    args = [a for a in sys.argv[1:] if not a.startswith('--')]
    use_env = '--no-env' not in sys.argv
    ckpt = os.path.abspath(args[0])
    run_name = os.path.basename(os.path.dirname(ckpt))
    tag = os.path.splitext(os.path.basename(ckpt))[0]
    out_dir = os.path.abspath(args[1]) if len(args) > 1 else os.path.join(HERE, 'export', f'{run_name}_{tag}')
    os.makedirs(out_dir, exist_ok=True)

    import torch
    from stable_baselines3 import PPO
    torch.set_num_threads(1)
    model = PPO.load(ckpt, device='cpu')
    pol = model.policy.eval()

    class Actor(torch.nn.Module):
        def __init__(self, p):
            super().__init__()
            self.extract = p.mlp_extractor.policy_net
            self.head = p.action_net

        def forward(self, obs):
            return torch.clamp(self.head(self.extract(obs)), -1.0, 1.0)

    actor = Actor(pol).eval()
    n_obs = pol.observation_space.shape[0]
    n_act = pol.action_space.shape[0]
    onnx_path = os.path.join(out_dir, 'ventricle_policy.onnx')
    torch.onnx.export(actor, torch.zeros(1, n_obs), onnx_path, input_names=['obs'],
                      output_names=['action'], dynamic_axes={'obs': {0: 'batch'}, 'action': {0: 'batch'}},
                      opset_version=OPSET, dynamo=False)
    import onnx
    onnx.checker.check_model(onnx.load(onnx_path))

    # ---------------------------------------------------------------- verify
    import onnxruntime as ort
    sess = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
    rng = np.random.default_rng(0)
    tests = {'random obs': rng.uniform(-2, 2, (256, n_obs)).astype(np.float32)}
    meta = {}
    if use_env:
        from ventricle_env import VentricleNavEnv
        import ventricle_env as ve
        env = VentricleNavEnv(max_steps=60)
        obs, _ = env.reset(seed=0)
        rec = [obs]
        for _ in range(59):
            a, _ = model.predict(obs, deterministic=True)
            obs, _, term, trunc, _ = env.step(a)
            rec.append(obs)
            if term or trunc:
                break
        tests['env rollout'] = np.array(rec, np.float32)
        meta = dict(
            case=os.environ.get('ROBOT_CASE'),
            entry_point_F_mm=env.F.tolist(),
            insertion_axis=(np.asarray(env.ctrl.rod['d'])).tolist(),
            goal_mm=env.goal.tolist(),
            planned_path_mm=env.path.tolist(),
            robot=dict(sections=[dict(L_mm=1e3 * s['L'], radius_mm=1e3 * s['radius'],
                                      chamber_offset_mm=1e3 * s['chamberOffset'],
                                      chamber_radius_mm=1e3 * s['chamberRadius'], n_chambers=s['nChambers'],
                                      yeoh_C_Pa=list(s['yeoh'])) for s in env.scene.SECTIONS]),
            kappa_fit_per_mm=(env.k_fit / 1e3).tolist(),
            constants=dict(DEPTH_STEP_MM=ve.DEPTH_STEP_MM, DKAPPA_per_mm=ve.DKAPPA / 1e3,
                           KAPPA_MAX_per_mm=ve.KAPPA_MAX / 1e3, GOAL_TOL_MM=ve.GOAL_TOL_MM,
                           N_SHAPE_POINTS=ve.N_SHAPE_POINTS, control_period_s=env.sim_steps * env.root.dt.value))
        env.close()
    for label, X in tests.items():
        ref = np.array([model.predict(x, deterministic=True)[0] for x in X])
        out = sess.run(['action'], {'obs': X})[0]
        print(f'{label:12s}: {len(X)} obs, max |onnx - sb3| = {np.abs(out - ref).max():.2e}')

    spec = dict(
        model_file='ventricle_policy.onnx', checkpoint=ckpt, opset=OPSET,
        input=dict(name='obs', shape=['batch', n_obs], dtype='float32', normalisation='none (raw)'),
        output=dict(name='action', shape=['batch', n_act], dtype='float32', range=[-1, 1],
                    deterministic='Gaussian mean, clipped'),
        frame=('All positions/directions are in the patient image frame: LPS, millimetres '
               '(the frame of the ventricle STL and planned path). Unity must transform its '
               'world points back into this frame before building obs, and transform '
               'commands/robot shape out of it.'),
        observation=[
            dict(index=[0, 3], name='tip - goal', scale='/ 50 mm'),
            dict(index=[3, 6], name='tip - planned path point at arc length = current depth', scale='/ 50 mm'),
            dict(index=[6, 9], name='tip tangent (unit vector)', scale='none'),
            dict(index=[9, 10], name='inserted depth / total soft length (80 mm)', scale='none'),
            dict(index=[10, 14], name='curvature commands [sec1 k_y, sec1 k_z, sec2 k_y, sec2 k_z]',
                 scale='/ KAPPA_MAX', note='body axes: x = tangent; k_y = -k sin(plane), k_z = k cos(plane)'),
            dict(index=[14, 32], name=f'robot centreline at 6 evenly spaced nodes (base..tip) minus F',
                 scale='/ 50 mm', note='nodes still inside the sheath lie on the insertion axis behind F'),
            dict(index=[32, 33], name='wall clearance at the tip (distance to wall - robot radius)', scale='/ 10 mm'),
            dict(index=[33, 34], name='minimum wall clearance along the inserted robot', scale='/ 10 mm'),
            dict(index=[34, 35], name='total wall contact force (sum of normal contact multipliers)',
                 scale='/ 0.05 N'),
        ],
        observation_clip=[-10, 10],
        action=[
            dict(index=0, name='insertion speed', meaning='depth += DEPTH_STEP_MM * (0.5 + 0.5 a), capped at 80 mm (no retraction)'),
            dict(index=[1, 3], name='section-1 curvature correction change', meaning='c1 = clip(c1 + a * DKAPPA, +-KAPPA_MAX)'),
            dict(index=[3, 5], name='section-2 curvature correction change', meaning='c2 = clip(c2 + a * DKAPPA, +-KAPPA_MAX)'),
            dict(index='all', name='commanded curvature', meaning='k_j = (depth / 80 mm) * kappa_fit_j + c_j, |k_j| <= KAPPA_MAX; kappa_fit in this file'),
        ],
        actuation=('A curvature command is a desired FREE bend (constant curvature per section). '
                   'The SOFA env turns it into chamber pressures with the Yeoh moment law '
                   '(cosserat_robot_01_from_matlab.pressure_from_bend); a kinematic Unity robot '
                   'can apply it directly as piecewise-constant curvature.'),
        **meta)
    with open(os.path.join(out_dir, 'ventricle_policy_spec.json'), 'w') as f:
        json.dump(spec, f, indent=1)
    print('wrote', out_dir, sorted(os.listdir(out_dir)))


if __name__ == '__main__':
    main()
