"""
Roll out a PPO checkpoint (or the fitted schedule with zero residual) on VentricleNavEnv and
save tip trajectories + final rod shapes for plotting.

    rl/run.sh rl/rollout.py --ckpt rl/runs/ppo_p44_v3/ckpt_10000_steps.zip --episodes 6 --out traj.npz
    rl/run.sh rl/rollout.py --baseline --out baseline.npz     # full-speed insertion, no residual

--record-rod also stores every robot node at every step (rods_<e>: [steps+1, nodes, 3] mm,
including the nodes still in the sheath behind F) and the commanded depth (depth_<e>), for
replaying the insertion (e.g. rl/export_unity_p44.py).
"""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default='')
    ap.add_argument('--baseline', action='store_true')
    ap.add_argument('--episodes', type=int, default=6)
    ap.add_argument('--deterministic', action='store_true')
    ap.add_argument('--max-steps', type=int, default=250)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--record-rod', action='store_true')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()

    from ventricle_env import VentricleNavEnv
    env = VentricleNavEnv(max_steps=a.max_steps)
    model = None
    if not a.baseline:
        import torch
        torch.set_num_threads(1)
        from stable_baselines3 import PPO
        model = PPO.load(a.ckpt, device='cpu')

    rng = np.random.default_rng(a.seed)
    out = dict(path=env.path, wall=env.wall_tri.reshape(-1, 3)[::3])
    n_eps = 1 if a.baseline or a.deterministic else a.episodes
    for e in range(n_eps):
        obs, info = env.reset(seed=int(rng.integers(1 << 30)))
        tips, done = [], False
        st = env._state()
        rods, depths = [st['pts'].copy()], [st['depth_mm']]
        while not done:
            if model is None:
                act = np.array([1.0, 0, 0, 0, 0], np.float32)
            else:
                act, _ = model.predict(obs, deterministic=a.deterministic)
            obs, r, term, trunc, info = env.step(act)
            st = env._state()
            tips.append(st['tip'].copy())
            rods.append(st['pts'].copy())
            depths.append(st['depth_mm'])
            done = term or trunc
        out[f'tip_{e}'] = np.array(tips)
        out[f'rod_{e}'] = st['pts'][st['free']]
        if a.record_rod:
            out[f'rods_{e}'] = np.array(rods)
            out[f'depth_{e}'] = np.array(depths)
        out[f'goal_err_{e}'] = info['goal_err_mm']
        out[f'success_{e}'] = info['success']
        print(f'ep {e}: goal_err {info["goal_err_mm"]:.1f} mm, depth {info["depth_mm"]:.1f}, '
              f'success {info["success"]}', flush=True)
    out['goal'] = env.goal
    out['F'] = env.F
    out['r_mm'] = env.r_mm
    out['d'] = np.asarray(env.ctrl.rod['d'], float)          # sheath axis (unit), towards F
    out['L_rigid_mm'] = float(env.scene.ROBOT['Lrigid']) * 1e3
    out['path_s'] = env.path_s
    out['n'] = n_eps
    np.savez(a.out, **out)
    env.close()


if __name__ == '__main__':
    main()
