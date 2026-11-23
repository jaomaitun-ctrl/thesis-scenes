"""
Smoke test of VentricleNavEnv: gymnasium's env checker, then one episode each of
  random   - uniform random actions
  scripted - insert at full speed while ramping the curvatures towards the bends fitted
             to this patient's path (the same schedule as the scripted wall test)
and the wall-clock cost per step.

    rl/run.sh rl/check_env.py [max_steps]
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ventricle_env import VentricleNavEnv  # noqa: E402

MAX_STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 250


def run(env, policy, label):
    obs, info = env.reset(seed=0)
    total, t0, n = 0.0, time.perf_counter(), 0
    for n in range(1, MAX_STEPS + 1):
        obs, r, term, trunc, info = env.step(policy(obs, info))
        total += r
        if term or trunc:
            break
    dt = time.perf_counter() - t0
    print(f'[{label:8s}] {n} steps, return {total:8.2f} | depth {info["depth_mm"]:5.1f} mm, tip err '
          f'{info["tip_err_mm"]:5.1f} mm, shape err {info["shape_err_mm"]:5.1f} mm, goal err '
          f'{info["goal_err_mm"]:5.1f} mm, min clearance {info["clear_min_mm"]:+.2f} mm, force '
          f'{info["force"]:.4f}, outside {info["outside"]}, success {info.get("success", False)} | '
          f'{1e3 * dt / n:.0f} ms/step (SOFA {1e3 * env.sim_time / n:.0f})', flush=True)


def main():
    from gymnasium.utils.env_checker import check_env
    env = VentricleNavEnv(max_steps=MAX_STEPS)
    check_env(env, skip_render_check=True)
    print('gymnasium check_env: ok', flush=True)

    rng = np.random.default_rng(0)
    run(env, lambda o, i: rng.uniform(-1, 1, 5), 'random')

    # nominal: zero action = scripted schedule (fitted bends ramped with depth) at 0.5 mm/step
    run(env, lambda o, i: np.zeros(5), 'nominal')
    # full speed, no steering
    run(env, lambda o, i: np.r_[1.0, np.zeros(4)], 'fast')
    # idle: never insert
    run(env, lambda o, i: np.r_[-1.0, np.zeros(4)], 'idle')
    env.close()


if __name__ == '__main__':
    main()
