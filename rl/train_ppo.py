"""
PPO on VentricleNavEnv (stable-baselines3), parallel SOFA environments in subprocesses.

    rl/run.sh rl/train_ppo.py --name ppo_p44_v1 --steps 200000 --envs 4

Writes rl/runs/<name>/: monitor CSVs (one row per episode, with success / depth / tip and
goal error), checkpoints every --ckpt steps, final model + VecNormalize statistics, and
progress.csv (SB3 logger).  Resume with --resume <checkpoint.zip>.
"""
import argparse
import glob
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

INFO_KEYS = ('success', 'depth_mm', 'tip_err_mm', 'goal_err_mm', 'shape_err_mm', 'force', 'outside')


def make_env(rank, run_dir, max_steps, w_shape=0.0):
    def _init():
        import torch
        torch.set_num_threads(1)
        from stable_baselines3.common.monitor import Monitor
        from ventricle_env import VentricleNavEnv
        env = VentricleNavEnv(max_steps=max_steps, w_shape=w_shape)
        return Monitor(env, os.path.join(run_dir, f'monitor_{rank}'), info_keywords=INFO_KEYS)
    return _init


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--name', default='ppo_p44_v1')
    ap.add_argument('--steps', type=int, default=200_000)
    ap.add_argument('--envs', type=int, default=4)
    ap.add_argument('--max-episode-steps', type=int, default=200)
    ap.add_argument('--n-steps', type=int, default=256, help='rollout length per env')
    ap.add_argument('--ckpt', type=int, default=10_000)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--resume', default='')
    ap.add_argument('--w-shape', type=float, default=0.0,
                    help='reward weight per step on the shape error [1/mm] (0 = v3 reward)')
    a = ap.parse_args()

    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import CheckpointCallback
    from stable_baselines3.common.logger import configure
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize

    torch.set_num_threads(1)
    run_dir = os.path.join(HERE, 'runs', a.name)
    os.makedirs(run_dir, exist_ok=True)
    fns = [make_env(i, run_dir, a.max_episode_steps, a.w_shape) for i in range(a.envs)]
    venv = SubprocVecEnv(fns, start_method='spawn') if a.envs > 1 else DummyVecEnv(fns)
    norm_path = os.path.join(run_dir, 'vecnormalize.pkl')
    if a.resume and os.path.exists(norm_path):
        venv = VecNormalize.load(norm_path, venv)
    else:
        # observations are already O(1); normalise rewards only
        venv = VecNormalize(venv, norm_obs=False, norm_reward=True, gamma=0.99)

    if a.resume:
        model = PPO.load(a.resume, env=venv, device='cpu')
    else:
        model = PPO('MlpPolicy', venv, n_steps=a.n_steps, batch_size=256, n_epochs=10,
                    learning_rate=3e-4, gamma=0.99, gae_lambda=0.95, clip_range=0.2,
                    ent_coef=0.0, policy_kwargs=dict(net_arch=[128, 128]),
                    seed=a.seed, device='cpu', verbose=1)
    model.set_logger(configure(run_dir, ['stdout', 'csv']))

    class SaveNorm(CheckpointCallback):
        def _on_step(self):
            ok = super()._on_step()
            if self.n_calls % self.save_freq == 0:
                venv.save(norm_path)
            return ok

    ckpt = SaveNorm(save_freq=max(a.ckpt // a.envs, 1), save_path=run_dir, name_prefix='ckpt')
    model.learn(total_timesteps=a.steps, callback=ckpt, reset_num_timesteps=not a.resume,
                progress_bar=False)
    model.save(os.path.join(run_dir, 'final_model'))
    venv.save(norm_path)
    venv.close()
    print('done:', run_dir, sorted(os.path.basename(p) for p in glob.glob(os.path.join(run_dir, '*.zip'))))


if __name__ == '__main__':
    main()
