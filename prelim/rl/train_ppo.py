"""
PPO on VentricleNavEnv (stable-baselines3), parallel SOFA environments in subprocesses.

    rl/run.sh rl/train_ppo.py --name ppo_p44_v1 --steps 200000 --envs 4

Writes rl/runs/<name>/: monitor CSVs (one row per episode, with success / depth / tip and
goal error), checkpoints every --ckpt steps, final model + VecNormalize statistics, and
progress.csv (SB3 logger).  Resume with --resume <checkpoint.zip>.

Memory: every episode reset rebuilds the SOFA scene and the env worker processes leak
(~30 MB/min/env on the Linux server), which OOM-killed a 200k run.  Training therefore runs
one PPO rollout at a time and, between rollouts, checks the workers' RSS and the machine's
available memory.  Past --max-worker-mb (per worker) or below --min-free-mb, the workers are
closed and respawned; the policy, optimiser, VecNormalize statistics and step count stay in
the main process, so no steps are lost.  Recycles are logged to recycles.csv.
Episodes in progress at a recycle are cut off (not logged).  For crashes that still happen,
rl/train_supervised.sh restarts from the newest checkpoint.
"""
import argparse
import glob
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

INFO_KEYS = ('success', 'depth_mm', 'tip_err_mm', 'goal_err_mm', 'shape_err_mm', 'force', 'outside')


def rss_mb(pids):
    """Resident memory of each pid [MB] (Linux /proc, otherwise ps)."""
    out = []
    for pid in pids:
        try:
            with open(f'/proc/{pid}/status') as f:
                out.append(next(int(l.split()[1]) for l in f if l.startswith('VmRSS')) / 1024)
        except OSError:
            try:
                out.append(int(subprocess.check_output(['ps', '-o', 'rss=', '-p', str(pid)])) / 1024)
            except (subprocess.CalledProcessError, ValueError):
                out.append(0.0)
    return out


def available_mb():
    """MemAvailable [MB] on Linux; None elsewhere (the per-worker limit still applies)."""
    try:
        with open('/proc/meminfo') as f:
            return next(int(l.split()[1]) for l in f if l.startswith('MemAvailable')) / 1024
    except (OSError, StopIteration):
        return None


def make_env(rank, run_dir, max_steps, w_shape=0.0, append=False):
    def _init():
        import torch
        torch.set_num_threads(1)
        from stable_baselines3.common.monitor import Monitor
        from ventricle_env import VentricleNavEnv
        env = VentricleNavEnv(max_steps=max_steps, w_shape=w_shape)
        # append on --resume so a run trained in chunks keeps one episode history per env
        return Monitor(env, os.path.join(run_dir, f'monitor_{rank}'), info_keywords=INFO_KEYS,
                       override_existing=not append)
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
    ap.add_argument('--total-steps', type=int, default=0,
                    help='train until this many steps in total (overrides --steps; used by '
                         'train_supervised.sh to resume from any checkpoint)')
    ap.add_argument('--max-worker-mb', type=float, default=1500,
                    help='respawn the env workers when any one exceeds this RSS')
    ap.add_argument('--min-free-mb', type=float, default=1500,
                    help='respawn the env workers when available memory drops below this')
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
    def new_workers(append):
        fns = [make_env(i, run_dir, a.max_episode_steps, a.w_shape, append=append) for i in range(a.envs)]
        return SubprocVecEnv(fns, start_method='spawn') if a.envs > 1 else DummyVecEnv(fns)

    venv = new_workers(append=bool(a.resume))
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
    prog = os.path.join(run_dir, 'progress.csv')
    if a.resume and os.path.exists(prog):              # SB3's CSV logger would overwrite it
        os.rename(prog, os.path.join(run_dir, f'progress_upto_{model.num_timesteps}.csv'))
    model.set_logger(configure(run_dir, ['stdout', 'csv']))

    class SaveNorm(CheckpointCallback):
        def _on_step(self):
            ok = super()._on_step()
            if self.n_calls % self.save_freq == 0:
                venv.save(norm_path)
            return ok

    ckpt = SaveNorm(save_freq=max(a.ckpt // a.envs, 1), save_path=run_dir, name_prefix='ckpt')
    if a.total_steps:
        target = a.total_steps
    else:
        target = model.num_timesteps + a.steps if a.resume else a.steps
    rollout = a.n_steps * a.envs
    rec_log = os.path.join(run_dir, 'recycles.csv')
    if not os.path.exists(rec_log):
        with open(rec_log, 'w') as f:
            f.write('time,steps,max_worker_mb,sum_worker_mb,available_mb,reason\n')
    print(f'training {model.num_timesteps} -> {target} steps, {rollout} per rollout', flush=True)
    while model.num_timesteps < target:
        model.learn(total_timesteps=min(rollout, target - model.num_timesteps), callback=ckpt,
                    reset_num_timesteps=False, progress_bar=False)
        if not isinstance(venv.venv, SubprocVecEnv) or model.num_timesteps >= target:
            continue
        mem = rss_mb([p.pid for p in venv.venv.processes])
        free = available_mb()
        reason = ('worker' if max(mem) > a.max_worker_mb else
                  'free' if free is not None and free < a.min_free_mb else '')
        if reason:
            with open(rec_log, 'a') as f:
                f.write(f'{time.strftime("%Y-%m-%d %H:%M:%S")},{model.num_timesteps},{max(mem):.0f},'
                        f'{sum(mem):.0f},{free if free is None else round(free)},{reason}\n')
            print(f'recycling env workers at {model.num_timesteps} steps ({reason}: max worker '
                  f'{max(mem):.0f} MB, available {free} MB)', flush=True)
            venv.venv.close()
            venv.venv = new_workers(append=True)
            model._last_obs = None                     # next learn() resets the new workers
    model.save(os.path.join(run_dir, 'final_model'))
    venv.save(norm_path)
    venv.close()
    print('done:', run_dir, sorted(os.path.basename(p) for p in glob.glob(os.path.join(run_dir, '*.zip'))))


if __name__ == '__main__':
    main()
