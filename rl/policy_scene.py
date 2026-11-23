"""
runSofa scene: the patient-44 insertion driven live by a trained PPO policy (the same scene,
observation and action mapping as rl/ventricle_env.py, so it reproduces rl/rollout.py
--deterministic step for step).

    ROBOT_POLICY=rl/runs/<run>/final_model.zip  runSofa -l SofaPython3 -g imgui rl/policy_scene.py
    (needs torch + stable-baselines3 on PYTHONPATH, e.g. thesis-scenes/.pydeps)

Every 10 simulation steps (one RL step, 0.1 s) the controller builds the 35-value observation,
asks the policy for an action (deterministic) and sets the insertion depth and chamber
pressures for the next 10 steps.  It stops commanding once the robot is fully inserted
(80 mm); press the GUI's reset to replay.  Progress is printed to the log every RL step.

Env vars: ROBOT_POLICY (default: v2 shape-reward run), ROBOT_CASE, ROBOT_WALL (defaults:
patient 44 right, the training wall), ROBOT_POLICY_STOCHASTIC=1 to sample actions.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SCENES = os.path.dirname(HERE)
for p in (HERE, SCENES, os.path.join(SCENES, '.pydeps')):
    if p not in sys.path:
        sys.path.insert(0, p)
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

import numpy as np  # noqa: E402
import Sofa  # noqa: E402
import Sofa.Core  # noqa: E402

from ventricle_env import VentricleNavEnv  # noqa: E402

_V2 = os.path.join('ppo_p44_server_v2_shape', 'final_model.zip')
_CANDIDATES = [os.path.join(HERE, 'runs', _V2),                         # trained locally
               os.path.join(SCENES, 'results', _V2),                    # SoftRobot-Ventricle-PPO/sofa_rl
               os.path.join(SCENES, 'submission', 'results', _V2)]      # thesis-scenes
POLICY = os.environ.get('ROBOT_POLICY') or next((p for p in _CANDIDATES if os.path.exists(p)), _CANDIDATES[0])
CASE = os.environ.get('ROBOT_CASE', 'design_patient44_right_r3_csf')
WALL = os.environ.get('ROBOT_WALL', os.path.join(SCENES, 'walls', 'patient44_right_entryhole.stl'))
DETERMINISTIC = os.environ.get('ROBOT_POLICY_STOCHASTIC', '0') != '1'
WARMUP_STEPS = 2          # VentricleNavEnv.reset() animates twice before the first observation


class PolicyDriver(Sofa.Core.Controller):
    def __init__(self, env, model, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.env, self.model = env, model
        self.k = 0                 # simulation steps since load / reset
        self.done = False

    def onAnimateBeginEvent(self, event):
        env = self.env
        if self.k == 0:
            env.attach(self.getContext().getRoot())
            self.done = False
            Sofa.msg_info('PolicyDriver', f'policy {os.path.basename(os.path.dirname(POLICY))} '
                          f'({"deterministic" if DETERMINISTIC else "stochastic"}), goal {np.round(env.goal, 1)} mm')
        k, self.k = self.k, self.k + 1
        if self.done or k < WARMUP_STEPS or (k - WARMUP_STEPS) % env.sim_steps:
            return
        st = env._state()
        if env.n > 0:
            Sofa.msg_info('PolicyDriver', f'step {env.n:3d}: inserted {st["depth_mm"]:5.1f} mm, '
                          f'tip -> goal {st["goal_err"]:5.1f} mm, tip -> path {st["tip_err"]:4.1f} mm, '
                          f'rod -> path {st["shape_err"]:4.1f} mm')
        if st['depth_mm'] >= st['L_mm'] - 1e-6 or not st['finite'] or st['outside'] > 0:
            self.done = True
            ok = st['goal_err'] <= 3.0 and st['outside'] == 0
            Sofa.msg_info('PolicyDriver', f'finished after {env.n} RL steps: goal error {st["goal_err"]:.2f} mm '
                          f'({"success" if ok else "no success"}), inserted rod {st["shape_err"]:.2f} mm from the path')
            return
        action, _ = self.model.predict(env._obs(st), deterministic=DETERMINISTIC)
        env.apply_action(action)
        env.n += 1

    def onSimulationInitDoneEvent(self, event):
        self.k = 0


def keep_libraries_on_reload():
    """runSofa's reload calls SofaRuntime.unloadModules(), which drops every module imported
    since start-up; torch cannot be imported a second time in one process ("already has a
    docstring"). Exclude everything except this project's scene code from that unloading."""
    try:
        import SofaRuntime
    except ImportError:
        return
    keep = getattr(SofaRuntime, '__SofaPythonEnvironment_modulesExcludedFromReload', None)
    if keep is None:
        return
    pydeps = os.path.join(SCENES, '.pydeps')
    for name, mod in list(sys.modules.items()):
        f = getattr(mod, '__file__', None) or ''
        if f.startswith(SCENES + os.sep) and not f.startswith(pydeps + os.sep):
            continue                                     # our own modules: reload them
        if name not in keep:
            keep.append(name)


def createScene(root):
    try:
        import torch
        torch.set_num_threads(1)
        from stable_baselines3 import PPO
        if not os.path.exists(POLICY):
            raise FileNotFoundError(f'policy not found: {POLICY} (set ROBOT_POLICY=<path to final_model.zip>)')

        env = VentricleNavEnv(case=CASE, wall=WALL)     # sets ROBOT_RL=1 etc. and imports the scene module
        model = PPO.load(POLICY, device='cpu')
        # added before the robot so that, within each step, it updates the commands before the
        # scene's pneumatics controller applies them (as VentricleNavEnv.step does)
        root.addObject(PolicyDriver(env, model, name='policy'))
        env.scene.createScene(root)
    finally:
        keep_libraries_on_reload()      # also after a failed load, so torch survives the next open/reload
    return root
