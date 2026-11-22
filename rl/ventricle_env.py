"""
Gymnasium environment: pneumatic 2-section soft robot navigating a patient's lateral
ventricle after Kocher's-point insertion, simulated in SOFA (cosserat_robot_01_from_matlab.py
in wall + RL mode).

Run with SOFA's own interpreter (rl/run.sh sets this up):
    python-sofa  (from the pixi env)  -- plain `python` crashes when importing Sofa on macOS,
    because the conda python links Python statically and SofaPython3 loads libpython again.
The scene module reads its configuration from environment variables at import time, so one
process holds one (case, wall) configuration; use subprocess vector envs for parallelism.

Action  a in [-1, 1]^5   (v3: RESIDUAL around a nominal schedule, forward-only insertion)
    a[0]      insertion speed: depth += DEPTH_STEP_MM * (0.5 + 0.5 a[0])   (0..1 mm per step)
    a[1:3]    change of the section-1 curvature CORRECTION (kappa_y, kappa_z) in body axes
    a[3:5]    change of the section-2 curvature correction
    commanded curvature = (depth / L) * kappa_fit + correction, |kappa| <= KAPPA_MAX, where
    kappa_fit are the bends fitted to this patient's path (the scripted schedule).  So the
    zero action reproduces the scripted insertion at 0.5 mm/step; the policy learns to steer
    it.  Curvatures become chamber pressures with the Yeoh moment law (pressureFromBend).
Observation (35, roughly unit scale)
    tip - goal, tip - path point at the current depth (both /50 mm), tip direction,
    depth / L, curvature commands / KAPPA_MAX, 6 robot points relative to F (/50 mm),
    wall clearance at the tip and minimum along the inserted robot (/10 mm),
    total wall contact force (/0.05 N)
Reward (per step)
    + W_DEPTH * inserted mm this step
    - W_TIP   * tip error to the path point at the current depth [mm]
    - w_shape * shape error: mean distance of the inserted robot nodes to the path point at
                the same arc length [mm] (constructor arg, default 0 = v3 reward)
    - W_FORCE * total wall contact (sum of normal contact multipliers)
    - W_TIME
    end of episode when fully inserted: + SUCCESS_BONUS * exp(-goal error / GOAL_SCALE_MM)
    leaving the lumen / non-finite state: - FAIL_PENALTY, terminate
    Completing the insertion always beats stalling (the scripted schedule, 37 mm off,
    still scores above an idle robot), so the "don't insert" optimum of v2 is gone; there is
    no retraction, so the insert/retract cycling of v1 is impossible.
"""
import importlib
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SCENES = os.path.dirname(HERE)
for p in (SCENES, os.path.join(SCENES, 'tools'), os.path.join(SCENES, '.pydeps')):
    if p not in sys.path:
        sys.path.append(p)

import gymnasium as gym  # noqa: E402
from gymnasium import spaces  # noqa: E402

DEPTH_STEP_MM = 1.0
DKAPPA = 0.01e3          # [1/m] curvature change per step at |a| = 1 (0.01 1/mm)
KAPPA_MAX = 0.12e3       # [1/m] ~1.4 x the largest curvature fitted to the 80 paths
W_DEPTH, W_TIP, W_FORCE, W_TIME = 0.2, 0.002, 20.0, 0.01
SUCCESS_BONUS, FAIL_PENALTY = 10.0, 10.0
GOAL_SCALE_MM = 10.0
GOAL_TOL_MM = 3.0
N_SHAPE_POINTS = 6


class VentricleNavEnv(gym.Env):
    metadata = {'render_modes': []}

    def __init__(self, case='design_patient44_right_r3_csf',
                 wall=os.path.join(SCENES, 'walls', 'patient44_right_entryhole.stl'),
                 sim_steps_per_action=10, max_steps=300, w_shape=0.0):
        self.w_shape = float(w_shape)
        os.environ.update(ROBOT_CASE=case, ROBOT_WALL=wall, ROBOT_RL='1')
        import Sofa.Core
        import Sofa.Simulation
        self.Sofa = Sofa
        self.scene = importlib.import_module('cosserat_robot_01_from_matlab')
        from ventricle_wall import read_stl, distance_to_mesh, inside
        self._dist, self._inside = distance_to_mesh, inside

        d = self.scene.DESIGN[case]
        self.wall_tri = read_stl(wall)                                          # mm
        full = os.path.join(os.path.dirname(os.path.dirname(d['ref_path_file'])), 'ventricles',
                            d['patient_path'] + '.stl')
        self.full_wall = read_stl(full) if os.path.exists(full) else self.wall_tri
        path = np.loadtxt(d['ref_path_file'], delimiter=',')                     # mm
        self.path_s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))]
        self.path = path
        self.sim_steps, self.max_steps = sim_steps_per_action, max_steps

        self.action_space = spaces.Box(-1.0, 1.0, (5,), np.float32)
        self.observation_space = spaces.Box(-10.0, 10.0, (35,), np.float32)
        self.root = None

    # ------------------------------------------------------------------ helpers
    def _path_at(self, s_mm):
        s_mm = np.clip(s_mm, 0, self.path_s[-1])
        return np.column_stack([np.interp(s_mm, self.path_s, self.path[:, k]) for k in range(3)])

    def _pressures(self):
        out = []
        for sec, k in zip(self.scene.SECTIONS, self.kappa):
            mag = np.linalg.norm(k)
            if mag < 1e-9:
                out.append(np.zeros(sec['nChambers']))
                continue
            plane = np.arctan2(-k[0], k[1])           # MATLAB: kappa_y = -k sin(phi), kappa_z = k cos(phi)
            out.append(self.scene.pressure_from_bend(sec, mag * sec['L'], plane))
        return out

    def _state(self):
        rod = self.ctrl.rod
        x = np.asarray(self.ctrl.dofs.position.value)
        L_mm = rod['s'][-1] * 1e3
        depth_mm = self.ctrl.depth * 1e3
        s_out = rod['s'] * 1e3 - (L_mm - depth_mm)          # arc length past F (<= 0: in sheath)
        free = s_out > 1e-6
        pts = x[:, :3] * 1e3
        tip = pts[-1]
        R = self.scene.quat_to_rotm(x[-1, 3:])
        st = dict(x=x, pts=pts, free=free, s_out=s_out, tip=tip, tangent=R[:, 0],
                  depth_mm=depth_mm, L_mm=L_mm)
        if free.any():
            dist = self._dist(pts[free], self.wall_tri)
            st['clear_tip'] = dist[-1] - self.r_mm
            st['clear_min'] = dist.min() - self.r_mm
            p_ref = self._path_at(s_out[free])
            st['shape_err'] = np.linalg.norm(pts[free] - p_ref, axis=1).mean()
            st['outside'] = int(np.sum(~self._inside(pts[free], self.full_wall)))
        else:
            st.update(clear_tip=10.0, clear_min=10.0, shape_err=0.0, outside=0)
        st['tip_err'] = np.linalg.norm(tip - self._path_at(np.array([max(depth_mm, 0.0)]))[0])
        st['goal_err'] = np.linalg.norm(tip - self.goal)
        lam = np.asarray(self.solver.constraintForces.value, float)
        st['force'] = float(np.abs(lam[0::3]).sum()) if lam.size else 0.0   # normal rows of friction contacts
        st['finite'] = bool(np.all(np.isfinite(x)))
        return st

    def _obs(self, st):
        x, free, pts = st['x'], st['free'], st['pts']
        F = self.F
        idx = np.linspace(0, len(pts) - 1, N_SHAPE_POINTS).astype(int)
        shape = (pts[idx] - F) / 50.0
        ref_now = self._path_at(np.array([max(st['depth_mm'], 0.0)]))[0]
        o = np.concatenate([(st['tip'] - self.goal) / 50.0, (st['tip'] - ref_now) / 50.0, st['tangent'],
                            [st['depth_mm'] / st['L_mm']], np.ravel(self.kappa) / KAPPA_MAX, shape.ravel(),
                            [st['clear_tip'] / 10.0, st['clear_min'] / 10.0, st['force'] / 0.05]])
        return np.clip(np.nan_to_num(o, nan=0.0, posinf=10.0, neginf=-10.0), -10, 10).astype(np.float32)

    # ------------------------------------------------------------------ gym API
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if self.root is not None:
            self.Sofa.Simulation.unload(self.root)
        self.root = self.Sofa.Core.Node('root')
        self.scene.createScene(self.root)
        self.Sofa.Simulation.initRoot(self.root)
        self.attach(self.root)
        for _ in range(2):
            self.Sofa.Simulation.animate(self.root, self.root.dt.value)
        st = self._state()
        return self._obs(st), self._info(st)

    def attach(self, root):
        """Bind to an initialised scene (created here by reset(), or by runSofa for
        rl/policy_scene.py) and set the command state of a fresh episode."""
        self.root = root
        self.ctrl = self.root.SoftRobot.pneumatics
        self.solver = self.root.constraint_solver
        self.F = np.asarray(self.ctrl.rod['F']) * 1e3
        self.r_mm = max(sec['radius'] for sec in self.scene.SECTIONS) * 1e3
        self.goal = self._path_at(np.array([self.ctrl.rod['s'][-1] * 1e3]))[0]
        self.kappa = np.zeros((2, 2))
        self.ctrl.cmd_depth = 0.0
        self.ctrl.cmd_pressures = self._pressures()
        self.n = 0
        self.sim_time = 0.0
        secs = self.scene.SECTIONS
        self.k_fit = np.array([[-s['bend'] / s['L'] * np.sin(s['plane']), s['bend'] / s['L'] * np.cos(s['plane'])]
                               for s in secs])                       # [1/m], body (k_y, k_z)
        self.residual = np.zeros((2, 2))

    def step(self, action):
        prev_depth = self.apply_action(action)

        t0 = time.perf_counter()
        for _ in range(self.sim_steps):
            self.Sofa.Simulation.animate(self.root, self.root.dt.value)
        self.sim_time += time.perf_counter() - t0
        self.n += 1

        st = self._state()
        ddepth_mm = (self.ctrl.cmd_depth - prev_depth) * 1e3
        r = (W_DEPTH * ddepth_mm - W_TIP * st['tip_err'] - self.w_shape * st['shape_err']
             - W_FORCE * st['force'] - W_TIME)
        terminated = False
        done = st['depth_mm'] >= st['L_mm'] - 1e-6
        success = done and st['goal_err'] <= GOAL_TOL_MM
        if not st['finite'] or st['outside'] > 0:
            r -= FAIL_PENALTY
            terminated = True
        elif done:
            r += SUCCESS_BONUS * np.exp(-st['goal_err'] / GOAL_SCALE_MM)
            terminated = True
        truncated = self.n >= self.max_steps
        info = self._info(st)
        info['success'] = bool(success)
        return self._obs(st), float(r), terminated, truncated, info

    def apply_action(self, action):
        """Turn a policy action into the insertion depth and chamber pressures the scene
        controller applies over the next sim_steps; returns the previous depth [m]."""
        a = np.clip(np.asarray(action, float), -1, 1)
        L = self.ctrl.rod['s'][-1]
        prev_depth = self.ctrl.cmd_depth
        self.ctrl.cmd_depth = float(min(prev_depth + (0.5 + 0.5 * a[0]) * DEPTH_STEP_MM * 1e-3, L))
        self.residual = np.clip(self.residual + a[1:].reshape(2, 2) * DKAPPA, -KAPPA_MAX, KAPPA_MAX)
        self.kappa = self.ctrl.cmd_depth / L * self.k_fit + self.residual
        for j in range(2):
            m = np.linalg.norm(self.kappa[j])
            if m > KAPPA_MAX:
                self.kappa[j] *= KAPPA_MAX / m
        self.ctrl.cmd_pressures = self._pressures()
        return prev_depth

    def _info(self, st):
        return dict(depth_mm=st['depth_mm'], tip_err_mm=st['tip_err'], goal_err_mm=st['goal_err'],
                    shape_err_mm=st['shape_err'], clear_min_mm=st['clear_min'], force=st['force'],
                    outside=st['outside'], max_pressure_kpa=max(float(np.max(p)) for p in self.ctrl.cmd_pressures) / 1e3)

    def close(self):
        if self.root is not None:
            self.Sofa.Simulation.unload(self.root)
            self.root = None
