"""
Soft continuum robot imported from MATLAB (cosserat_yeoh_quasistatic.m) into SOFA.

Robot (same numbers as the MATLAB script):
    rigid insertion 80 mm  ->  soft section 1 (120 mm, r 6.0 mm, 3 chambers)
                           ->  soft section 2 (100 mm, r 5.5 mm, 3 chambers)
    base frame: body x (rod tangent) -> world +Z, base at the origin (MATLAB default),
    or on the planned Kocher's-point trajectory (ROBOT_CASE=kocher_*).

How each MATLAB ingredient maps to SOFA
---------------------------------------
MATLAB (Cosserat ODE + shooting)       SOFA (this scene)
- rod centreline r(s), q(s)            MechanicalObject<Rigid3d>, one frame every 5 mm
- linear twist/shear, Yeoh bending     BeamFEMForceField (corotational Timoshenko beam),
                                       E0 = 6*C10, G0 = 2*C10 (nu = 0.5).  Linear
                                       bending: Yeoh only differs at tens of % fibre
                                       strain; this robot reaches ~3 %.
- actuation wrench F_a = sum d x t tau One FOLLOWER couple +-R_mid*M per beam element,
  (internal, constant along section)   R_mid = element mid-frame, updated every step.
                                       (= end moments + distributed -(R M)'; end
                                       moments alone leak into the next section.)
- pressureFromBend(theta, phi)         same function, ported to numpy (Yeoh M(k) table)
- gravity rho*A*g, tip force           ConstantForceField (weight lumped per node)
- quasi-static, p(t) smooth-step ramp  same ramp (loads ramped too); implicit Euler +
                                       Rayleigh damping run to rest (default), or
                                       StaticSolver/Newton (ROBOT_SOLVER=static, fragile
                                       under gravity: the beam tangent lacks geometric
                                       stiffness)
- rigid insertion                      fixed first frame + display cylinder

Verified against the MATLAB solver (matlab_refs/<case>.csv, 600 steps, Yeoh-equivalent
actuation on):
    actuation_only      tip error 0.04 mm, centreline RMS 0.02 mm
    actuation_tipforce  tip error 1.46 mm, centreline RMS 0.53 mm
    hanging             tip error 0.94 mm, centreline RMS 0.58 mm
    matlab_default      DOES NOT MATCH, and should not: the robot points up and its
                        own weight (~1.1 N/m) is ~4x the self-weight buckling load of
                        the column (7.84*EI/L^3 ~ 0.26 N/m).  MATLAB's shooting method
                        returns the upright, UNSTABLE equilibrium; SOFA falls to the
                        stable one (tip z ~ -50 mm).
    kocher_head0_csf    tip error 0.34 mm, centreline RMS 0.17 mm
    kocher_head0_air    tip error 1.85 mm, centreline RMS 1.07 mm
    kocher_head30_csf   tip error 3.36 mm, centreline RMS 1.46 mm.  Mostly lateral,
                        i.e. twist.  About a third comes from a term this scene drops:
                        MATLAB tilts the chamber direction with twist (t = kappa x d +
                        sigma); with t = x in MATLAB too the error is 2.12 mm.  The
                        rest (Yeoh vs linear, inextensible vs not, 5 mm elements) is
                        not separated yet.
    design_patient44_right_{nogravity,csf}
                        RESIZED robot (25 + 55 mm, fitted to the 80 patient paths)
                        bent 77 deg / 88 deg: tip error 0.07 mm, RMS 0.05 mm.  Needs
                        the Yeoh-equivalent actuation: at 32 % fibre strain the linear
                        beam alone over-bends section 1 by ~9 deg (tip error 7.6 mm).
    design_patient44_right_r3_{nogravity,csf}
                        same, radius 3 / 2.75 mm (OD 6 mm, tools/size_radius.py): tip
                        error 0.07 / 0.09 mm.  Same shape at half the pressure (max
                        237 vs 514 kPa, placeholder C10) and half the strain (16 %).
References: matlab/run_cases.m (solver generated from cosserat_yeoh_quasistatic.m
by tools/make_matlab_case.py; Kocher pose from tools/kocher_pose.py).

Run (GUI)  -- from the SOFA checkout that has the pixi env:
    cd ~/Developer/sofa
    ROBOT_CASE=kocher_head0_csf pixi run -e supported-plugins \
        .pixi/envs/supported-plugins/bin/runSofa -l SofaPython3 \
        ~/sofa/thesis-scenes/cosserat_robot_01_from_matlab.py
Run (headless; writes matlab_refs/<case>_sofa.csv and prints the comparison):
    add  -g batch -n 610  to the command above.
Ventricle wall + insertion (design_* cases; wall from tools/ventricle_wall.py):
    ROBOT_WALL=walls/<patient>_entryhole.stl ROBOT_CASE=design_<patient>_r3_csf ... -n 1005
    patient44_right, r = 3 mm, 10 mm/s, final bends ramped with depth: 80 mm inserted in
    41 s wall time; stays in the lumen, no penetration of the wall; settles 14.6 mm (mean)
    from the planned path - the scripted schedule is not a navigation policy.
    Contact solver tolerance (ROBOT_LCP_TOL, default 1e-6): 1e-6..1e-8 agree (37-38 mm tip
    error); 1e-5 with 50 iterations is under-converged and lands in another branch of the
    ventricle (5 mm) - the insertion has a bifurcation that small differences can flip.
RL: ROBOT_RL=1 makes the controller apply cmd_depth / cmd_pressures (rl/ventricle_env.py).
"""
import json
import os
import sys
import numpy as np
import Sofa

HERE = os.path.dirname(os.path.abspath(__file__))

# Load case (select with:  ROBOT_CASE=actuation_only runSofa ...).  Each has a
# MATLAB reference in matlab_refs/<case>.csv produced by the unmodified solver.
#   matlab_default      gravity -Z, tip force   (robot points UP: weight compresses it)
#   actuation_only      no gravity, no tip force
#   actuation_tipforce  no gravity, tip force
#   hanging             gravity +Z (weight puts the rod in tension), tip force
#   kocher_*            robot on the planned Kocher's-point trajectory in the patient3
#                       mesh frame (poses/kocher_patient3.json), patient supine with the
#                       head of bed at 0 or 30 deg; '_csf' = weight minus CSF buoyancy
#                       (net ~4 % of the dry weight), '_air' = full weight.
POSE = json.load(open(os.path.join(HERE, 'poses', 'kocher_patient3.json')))
_TIP = [0.005, 0, -0.010]
CASES = {   # name: (gravity [m/s^2], tip force [N], pose)
    'matlab_default':     ([0, 0, -9.81], _TIP,      'matlab'),
    'actuation_only':     ([0, 0, 0],     [0, 0, 0], 'matlab'),
    'actuation_tipforce': ([0, 0, 0],     _TIP,      'matlab'),
    'hanging':            ([0, 0, 9.81],  _TIP,      'matlab'),
    'kocher_head0_air':   (POSE['gravity']['supine_head0_air'],  [0, 0, 0], 'kocher'),
    'kocher_head0_csf':   (POSE['gravity']['supine_head0_csf'],  [0, 0, 0], 'kocher'),
    'kocher_head30_csf':  (POSE['gravity']['supine_head30_csf'], [0, 0, 0], 'kocher'),
}
#   design_*            robot RESIZED to fit the patient paths (tools/fit_robot_design.py),
#                       placed on one patient's Kocher->F axis with that patient's fitted
#                       bends; everything comes from poses/design_*.json (made by
#                       tools/make_design_cases.py), which the MATLAB runner reads too.
DESIGN = {}
for _f in sorted(os.listdir(os.path.join(HERE, 'poses'))):
    if _f.startswith('design_') and _f.endswith('.json'):
        DESIGN[_f[:-5]] = json.load(open(os.path.join(HERE, 'poses', _f)))
        if not os.path.isabs(DESIGN[_f[:-5]].get('ref_path_file', '/')):   # repo-relative (data/)
            DESIGN[_f[:-5]]['ref_path_file'] = os.path.join(HERE, DESIGN[_f[:-5]]['ref_path_file'])
        CASES[_f[:-5]] = (DESIGN[_f[:-5]]['gravity'], [0, 0, 0], 'design')
CASE = os.environ.get('ROBOT_CASE', 'kocher_head0_csf')
MATLAB_CENTERLINE = os.path.join(HERE, 'matlab_refs', f'{CASE}.csv')
SOFA_CENTERLINE = os.path.join(HERE, 'matlab_refs', f'{CASE}_sofa.csv')

# ---------------------------------------------------------------- parameters
ROBOT = dict(Lrigid=0.080, rigidRadius=4.5e-3)
# name, L, radius, [C10 C20 C30], rho, chamberOffset, chamberRadius, nChambers, elementLength
SECTIONS = [
    dict(L=0.120, radius=6.0e-3, yeoh=[58.3e3, 5e3, 0.5e3], rho=1050,
         chamberOffset=3.5e-3, chamberRadius=2.0e-3, nChambers=3,
         bend=np.deg2rad(35), plane=np.deg2rad(0), color=[0.10, 0.55, 0.95, 1.0]),
    dict(L=0.100, radius=5.5e-3, yeoh=[41.7e3, 4e3, 0.4e3], rho=1050,
         chamberOffset=3.2e-3, chamberRadius=1.8e-3, nChambers=3,
         bend=np.deg2rad(50), plane=np.deg2rad(70), color=[1.00, 0.42, 0.16, 1.0]),
]
ELEMENT_LENGTH = 0.005                       # [m] beam element length
GRAVITY = np.array(CASES[CASE][0], float)     # world [m/s^2]
TIP_FORCE = np.array(CASES[CASE][1], float)   # world [N]
TIP_MOMENT = np.array([0.0, 0.0, 0.0])       # world [N*m]
T_RAMP = 1.0                                 # [s] smooth-step pressure (and load) ramp
DT = 0.01                                    # [s] time step
# 'dynamic': implicit Euler + Rayleigh damping, run until it settles (robust; the
#            equilibrium reached is the STABLE one).  'static': Newton per step.
SOLVER = os.environ.get('ROBOT_SOLVER', 'dynamic')
# Map chamber moments through the Yeoh law onto the linear beam (see actuation_moment_body)
YEOH_EQUIVALENT_ACTUATION = os.environ.get('ROBOT_YEOH_ACTUATION', '1') == '1'

# Ventricle wall + insertion (design_* cases only): ROBOT_WALL=<wall stl, mm, with entry
# hole from tools/ventricle_wall.py>.  The robot is pushed out of a rigid sheath that
# ends at the entry point F (soft_base_m): nodes still inside the sheath are held on the
# insertion axis, the rest are free and touch the wall through sphere contacts
# (Lagrangian, friction MU).  Pressures follow a scripted schedule (final bends ramped
# with insertion depth) - a smoke test of the environment, not a navigation policy.
WALL_STL = os.environ.get('ROBOT_WALL', '')
INSERT_SPEED = float(os.environ.get('ROBOT_INSERT_SPEED', 0.010))   # [m/s]
MU = 0.1                                                            # wall friction
# RL mode (rl/ventricle_env.py): the controller applies cmd_depth / cmd_pressures set by the
# environment instead of the scripted schedule
RL_MODE = os.environ.get('ROBOT_RL', '0') == '1'
EXPORT_AT_STEP = int(os.environ.get('ROBOT_EXPORT_STEP', 1000 if WALL_STL else 600))  # CSV + compare
if WALL_STL:
    SOFA_CENTERLINE = os.path.join(HERE, 'matlab_refs', f'{CASE}_wall_sofa.csv')

if CASES[CASE][2] == 'kocher':
    BASE_POS = np.array(POSE['base_position_m'])          # top of the rigid insertion
    BASE_QUAT = np.array(POSE['base_quat_xyzw'])          # body x -> insertion axis
elif CASES[CASE][2] == 'design':
    _d = DESIGN[CASE]
    BASE_POS = np.array(_d['base_position_m'])
    BASE_QUAT = np.array(_d['base_quat_xyzw'])
    for _sec, _s in zip(SECTIONS, _d['sections']):
        _sec.update(L=_s['L_m'], bend=_s['bend_rad'], plane=_s['plane_rad'])
        if 'radius_m' in _s:                                  # resized cross-section
            _sec.update(radius=_s['radius_m'], chamberOffset=_s['chamberOffset_m'],
                        chamberRadius=_s['chamberRadius_m'])
            ROBOT['rigidRadius'] = 0.75 * _d['sections'][0]['radius_m']
    # short, strongly bent sections: 2 mm elements (>= 10 per section)
    ELEMENT_LENGTH = min(0.002, min(s['L_m'] for s in _d['sections']) / 10)
else:
    # MATLAB baseQuaternion [w x y z] = [cos(-pi/4) 0 sin(-pi/4) 0]  ->  SOFA [x y z w]
    BASE_POS = np.zeros(3)
    BASE_QUAT = np.array([0.0, np.sin(-np.pi / 4), 0.0, np.cos(-np.pi / 4)])


# ------------------------------------------------- MATLAB functions, ported
def yeoh_stress(lam, C):
    I1 = lam ** 2 + 2.0 / lam
    dW = C[0] + 2 * C[1] * (I1 - 3) + 3 * C[2] * (I1 - 3) ** 2
    return (2 * lam - 2.0 / lam ** 2) * dW


def make_section(sec):
    """makeSection(): geometry, chamber layout, Yeoh M(k) table."""
    r = sec['radius']
    sec['A'] = np.pi * r ** 2
    sec['G0'] = 2 * sec['yeoh'][0]
    sec['E0'] = 3 * sec['G0']
    alpha = 2 * np.pi * np.arange(sec['nChambers']) / sec['nChambers']
    sec['d'] = np.vstack([np.zeros_like(alpha),
                          sec['chamberOffset'] * np.cos(alpha),
                          sec['chamberOffset'] * np.sin(alpha)])
    sec['Ain'] = np.pi * sec['chamberRadius'] ** 2
    nz = 600
    dz = 2 * r / nz
    z = -r + dz * (np.arange(1, nz + 1) - 0.5)
    w = 2 * np.sqrt(r ** 2 - z ** 2) * dz
    sec['kTab'] = np.linspace(0, 0.9 / r, 1500)
    sec['MTab'] = np.array([np.sum(yeoh_stress(1 + k * z, sec['yeoh']) * z * w)
                            for k in sec['kTab']])
    return sec


def pressure_from_bend(sec, bend_angle, bending_plane):
    """pressureFromBend(): chamber gauge pressures [Pa] for a requested bend."""
    k = bend_angle / sec['L']
    Mk = np.interp(k, sec['kTab'], sec['MTab'])
    m_target = Mk * np.array([-np.sin(bending_plane), np.cos(bending_plane)])
    M = np.vstack([sec['d'][2, :], -sec['d'][1, :]])
    tau = np.linalg.pinv(M) @ m_target
    tau = tau - tau.min()
    return tau / sec['Ain']


def actuation_moment_body(sec, p):
    """Moment part of actuationWrench() with t = body x: sum_i (d_i x x) tau_i.

    With YEOH_EQUIVALENT_ACTUATION the moment is mapped through the Yeoh bending law
    before it reaches the LINEAR beam: M_lin = E0*I * k_Yeoh(|M|), so free (actuation-only)
    bending gives exactly the Yeoh curvature.  Response to external loads stays linear.
    Matters above ~10 % fibre strain (11 % stiffer at 32 %); negligible below.
    """
    tau = p * sec['Ain']
    d = sec['d']
    m = np.array([0.0, np.sum(d[2] * tau), -np.sum(d[1] * tau)])
    mag = np.linalg.norm(m)
    if YEOH_EQUIVALENT_ACTUATION and mag > 0:
        k = np.interp(mag, sec['MTab'], sec['kTab'])
        m *= sec['E0'] * np.pi * sec['radius'] ** 4 / 4 * k / mag
    return m


def quat_to_rotm(q):
    """SOFA quaternion [x y z w] -> rotation matrix."""
    x, y, z, w = q / np.linalg.norm(q)
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def smooth_step(u):
    u = min(max(u, 0.0), 1.0)
    return 3 * u ** 2 - 2 * u ** 3


# ------------------------------------------------------------ discretisation
def build_rod():
    for sec in SECTIONS:
        make_section(sec)
        sec['pTarget'] = pressure_from_bend(sec, sec['bend'], sec['plane'])
    R0 = quat_to_rotm(BASE_QUAT)
    tangent = R0 @ np.array([1.0, 0.0, 0.0])
    soft_base = BASE_POS + ROBOT['Lrigid'] * tangent
    s, section_of_edge, node_ranges = [0.0], [], []
    for j, sec in enumerate(SECTIONS):
        n = int(round(sec['L'] / ELEMENT_LENGTH))
        first = len(s) - 1
        s += list(s[-1] + np.linspace(0, sec['L'], n + 1)[1:])
        section_of_edge += [j] * n
        node_ranges.append((first, len(s) - 1))       # (base node, tip node)
    s = np.array(s)
    pos = [list(soft_base + si * tangent) + list(BASE_QUAT) for si in s]
    F = soft_base
    if WALL_STL:          # start fully retracted in the sheath: tip at the entry point F
        F = np.array(DESIGN[CASE]['soft_base_m'])
        pos = [list(F + (si - s[-1]) * tangent) + list(BASE_QUAT) for si in s]
    section_of_node = [section_of_edge[0]] + section_of_edge
    edges = [[i, i + 1] for i in range(len(s) - 1)]
    # lumped weight rho*A*g per node (half of each adjacent element)
    weight = np.zeros((len(s), 3))
    for e, (a, b) in enumerate(edges):
        sec = SECTIONS[section_of_edge[e]]
        f = sec['rho'] * sec['A'] * (s[b] - s[a]) * GRAVITY
        weight[a] += 0.5 * f
        weight[b] += 0.5 * f
    return dict(s=s, pos=pos, edges=edges, section_of_edge=section_of_edge, F=F, d=tangent,
                section_of_node=section_of_node,
                node_ranges=node_ranges, weight=weight, soft_base=soft_base)


# ---------------------------------------------------------------- controller
class PneumaticFollowerController(Sofa.Core.Controller):
    """Every step: ramp pressures, convert to follower end moments, add loads."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.root = kwargs['root']
        self.dofs = kwargs['dofs']
        self.loads = kwargs['loads']
        self.rod = kwargs['rod']
        self.sheath = kwargs.get('sheath')          # FixedProjectiveConstraint (wall mode)
        self.step = 0
        self.depth = 0.0                            # inserted soft length [m] (wall mode)
        self.pressures = [np.zeros(sec['nChambers']) for sec in SECTIONS]
        self.cmd_depth = 0.0                        # RL mode commands
        self.cmd_pressures = [np.zeros(sec['nChambers']) for sec in SECTIONS]

    def set_pressures(self, pressures):
        """Hook for RL / teleoperation: list of per-chamber gauge pressures [Pa]."""
        self.pressures = [np.asarray(p, float) for p in pressures]

    def insert(self, depth):
        """Hold the not-yet-inserted nodes on the axis behind F (inside the sheath)."""
        rod = self.rod
        L = rod['s'][-1]
        self.depth = min(max(depth, 0.0), L)
        behind = rod['s'] - (L - self.depth)            # < 0: still in the sheath
        idx = np.where(behind <= 1e-9)[0]
        with self.dofs.position.writeableArray() as x:
            x[idx, :3] = rod['F'] + np.outer(behind[idx], rod['d'])
            x[idx, 3:] = BASE_QUAT
        self.sheath.indices.value = idx.tolist()

    def onAnimateBeginEvent(self, _):
        t = self.root.time.value
        if RL_MODE:
            if self.sheath is not None:
                self.insert(self.cmd_depth)
            scale, load_scale = None, 1.0
        elif self.sheath is not None:
            self.insert(INSERT_SPEED * t)
            scale = smooth_step(self.depth / self.rod['s'][-1])
            load_scale = 1.0
        else:
            scale = smooth_step(t / T_RAMP)
            # External loads are ramped as well (MATLAB applies them at once, but the
            # shooting method has no load path; Newton on the beam needs small
            # increments to stay in the basin of the equilibrium).
            load_scale = smooth_step(t / T_RAMP)
        self.set_pressures(self.cmd_pressures if RL_MODE else [scale * sec['pTarget'] for sec in SECTIONS])

        x = np.asarray(self.dofs.position.value)
        f = np.zeros((len(x), 6))
        f[:, :3] = load_scale * self.rod['weight']
        f[-1, :3] += load_scale * TIP_FORCE
        f[-1, 3:] += load_scale * TIP_MOMENT
        # The chamber moment is INTERNAL and uniform along a section: in the
        # Cosserat balance it acts like a body-frame rest-curvature shift.  Its
        # exact external equivalent is end moments +-R*M PLUS a distributed
        # moment -(R*M)'.  Lumped per element this is a couple +-R_mid*M on the
        # element's two nodes, with R_mid the element mid-frame.  (End couples
        # alone are wrong once the rod rotates: they leak into the next section.)
        for e, (a, b) in enumerate(self.rod['edges']):
            sec_id = self.rod['section_of_edge'][e]
            m_body = actuation_moment_body(SECTIONS[sec_id], self.pressures[sec_id])
            qa, qb = x[a, 3:], x[b, 3:]
            q_mid = qa + qb if np.dot(qa, qb) >= 0 else qa - qb
            m_world = quat_to_rotm(q_mid) @ m_body
            f[b, 3:] += m_world
            f[a, 3:] -= m_world
        with self.loads.forces.writeableArray() as forces:
            forces[:] = f

    def onAnimateEndEvent(self, _):
        self.step += 1
        if self.step == EXPORT_AT_STEP and not RL_MODE:
            self.export_and_compare()

    def export_and_compare(self):
        x = np.asarray(self.dofs.position.value)[:, :3]
        np.savetxt(SOFA_CENTERLINE, x, delimiter=',')
        print(f'[cosserat_robot] case = {CASE}, t = {self.root.time.value:.2f} s, '
              f'tip [mm] = {np.array2string(1e3 * x[-1], precision=2)}')
        for sec, p in zip(SECTIONS, self.pressures):
            print(f'[cosserat_robot] chamber pressures [kPa] = '
                  f'{np.array2string(p / 1e3, precision=2)}')
        if self.sheath is not None:
            self.report_wall(x)
            return
        if not os.path.exists(MATLAB_CENTERLINE):
            return
        ref = np.loadtxt(MATLAB_CENTERLINE, delimiter=',')
        # resample the MATLAB centreline at the SOFA arc-length stations
        s_ref = np.r_[0, np.cumsum(np.linalg.norm(np.diff(ref, axis=0), axis=1))]
        s_ref *= self.rod['s'][-1] / s_ref[-1]
        ref_i = np.column_stack([np.interp(self.rod['s'], s_ref, ref[:, k]) for k in range(3)])
        err = np.linalg.norm(x - ref_i, axis=1)
        print(f'[cosserat_robot] MATLAB tip [mm] = {np.array2string(1e3 * ref[-1], precision=2)}')
        print(f'[cosserat_robot] tip error = {1e3 * err[-1]:.2f} mm, '
              f'centreline RMS = {1e3 * np.sqrt(np.mean(err ** 2)):.2f} mm, '
              f'max = {1e3 * err.max():.2f} mm  (soft length {1e3 * self.rod["s"][-1]:.0f} mm)')

    def report_wall(self, x):
        """Wall mode: penetration into the FULL wall and deviation from the patient path."""
        sys.path.insert(0, os.path.join(HERE, 'tools'))
        from ventricle_wall import read_stl, distance_to_mesh, inside
        d = DESIGN[CASE]
        full_wall = os.path.join(os.path.dirname(os.path.dirname(d['ref_path_file'])), 'ventricles',
                                 d['patient_path'] + '.stl')
        free = self.rod['s'] > self.rod['s'][-1] - self.depth + 1e-9
        r = np.array([SECTIONS[self.rod['section_of_node'][i]]['radius'] for i in range(len(x))])
        for label, fn in [('wall with entry hole', WALL_STL), ('full wall', full_wall)]:
            if not os.path.exists(fn):
                continue
            dist = distance_to_mesh(1e3 * x[free], read_stl(fn)) / 1e3
            pen = (r[free] - dist) * 1e3
            s_out = (self.rod['s'][free] - (self.rod['s'][-1] - self.depth)) * 1e3   # mm past F
            where = (f' at {s_out[pen > 0.1].min():.0f}-{s_out[pen > 0.1].max():.0f} mm past F'
                     if np.any(pen > 0.1) else '')
            n_out = np.sum(~inside(1e3 * x[free], read_stl(fn))) if label == 'full wall' else 0
            print(f'[cosserat_robot] {label}: min centreline-wall distance {1e3 * dist.min():.2f} mm, '
                  f'max penetration (r - dist) {pen.max():+.2f} mm, nodes penetrating > 0.1 mm: '
                  f'{np.sum(pen > 0.1)} / {free.sum()}{where}'
                  + (f'; centreline nodes outside the lumen: {n_out}' if label == 'full wall' else ''))
        path = np.loadtxt(d['ref_path_file'], delimiter=',') / 1e3 if os.path.exists(d['ref_path_file']) else None
        if path is not None:
            sp = np.r_[0, np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1))]
            s_free = self.rod['s'][free] - (self.rod['s'][-1] - self.depth)
            ok = s_free <= sp[-1]
            p_i = np.column_stack([np.interp(s_free[ok], sp, path[:, k]) for k in range(3)])
            e = np.linalg.norm(x[free][ok] - p_i, axis=1) * 1e3
            print(f'[cosserat_robot] inserted {1e3 * self.depth:.1f} mm; deviation from the patient path: '
                  f'mean {e.mean():.2f} mm, tip {e[-1]:.2f} mm')


# --------------------------------------------------------------------- scene
def createScene(root):
    for p in ['Sofa.Component.Constraint.Projective',   # FixedProjectiveConstraint
              'Sofa.Component.Mass',                    # UniformMass
              'Sofa.Component.LinearSolver.Direct',     # SparseLDLSolver
              'Sofa.Component.Mapping.Linear',          # IdentityMapping
              'Sofa.Component.MechanicalLoad',          # ConstantForceField
              'Sofa.Component.ODESolver.Backward',      # EulerImplicitSolver / StaticSolver
              'Sofa.Component.SolidMechanics.FEM.Elastic',  # BeamFEMForceField
              'Sofa.Component.StateContainer',          # MechanicalObject
              'Sofa.Component.Topology.Container.Dynamic',  # EdgeSetTopologyContainer
              'Sofa.Component.Visual',                  # VisualStyle, CylinderVisualModel
              'Sofa.GL.Component.Rendering3D']:         # OglSceneFrame
        root.addObject('RequiredPlugin', name=p.replace('.', '_'), pluginName=p)

    root.dt = DT
    root.gravity = [0, 0, 0]          # weight is applied explicitly (exact rho*A*g per node)
    root.addObject('VisualStyle', displayFlags='showVisualModels showBehaviorModels')
    if WALL_STL:
        for p in ['Sofa.Component.AnimationLoop', 'Sofa.Component.Collision.Detection.Algorithm',
                  'Sofa.Component.Collision.Detection.Intersection', 'Sofa.Component.Collision.Geometry',
                  'Sofa.Component.Collision.Response.Contact', 'Sofa.Component.Constraint.Lagrangian.Correction',
                  'Sofa.Component.Constraint.Lagrangian.Solver', 'Sofa.Component.IO.Mesh',
                  'Sofa.Component.Topology.Container.Constant']:
            root.addObject('RequiredPlugin', name=p.replace('.', '_'), pluginName=p)
        root.addObject('FreeMotionAnimationLoop')
        root.addObject('BlockGaussSeidelConstraintSolver', name='constraint_solver',
                       maxIterations=int(os.environ.get('ROBOT_LCP_ITERS', 200)),
                       tolerance=float(os.environ.get('ROBOT_LCP_TOL', 1e-6)), computeConstraintForces=True)
        root.addObject('CollisionPipeline')
        root.addObject('BruteForceBroadPhase')
        root.addObject('BVHNarrowPhase')
        root.addObject('MinProximityIntersection', alarmDistance=1.0e-3, contactDistance=0.1e-3)
        root.addObject('CollisionResponse', response='FrictionContactConstraint',
                       responseParams=f'mu={MU}')
        wall = root.addChild('VentricleWall')
        wall.addObject('MeshSTLLoader', name='loader', filename=WALL_STL, scale3d=[1e-3] * 3)
        wall.addObject('MeshTopology', src='@loader')
        wall.addObject('MechanicalObject', template='Vec3d', position='@loader.position')
        for cm in ('TriangleCollisionModel', 'LineCollisionModel', 'PointCollisionModel'):
            wall.addObject(cm, moving=False, simulated=False)
        wall.addChild('Visual').addObject('OglModel', src='@../loader',
                                          color=[0.12, 0.59, 0.99, 0.25])
    else:
        root.addObject('DefaultAnimationLoop')
    root.addObject('OglSceneFrame', style='Arrows', alignment='TopRight')

    rod = build_rod()

    robot = root.addChild('SoftRobot')
    if SOLVER == 'static':
        robot.addObject('NewtonRaphsonSolver', name='newton', maxNbIterationsNewton=20,
                        absoluteResidualStoppingThreshold=1e-10,
                        relativeEstimateDifferenceThreshold=1e-8,
                        updateStateWhenDiverged=False,
                        warnWhenLineSearchFails=False)
        robot.addObject('StaticSolver', newtonSolver='@newton')
    else:
        robot.addObject('EulerImplicitSolver', rayleighStiffness=0.1, rayleighMass=2.0)
    robot.addObject('SparseLDLSolver', template='CompressedRowSparseMatrixd')
    robot.addObject('EdgeSetTopologyContainer', name='topology',
                    position=[p[:3] for p in rod['pos']], edges=rod['edges'])
    dofs = robot.addObject('MechanicalObject', name='frames', template='Rigid3d',
                           position=rod['pos'], showObject=True, showObjectScale=0.006)
    if WALL_STL:
        robot.addObject('GenericConstraintCorrection')
    sheath = robot.addObject('FixedProjectiveConstraint', name='sheath',   # rigid insertion
                             indices=list(range(len(rod['pos']))) if WALL_STL else [0])
    # Inertia only matters for the transient (weight is applied explicitly).
    # Per-frame mass of a 5 mm slice; inertia of a solid cylinder slice.
    sec0 = SECTIONS[0]
    m = sec0['rho'] * sec0['A'] * ELEMENT_LENGTH
    r2, l2 = sec0['radius'] ** 2, ELEMENT_LENGTH ** 2
    Ix, Iyz = m * r2 / 2, m * (3 * r2 + l2) / 12
    # RigidMass text format: mass volume inertia(3x3, per unit mass)
    robot.addObject('UniformMass', vertexMass=' '.join(
        str(v) for v in [m, 1.0, Ix / m, 0, 0, 0, Iyz / m, 0, 0, 0, Iyz / m]))
    for j, sec in enumerate(SECTIONS):
        segs = [e for e, sj in enumerate(rod['section_of_edge']) if sj == j]
        robot.addObject('BeamFEMForceField', name=f'section{j + 1}',
                        youngModulus=sec['E0'], poissonRatio=0.499,
                        radius=sec['radius'], listSegment=segs)
    loads = robot.addObject('ConstantForceField', name='loads', template='Rigid3d',
                            indices=list(range(len(rod['pos']))),
                            forces=[[0] * 6] * len(rod['pos']), showArrowSize=0)

    # CylinderVisualModel is itself a (visual) state: one per node, mapped from the frames
    for j, sec in enumerate(SECTIONS):
        edges = [rod['edges'][e] for e, sj in enumerate(rod['section_of_edge']) if sj == j]
        visu = robot.addChild(f'VisualSection{j + 1}')
        tube = visu.addObject('CylinderVisualModel', name='tube', radius=sec['radius'],
                              color=sec['color'], edges=edges)
        visu.addObject('IdentityMapping', input=dofs.getLinkPath(), output=tube.getLinkPath())

    if WALL_STL:   # contact spheres of the robot radius on every frame
        col = robot.addChild('Collision')
        col.addObject('MechanicalObject', template='Vec3d')
        # one radius for all spheres (listRadius is not picked up here and falls back to the
        # default radius 1 = 1 m); section 1 is the thicker one, so this is conservative
        col.addObject('SphereCollisionModel', radius=max(sec['radius'] for sec in SECTIONS))
        col.addObject('IdentityMapping')
        rigid_pts = [list(rod['F'] - ROBOT['Lrigid'] * rod['d']), list(rod['F'])]
    else:
        rigid_pts = [list(BASE_POS), list(rod['soft_base'])]

    rigid = root.addChild('RigidInsertion')
    rigid.addObject('CylinderVisualModel', radius=ROBOT['rigidRadius'],
                    color=[0.25, 0.28, 0.32, 1.0], edges=[[0, 1]],
                    position=rigid_pts)

    if os.path.exists(MATLAB_CENTERLINE) and not WALL_STL:   # MATLAB equilibrium overlay
        ref = root.addChild('MatlabReference')
        ref.addObject('MechanicalObject', template='Vec3d',
                      position=np.loadtxt(MATLAB_CENTERLINE, delimiter=',').tolist(),
                      showObject=True, showObjectScale=4, showColor=[1, 0, 1, 1])

    robot.addObject(PneumaticFollowerController(name='pneumatics', root=root, dofs=dofs,
                                                loads=loads, rod=rod,
                                                sheath=sheath if WALL_STL else None))
    return root
