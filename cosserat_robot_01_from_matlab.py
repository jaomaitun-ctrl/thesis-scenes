"""
Soft continuum robot imported from MATLAB (cosserat_yeoh_quasistatic.m) into SOFA.

Robot (same numbers as the MATLAB script):
    rigid insertion 80 mm  ->  soft section 1 (120 mm, r 6.0 mm, 3 chambers)
                           ->  soft section 2 (100 mm, r 5.5 mm, 3 chambers)
    base frame: body x (rod tangent) -> world +Z, base at the origin.

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

Verified against the MATLAB solver (matlab_refs/<case>.csv, 600 steps):
    actuation_only      tip error 0.14 mm, centreline RMS 0.06 mm
    actuation_tipforce  tip error 1.35 mm, centreline RMS 0.47 mm
    hanging             tip error 0.98 mm, centreline RMS 0.59 mm
    matlab_default      DOES NOT MATCH, and should not: the robot points up and its
                        own weight (~1.1 N/m) is ~4x the self-weight buckling load of
                        the column (7.84*EI/L^3 ~ 0.26 N/m).  MATLAB's shooting method
                        returns the upright, UNSTABLE equilibrium; SOFA falls to the
                        stable one (tip z ~ -50 mm).

Run (GUI)  -- from the SOFA checkout that has the pixi env:
    cd ~/Developer/sofa
    ROBOT_CASE=actuation_only pixi run -e supported-plugins \
        .pixi/envs/supported-plugins/bin/runSofa -l SofaPython3 \
        ~/sofa/thesis-scenes/cosserat_robot_01_from_matlab.py
Run (headless; writes matlab_refs/<case>_sofa.csv and prints the comparison):
    add  -g batch -n 610  to the command above.
"""
import os
import numpy as np
import Sofa

HERE = os.path.dirname(os.path.abspath(__file__))

# Load case (select with:  ROBOT_CASE=actuation_only runSofa ...).  Each has a
# MATLAB reference in matlab_refs/<case>.csv produced by the unmodified solver.
#   matlab_default      gravity -Z, tip force   (robot points UP: weight compresses it)
#   actuation_only      no gravity, no tip force
#   actuation_tipforce  no gravity, tip force
#   hanging             gravity +Z (weight puts the rod in tension), tip force
CASES = {
    'matlab_default':     ([0, 0, -9.81], [0.005, 0, -0.010]),
    'actuation_only':     ([0, 0, 0],     [0, 0, 0]),
    'actuation_tipforce': ([0, 0, 0],     [0.005, 0, -0.010]),
    'hanging':            ([0, 0, 9.81],  [0.005, 0, -0.010]),
}
CASE = os.environ.get('ROBOT_CASE', 'actuation_only')
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
EXPORT_AT_STEP = int(os.environ.get('ROBOT_EXPORT_STEP', 600))   # write CSV + compare

# MATLAB baseQuaternion [w x y z] = [cos(-pi/4) 0 sin(-pi/4) 0]  ->  SOFA [x y z w]
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
    """Moment part of actuationWrench() with t = body x: sum_i (d_i x x) tau_i."""
    tau = p * sec['Ain']
    d = sec['d']
    return np.array([0.0, np.sum(d[2] * tau), -np.sum(d[1] * tau)])


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
    soft_base = ROBOT['Lrigid'] * tangent
    s, section_of_edge, node_ranges = [0.0], [], []
    for j, sec in enumerate(SECTIONS):
        n = int(round(sec['L'] / ELEMENT_LENGTH))
        first = len(s) - 1
        s += list(s[-1] + np.linspace(0, sec['L'], n + 1)[1:])
        section_of_edge += [j] * n
        node_ranges.append((first, len(s) - 1))       # (base node, tip node)
    s = np.array(s)
    pos = [list(soft_base + si * tangent) + list(BASE_QUAT) for si in s]
    edges = [[i, i + 1] for i in range(len(s) - 1)]
    # lumped weight rho*A*g per node (half of each adjacent element)
    weight = np.zeros((len(s), 3))
    for e, (a, b) in enumerate(edges):
        sec = SECTIONS[section_of_edge[e]]
        f = sec['rho'] * sec['A'] * (s[b] - s[a]) * GRAVITY
        weight[a] += 0.5 * f
        weight[b] += 0.5 * f
    return dict(s=s, pos=pos, edges=edges, section_of_edge=section_of_edge,
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
        self.step = 0
        self.pressures = [np.zeros(sec['nChambers']) for sec in SECTIONS]

    def set_pressures(self, pressures):
        """Hook for RL / teleoperation: list of per-chamber gauge pressures [Pa]."""
        self.pressures = [np.asarray(p, float) for p in pressures]

    def onAnimateBeginEvent(self, _):
        t = self.root.time.value
        scale = smooth_step(t / T_RAMP)
        self.set_pressures([scale * sec['pTarget'] for sec in SECTIONS])

        # External loads are ramped as well (MATLAB applies them at once, but the
        # shooting method has no load path; Newton on the beam needs small
        # increments to stay in the basin of the equilibrium).
        load_scale = smooth_step(t / T_RAMP)
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
        if self.step == EXPORT_AT_STEP:
            self.export_and_compare()

    def export_and_compare(self):
        x = np.asarray(self.dofs.position.value)[:, :3]
        np.savetxt(SOFA_CENTERLINE, x, delimiter=',')
        print(f'[cosserat_robot] case = {CASE}, t = {self.root.time.value:.2f} s, '
              f'tip [mm] = {np.array2string(1e3 * x[-1], precision=2)}')
        for sec, p in zip(SECTIONS, self.pressures):
            print(f'[cosserat_robot] chamber pressures [kPa] = '
                  f'{np.array2string(p / 1e3, precision=2)}')
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
              f'max = {1e3 * err.max():.2f} mm  (soft length 220 mm)')


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
    robot.addObject('FixedProjectiveConstraint', indices=[0])   # end of rigid insertion
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

    rigid = root.addChild('RigidInsertion')
    rigid.addObject('CylinderVisualModel', radius=ROBOT['rigidRadius'],
                    color=[0.25, 0.28, 0.32, 1.0], edges=[[0, 1]],
                    position=[[0, 0, 0], list(rod['soft_base'])])

    if os.path.exists(MATLAB_CENTERLINE):          # MATLAB final equilibrium, for overlay
        ref = root.addChild('MatlabReference')
        ref.addObject('MechanicalObject', template='Vec3d',
                      position=np.loadtxt(MATLAB_CENTERLINE, delimiter=',').tolist(),
                      showObject=True, showObjectScale=4, showColor=[1, 0, 1, 1])

    robot.addObject(PneumaticFollowerController(name='pneumatics', root=root, dofs=dofs,
                                                loads=loads, rod=rod))
    return root
