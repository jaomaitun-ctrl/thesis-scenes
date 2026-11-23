"""
GUI demo for ViscoOgdenFEMForceField: a brain-tissue-soft column pressed from the top for
1 s, then released.  It sinks quickly, keeps creeping while loaded, and recovers slowly
after release (the internal variables relax with tau = 0.05 s and 1 s).  The log prints the
top displacement and the mean internal-stress norm every 0.1 s.

    open in SOFA.app (File > Open Simulation), or
    runSofa -l SofaPython3 plugins/ViscoOgden/examples/viscoogden_column.py
(SOFA_PLUGIN_PATH must include plugins/ViscoOgden/build; SOFA.app sets it.)
"""
import numpy as np

import Sofa
import Sofa.Core

NX, NY, NZ, H = 4, 4, 8, 2.5e-3
LOAD_PA, T_LOAD = 400.0, 1.0


def grid():
    idx = lambda i, j, k: (k * (NY + 1) + j) * (NX + 1) + i
    pts = [[i * H, j * H, k * H] for k in range(NZ + 1) for j in range(NY + 1) for i in range(NX + 1)]
    tets = []
    for k in range(NZ):
        for j in range(NY):
            for i in range(NX):
                v = [idx(i, j, k), idx(i + 1, j, k), idx(i + 1, j + 1, k), idx(i, j + 1, k),
                     idx(i, j, k + 1), idx(i + 1, j, k + 1), idx(i + 1, j + 1, k + 1), idx(i, j + 1, k + 1)]
                for a, b in [(1, 2), (2, 3), (3, 7), (7, 4), (4, 5), (5, 1)]:
                    tets.append([v[0], v[a], v[b], v[6]])
    return np.array(pts), np.array(tets)


def surface(pts, tets):
    """Boundary triangles, oriented outwards."""
    faces = {}
    for t in tets:
        for f, o in [((0, 1, 2), 3), ((0, 1, 3), 2), ((0, 2, 3), 1), ((1, 2, 3), 0)]:
            tri = [t[i] for i in f]
            key = tuple(sorted(tri))
            faces[key] = None if key in faces else (tri, t[o])
    out = []
    for v in faces.values():
        if v is None:
            continue
        (a, b, c), o = v
        n = np.cross(pts[b] - pts[a], pts[c] - pts[a])
        out.append([a, b, c] if np.dot(n, pts[o] - pts[a]) < 0 else [a, c, b])
    return out


class Loader(Sofa.Core.Controller):
    def __init__(self, cff, forces, mo, top, ff, z0, *a, **k):
        super().__init__(*a, **k)
        self.cff, self.forces, self.mo, self.top, self.ff = cff, forces, mo, top, ff
        self.t, self.next_log = 0.0, 0.0
        self.z0 = z0

    def onAnimateBeginEvent(self, e):
        dt = self.getContext().getDt()
        on = (self.t % 3.0) < T_LOAD          # 1 s on, 2 s off, repeating
        self.cff.forces.value = (self.forces if on else 0 * self.forces).tolist()
        self.t += dt
        if self.t >= self.next_log - 1e-9:
            self.next_log += 0.1
            uz = (np.array(self.mo.position.value)[self.top, 2] - self.z0).mean() * 1e3
            q = np.mean(self.ff.internalStressNorm.value)
            Sofa.msg_info('ViscoOgden demo', f't = {self.t:4.1f} s  load {"ON " if on else "off"}  '
                                             f'top {uz:7.3f} mm  mean |Q| {q:7.1f} Pa')


def createScene(root):
    pts, tets = grid()
    top = np.where(np.isclose(pts[:, 2], NZ * H))[0]
    bottom = np.where(np.isclose(pts[:, 2], 0))[0]
    w = np.ones(len(top))
    w[np.isclose(pts[top, 0], 0) | np.isclose(pts[top, 0], NX * H)] *= 0.5
    w[np.isclose(pts[top, 1], 0) | np.isclose(pts[top, 1], NY * H)] *= 0.5
    forces = np.column_stack([0 * w, 0 * w, -LOAD_PA * H * H * w])

    root.dt = 0.005
    root.gravity = [0, 0, 0]
    for p in ['Sofa.Component.ODESolver.Backward', 'Sofa.Component.LinearSolver.Direct',
              'Sofa.Component.StateContainer', 'Sofa.Component.Topology.Container.Constant',
              'Sofa.Component.Mass', 'Sofa.Component.Constraint.Projective',
              'Sofa.Component.MechanicalLoad', 'Sofa.Component.Mapping.Linear',
              'Sofa.Component.Visual', 'Sofa.GL.Component.Rendering3D', 'ViscoOgden']:
        root.addObject('RequiredPlugin', pluginName=p, name=p.replace('.', '_'))
    root.addObject('DefaultAnimationLoop')
    root.addObject('VisualStyle', displayFlags='showVisualModels showWireframe')
    n = root.addChild('column')
    n.addObject('EulerImplicitSolver', rayleighMass=20, rayleighStiffness=0)
    n.addObject('SparseLDLSolver', template='CompressedRowSparseMatrixMat3x3d')
    mo = n.addObject('MechanicalObject', name='dofs', position=pts.tolist())
    n.addObject('MeshTopology', tetrahedra=tets.tolist())
    n.addObject('UniformMass', totalMass=1000.0 * NX * NY * NZ * H ** 3)
    ff = n.addObject('ViscoOgdenFEMForceField', name='tissue', mu=[1000.0], alpha=[-20.0], bulkModulus=5e4,
                     beta=[2.0, 0.5], tau=[0.05, 1.0])
    n.addObject('FixedProjectiveConstraint', indices=bottom.tolist())
    cff = n.addObject('ConstantForceField', indices=top.tolist(), forces=forces.tolist())
    v = n.addChild('visual')
    v.addObject('OglModel', position=pts.tolist(), triangles=surface(pts, tets), color=[0.95, 0.55, 0.45, 1])
    v.addObject('IdentityMapping')
    root.addObject(Loader(cff, forces, mo, top, ff, pts[top, 2], name='loader'))
    return root
