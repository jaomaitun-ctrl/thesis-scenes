"""
Dynamic check of ViscoOgdenFEMForceField in a real solve: creep of a soft column.

A 10 x 10 x 20 mm column (Kuhn 6-tet cells), bottom face fixed, a constant compressive
load (200 Pa) on the top face from t = 0.  Implicit Euler with a direct solver (so the
assembled stiffness matrix path, buildStiffnessMatrix, is used), heavy Rayleigh mass damping
so the response is quasi-static.

Expected: the viscoelastic column first deforms like a hyperelastic one with the
instantaneous moduli (1 + sum beta) mu, then creeps to the hyperelastic long-term solution.
Both bounds are computed with viscous=False runs and the same mesh.

    plugins/ViscoOgden/run.sh tests/creep_test.py
"""
import os
import sys

import numpy as np

import Sofa
import Sofa.Core
import Sofa.Simulation

HERE = os.path.dirname(os.path.abspath(__file__))
NX, NY, NZ, H = 4, 4, 8, 2.5e-3          # cells, cell size [m]
MAT = dict(mu=[1000.0], alpha=[-20.0], bulkModulus=5e4, beta=[2.0, 0.5], tau=[0.05, 1.0])
LOAD_PA = 200.0
DT, T_END = 0.005, 4.0


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
    return np.array(pts), tets


def run(mat, viscous, check=False):
    pts, tets = grid()
    top = np.where(np.isclose(pts[:, 2], NZ * H))[0]
    bottom = np.where(np.isclose(pts[:, 2], 0))[0]
    # consistent nodal loads on the top face (corner/edge/interior weights of bilinear cells)
    w = np.ones(len(top))
    on_x = np.isclose(pts[top, 0], 0) | np.isclose(pts[top, 0], NX * H)
    on_y = np.isclose(pts[top, 1], 0) | np.isclose(pts[top, 1], NY * H)
    w[on_x] *= 0.5
    w[on_y] *= 0.5
    f = -LOAD_PA * H * H * w

    root = Sofa.Core.Node('root')
    root.dt = DT
    root.gravity = [0, 0, 0]
    for p in ['Sofa.Component.ODESolver.Backward', 'Sofa.Component.LinearSolver.Direct',
              'Sofa.Component.StateContainer', 'Sofa.Component.Topology.Container.Constant',
              'Sofa.Component.Mass', 'Sofa.Component.Constraint.Projective',
              'Sofa.Component.MechanicalLoad', 'ViscoOgden']:
        root.addObject('RequiredPlugin', pluginName=p, name=p.replace('.', '_'))
    root.addObject('DefaultAnimationLoop')
    n = root.addChild('column')
    n.addObject('EulerImplicitSolver', rayleighMass=20, rayleighStiffness=0)
    n.addObject('SparseLDLSolver', template='CompressedRowSparseMatrixMat3x3d')
    mo = n.addObject('MechanicalObject', position=pts.tolist())
    n.addObject('MeshTopology', tetrahedra=tets)
    n.addObject('UniformMass', totalMass=1000.0 * NX * NY * NZ * H ** 3)
    m = dict(mat)
    n.addObject('ViscoOgdenFEMForceField', viscous=viscous, checkTangent=check, printLog=check, **m)
    n.addObject('FixedProjectiveConstraint', indices=bottom.tolist())
    n.addObject('ConstantForceField', indices=top.tolist(), forces=[[0, 0, fi] for fi in f])
    Sofa.Simulation.initRoot(root)
    t, uz = [], []
    for s in range(int(round(T_END / DT))):
        Sofa.Simulation.animate(root, DT)
        x = np.array(mo.position.value)
        t.append((s + 1) * DT)
        uz.append((x[top, 2] - pts[top, 2]).mean())
    Sofa.Simulation.unload(root)
    return np.array(t), np.array(uz) * 1e3      # mm


def main():
    t, visco = run(MAT, True, check=True)
    _, longterm = run(MAT, False)
    inst = dict(MAT, mu=[MAT['mu'][0] * (1 + sum(MAT['beta']))])
    _, instant = run(inst, False)
    u_inst, u_long = instant[-1], longterm[-1]
    # dynamics of the damped bound runs settle within ~0.3 s; compare the creep curve there
    k = np.searchsorted(t, 0.3)
    print(f'top displacement [mm]: instantaneous bound {u_inst:.4f}, long-term bound {u_long:.4f}')
    print(f'viscoelastic: t=0.3 s {visco[k]:.4f}, t=1 s {visco[np.searchsorted(t, 1.0)]:.4f}, '
          f't={t[-1]:.1f} s {visco[-1]:.4f}')
    between = np.all(visco[k:] <= u_inst * 0.98 + 1e-9) and np.all(visco[k:] >= u_long * 1.02 - 1e-9)
    monotone = np.all(np.diff(visco[k:]) <= 1e-9)
    approach = abs(visco[-1] - u_long) / abs(u_long)
    print(f'creep stays between the bounds: {between}; monotone: {monotone}; '
          f'final gap to long-term {100 * approach:.1f} %')
    np.savez(os.path.join(HERE, 'creep_test.npz'), t=t, visco=visco, longterm=longterm, instant=instant)
    ok = between and monotone and approach < 0.05
    print('PASS' if ok else 'FAIL')
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
