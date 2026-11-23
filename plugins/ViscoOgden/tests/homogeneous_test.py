"""
Validation of ViscoOgdenFEMForceField under prescribed homogeneous deformation x = F(t) X.

All nodes of a unit cube (6 tetrahedra) are moved by a controller and held by a
FixedProjectiveConstraint, so the force field only evaluates and commits the constitutive
update.  Its per-tetrahedron 2nd Piola-Kirchhoff stress is compared with an independent
NumPy reference: the stress is obtained by finite differences of the strain energy W(C)
(not from the closed-form expressions used in C++), followed by the same Simo/Holzapfel
internal-variable update.

    plugins/ViscoOgden/run.sh tests/homogeneous_test.py

Cases
  step     : uniaxial isochoric stretch 1.25 applied at t = dt, held -> stress relaxation
  shear    : cyclic simple shear gamma = 0.3 sin(2 pi t) -> hysteresis (and none with viscous=0)
  ogden2   : two Ogden terms (mu = 600, 400; alpha = -20, 5), compressible step stretch
Prints max relative error per case and writes tests/homogeneous_test.npz.
"""
import os
import sys

import numpy as np

import Sofa
import Sofa.Core
import Sofa.Simulation

HERE = os.path.dirname(os.path.abspath(__file__))

X = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0], [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1]], float)
TETS = [[0, 1, 3, 4], [1, 2, 3, 6], [1, 3, 4, 6], [1, 4, 5, 6], [3, 4, 6, 7], [1, 6, 3, 4]][:5]   # 5-tet cube


# ----------------------------------------------------------------- reference model (NumPy)
def energy(C, mu, alpha, kappa):
    J = np.sqrt(np.linalg.det(C))
    c = np.linalg.eigvalsh(C)
    W = kappa / 2 * np.log(J) ** 2
    Wiso = 0.0
    for m, a in zip(mu, alpha):
        lb = J ** (-a / 3) * c ** (a / 2)
        Wiso += 2 * m / a ** 2 * (lb.sum() - 3)
    return W, Wiso


def stress_fd(C, mu, alpha, kappa, h=1e-6):
    """S_vol, S_iso = 2 dW/dC by central differences on the symmetric C."""
    Sv, Si = np.zeros((3, 3)), np.zeros((3, 3))
    for i in range(3):
        for j in range(i, 3):
            E = np.zeros((3, 3))
            E[i, j] = E[j, i] = h if i != j else h
            wp = energy(C + E, mu, alpha, kappa)
            wm = energy(C - E, mu, alpha, kappa)
            k = 1.0 if i == j else 0.5          # dW/dC_ij for a symmetric perturbation of both entries
            Sv[i, j] = Sv[j, i] = 2 * k * (wp[0] - wm[0]) / (2 * h)
            Si[i, j] = Si[j, i] = 2 * k * (wp[1] - wm[1]) / (2 * h)
    return Sv, Si


def reference(Fs, dt, mu, alpha, kappa, beta, tau, viscous=True):
    beta, tau = np.asarray(beta, float), np.asarray(tau, float)
    Q = np.zeros((len(beta), 3, 3))
    Sprev = np.zeros((3, 3))
    out = []
    for F in Fs:
        Sv, Si = stress_fd(F.T @ F, mu, alpha, kappa)
        if viscous and len(beta):
            ef, eh = np.exp(-dt / tau), np.exp(-dt / (2 * tau))
            Q = ef[:, None, None] * Q + (beta * eh)[:, None, None] * (Si - Sprev)[None]
            Sprev = Si
        out.append(Sv + Si + Q.sum(0))
    return np.array(out)


# ----------------------------------------------------------------- SOFA run
class Driver(Sofa.Core.Controller):
    def __init__(self, mo, Fs, *a, **k):
        super().__init__(*a, **k)
        self.mo, self.Fs, self.k = mo, Fs, 0

    def onAnimateBeginEvent(self, e):
        F = self.Fs[min(self.k, len(self.Fs) - 1)]
        self.mo.position.value = (X @ F.T).tolist()
        self.k += 1


def run_sofa(Fs, dt, mat, viscous=True):
    root = Sofa.Core.Node('root')
    root.dt = dt
    root.gravity = [0, 0, 0]
    for p in ['Sofa.Component.ODESolver.Backward', 'Sofa.Component.LinearSolver.Iterative',
              'Sofa.Component.StateContainer', 'Sofa.Component.Topology.Container.Constant',
              'Sofa.Component.Mass', 'Sofa.Component.Constraint.Projective', 'ViscoOgden']:
        root.addObject('RequiredPlugin', pluginName=p)
    root.addObject('DefaultAnimationLoop')
    n = root.addChild('cube')
    n.addObject('EulerImplicitSolver', rayleighMass=0, rayleighStiffness=0)
    n.addObject('CGLinearSolver', iterations=50, tolerance=1e-12, threshold=1e-14)
    mo = n.addObject('MechanicalObject', position=X.tolist())
    n.addObject('MeshTopology', tetrahedra=TETS)
    n.addObject('UniformMass', totalMass=1.0)
    ff = n.addObject('ViscoOgdenFEMForceField', mu=mat['mu'], alpha=mat['alpha'], bulkModulus=mat['kappa'],
                     beta=mat['beta'], tau=mat['tau'], viscous=viscous, checkTangent=True, printLog=True)
    n.addObject('FixedProjectiveConstraint', fixAll=True)
    n.addObject(Driver(mo, Fs, name='driver'))
    Sofa.Simulation.initRoot(root)
    S = []
    for _ in range(len(Fs)):
        Sofa.Simulation.animate(root, dt)
        S.append(np.array(ff.stress.value, float).reshape(-1, 3, 3))
    Sofa.Simulation.unload(root)
    return np.array(S)          # [steps, tets, 3, 3]


def check(name, Fs, dt, mat, viscous=True):
    S = run_sofa(Fs, dt, mat, viscous)
    R = reference(Fs, dt, mat['mu'], mat['alpha'], mat['kappa'], mat['beta'], mat['tau'], viscous)
    spread = np.abs(S - S[:, :1]).max()                          # homogeneous: all tets equal
    scale = np.abs(R).max()
    err = np.abs(S[:, 0] - R).max() / scale
    print(f'{name:8s}: max |S_sofa - S_ref| / max|S_ref| = {err:.2e}, tet spread {spread / scale:.1e}, '
          f'peak |S| {scale:.1f} Pa', flush=True)
    return S[:, 0], R, err


def main():
    dt = 0.01
    t = np.arange(1, 301) * dt
    brain = dict(mu=[1000.0], alpha=[-20.0], kappa=5e4, beta=[2.0, 0.5], tau=[0.05, 1.0])
    out, errs = {}, []

    lam = 1.25
    Fstep = [np.diag([lam, lam ** -0.5, lam ** -0.5])] * len(t)
    s, r, e = check('step', Fstep, dt, brain); out.update(step_t=t, step_sofa=s, step_ref=r); errs.append(e)

    g = 0.3 * np.sin(2 * np.pi * t)
    Fsh = [np.array([[1, gi, 0], [0, 1, 0], [0, 0, 1]]) for gi in g]
    s, r, e = check('shear', Fsh, dt, brain); out.update(shear_g=g, shear_sofa=s, shear_ref=r); errs.append(e)
    s0, r0, e0 = check('shear-el', Fsh, dt, brain, viscous=False); out.update(shear_el=s0); errs.append(e0)
    # hysteresis: loop area of S_xy vs gamma over the last cycle
    last = t >= 2.0
    area = lambda S: abs(np.trapezoid(S[last, 0, 1], g[last]))
    print(f'hysteresis area (last cycle): viscous {area(out["shear_sofa"]):.2f} Pa, elastic {area(s0):.2e} Pa')

    two = dict(mu=[600.0, 400.0], alpha=[-20.0, 5.0], kappa=2e4, beta=[1.0], tau=[0.2])
    Fc = [np.diag([1.2, 0.95, 0.9])] * 100
    s, r, e = check('ogden2', Fc, dt, two); out.update(o2_sofa=s, o2_ref=r); errs.append(e)

    np.savez(os.path.join(HERE, 'homogeneous_test.npz'), **out)
    ok = max(errs) < 1e-4
    print('PASS' if ok else 'FAIL', f'(max relative error {max(errs):.2e})')
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
