"""Minimal Cosserat-plugin check: cantilever with tip moment vs analytic M L^2 / (2 E I)."""
import numpy as np
import Sofa

E, nu, r, L, nsec, M = 349.8e3, 0.499, 6e-3, 0.12, 24, 1e-4
I = np.pi * r ** 4 / 4


class Report(Sofa.Core.Controller):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.frames, self.strain, self.k = k['frames'], k['strain'], 0

    def onAnimateEndEvent(self, _):
        self.k += 1
        if self.k in (1000, 3000, 6000):
            x = np.asarray(self.frames.position.value)
            print(f'COSSERAT step {self.k} tip [mm] = {np.round(1e3 * x[-1, :3], 4)}  '
                  f'kappa_z[0] = {self.strain.position.value[0][2]:.5f} 1/m')
            print(f'COSSERAT analytic tip deflection {1e3 * M * L ** 2 / (2 * E * I):.4f} mm, '
                  f'curvature {M / (E * I):.5f} 1/m')


def createScene(root):
    for p in ['Cosserat', 'Sofa.Component.Constraint.Projective', 'Sofa.Component.LinearSolver.Direct',
              'Sofa.Component.Mass', 'Sofa.Component.MechanicalLoad', 'Sofa.Component.ODESolver.Backward',
              'Sofa.Component.StateContainer']:
        root.addObject('RequiredPlugin', name=p.replace('.', '_'), pluginName=p)
    root.gravity = [0, 0, 0]
    root.dt = 0.01
    root.addObject('DefaultAnimationLoop')

    solver = root.addChild('solver')
    solver.addObject('EulerImplicitSolver', rayleighStiffness=0.1, rayleighMass=2.0)
    solver.addObject('SparseLDLSolver', template='CompressedRowSparseMatrixd')

    base = solver.addChild('RigidBase')
    base_mo = base.addObject('MechanicalObject', name='RigidBaseMO', template='Rigid3d',
                             position=[[0, 0, 0, 0, 0, 0, 1]])
    base.addObject('FixedProjectiveConstraint', indices=[0])

    ds = L / nsec
    strain_node = solver.addChild('cosseratCoordinate')
    strain = strain_node.addObject('MechanicalObject', name='cosseratCoordinateMO', template='Vec3d',
                                   position=[[0, 0, 0]] * nsec)
    strain_node.addObject('BeamHookeLawForceField', crossSectionShape='circular',
                          length=[ds] * nsec, radius=r, youngModulus=E, poissonRatio=nu)

    s_in = list(np.linspace(0, L, nsec + 1))
    s_out = s_in
    frames_node = base.addChild('frames')
    strain_node.addChild(frames_node)
    frames = frames_node.addObject('MechanicalObject', name='FramesMO', template='Rigid3d',
                                   position=[[s, 0, 0, 0, 0, 0, 1] for s in s_out])
    frames_node.addObject('UniformMass', totalMass=0.0143)
    frames_node.addObject('DiscreteCosseratMapping', curv_abs_input=s_in, curv_abs_output=s_out,
                          input1=strain.getLinkPath(), input2=base_mo.getLinkPath(),
                          output=frames.getLinkPath(), radius=r)
    frames_node.addObject('ConstantForceField', template='Rigid3d', indices=[nsec],
                          forces=[[0, 0, 0, 0, 0, M]])
    root.addObject(Report(frames=frames, strain=strain))
