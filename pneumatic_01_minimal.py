"""
Pneumatic actuation in SOFA -- minimal working scene.

Read the LAYER comments top to bottom. Every soft-robot scene in SOFA has
these same layers; only the meshes and numbers change.

Run:
    cd ~/sofa
    pixi run -e supported-plugins runSofa thesis-scenes/pneumatic_01_minimal.py
"""
import os
import Sofa

MESH = os.path.expanduser(
    '~/sofa/.pixi/envs/supported-plugins/plugins/SoftRobots/share/sofa/examples/'
    'SoftRobots/component/constraint/SurfacePressureConstraint/mesh/')

PLUGINS = [
    'Sofa.Component.AnimationLoop',                     # FreeMotionAnimationLoop
    'Sofa.Component.Constraint.Lagrangian.Correction',  # LinearSolverConstraintCorrection
    'Sofa.Component.Constraint.Lagrangian.Solver',      # BlockGaussSeidelConstraintSolver
    'Sofa.Component.Engine.Select',                     # BoxROI
    'Sofa.Component.IO.Mesh',                           # MeshOBJLoader, MeshVTKLoader
    'Sofa.Component.LinearSolver.Direct',               # EigenSimplicialLDLT
    'Sofa.Component.Mapping.Linear',                    # BarycentricMapping, IdentityMapping
    'Sofa.Component.Mass',                              # UniformMass
    'Sofa.Component.ODESolver.Backward',                # EulerImplicitSolver
    'Sofa.Component.SolidMechanics.FEM.Elastic',        # FastTetrahedralCorotationalForceField
    'Sofa.Component.SolidMechanics.Spring',             # RestShapeSpringsForceField
    'Sofa.Component.StateContainer',                    # MechanicalObject
    'Sofa.Component.Topology.Container.Constant',       # MeshTopology
    'Sofa.Component.Topology.Container.Dynamic',        # Tetrahedron/TriangleSetTopology*
    'Sofa.Component.Topology.Mapping',                  # Tetra2TriangleTopologicalMapping
    'Sofa.Component.Visual',                            # VisualStyle
    'Sofa.GL.Component.Rendering3D',                    # OglModel
    'SoftRobots',                                       # SurfacePressureConstraint
]


def createScene(root):

    # ---------------------------------------------------------------- LAYER 0
    # Plugins. SOFA loads almost nothing by default; every component you name
    # later must have its module declared here or the scene fails to build.
    for p in PLUGINS:
        root.addObject('RequiredPlugin', name=p)

    root.addObject('VisualStyle', displayFlags='showVisualModels showCollisionModels '
                                               'hideBoundingCollisionModels hideForceFields '
                                               'showInteractionForceFields hideWireframe')

    # ---------------------------------------------------------------- LAYER 1
    # The constraint machinery. THIS is what makes pneumatics possible.
    # FreeMotionAnimationLoop splits every timestep into two passes:
    #   (a) free motion   -- integrate the FEM ignoring all constraints
    #   (b) correction    -- solve for the Lagrange multipliers (= the forces
    #                        the constraints must apply) and correct positions.
    # A pneumatic cavity is implemented as a constraint, so without this loop
    # SurfacePressureConstraint silently does nothing.
    root.addObject('FreeMotionAnimationLoop')
    root.addObject('DefaultVisualManagerLoop')
    root.addObject('BlockGaussSeidelConstraintSolver', maxIterations=100, tolerance=1e-7)

    # ---------------------------------------------------------------- LAYER 2
    # The deformable body: the silicone. This node owns the DOFs.
    robot = root.addChild('robot')

    #   time integration (implicit -> stable at large dt, needed for stiff FEM)
    robot.addObject('EulerImplicitSolver', name='odesolver')
    #   linear solver: solves K dx = f each Newton step. Direct (LDLT) because
    #   LinearSolverConstraintCorrection below needs the factorised system.
    robot.addObject('EigenSimplicialLDLT', template='CompressedRowSparseMatrixMat3x3')

    #   geometry: a TETRAHEDRAL mesh. FEM strain is computed per tetrahedron,
    #   which is why a surface mesh is not enough for the deformable body.
    robot.addObject('MeshVTKLoader', name='loader', filename=MESH + 'Hollow_Stanford_Bunny.vtu')
    robot.addObject('TetrahedronSetTopologyContainer', src='@loader', name='container')
    robot.addObject('TetrahedronSetTopologyModifier')

    #   state vector: the actual x, v of every node. This is what you will read
    #   for RL observations later.
    robot.addObject('MechanicalObject', name='tetras', template='Vec3')
    robot.addObject('UniformMass', totalMass=0.5)

    #   constitutive law: linear elastic with corotational correction (handles
    #   large rotations, small strain). youngModulus = stiffness of the silicone.
    robot.addObject('FastTetrahedralCorotationalForceField', template='Vec3', name='FEM',
                    method='large', poissonRatio=0.3, youngModulus=18000)

    #   boundary condition: clamp the base. For your robot this is the point
    #   where the catheter enters -- the insertion DOF will move this later.
    robot.addObject('BoxROI', name='boxROI', box=[-5, -7, -5, 5, -4.5, 5], drawBoxes=True,
                    position='@tetras.rest_position', tetrahedra='@container.tetrahedra')
    robot.addObject('RestShapeSpringsForceField', points='@boxROI.indices', stiffness=1e12)

    #   tells the constraint solver how a constraint force maps back onto these
    #   DOFs. Must match the linear solver above.
    robot.addObject('LinearSolverConstraintCorrection')

    # ---------------------------------------------------------------- LAYER 3
    # THE ACTUATOR. A cavity is a CLOSED TRIANGULAR SURFACE living inside the
    # tet mesh, mapped onto it. SurfacePressureConstraint enforces one scalar
    # relation between the cavity volume and the pressure across that surface.
    #
    #   valueType=0 -> "pressure"     : you impose p, volume is the response
    #   valueType=1 -> "volumeGrowth" : you impose dV, pressure is the response
    #
    # Default is pressure. Note the shipped bunny demo uses valueType=1.
    cavity = robot.addChild('cavity')
    cavity.addObject('MeshOBJLoader', name='loader', filename=MESH + 'Hollow_Bunny_Body_Cavity.obj')
    cavity.addObject('MeshTopology', src='@loader', name='topo')
    cavity.addObject('MechanicalObject', name='cavity')
    cavity.addObject('SurfacePressureConstraint', name='pressure',
                     triangles='@topo.triangles',
                     value=[0.0],          # <-- the actuation input. RL writes here.
                     valueType=0,          # 0 = pressure  (try 1 = volumeGrowth)
                     drawPressure=True, drawScale=0.02)
    #   BarycentricMapping: the cavity surface has no physics of its own. It is
    #   glued into the tet mesh -- its nodes follow the tets, and the pressure
    #   forces it generates are pushed back onto the tets.
    cavity.addObject('BarycentricMapping', name='mapping', mapForces=False, mapMasses=False)

    # ---------------------------------------------------------------- LAYER 4
    # Rendering only. No physics. Extract the outer surface of the tet mesh.
    visu = robot.addChild('visu')
    visu.addObject('TriangleSetTopologyContainer', name='container')
    visu.addObject('TriangleSetTopologyModifier')
    visu.addObject('Tetra2TriangleTopologicalMapping', name='Mapping',
                   input='@../container', output='@container')
    visu.addObject('OglModel', color=[0.3, 0.2, 0.2, 0.6])
    visu.addObject('IdentityMapping')

    return root
