"""
Same cavity, but driven from Python instead of a hard-coded number.

This file exists to show you ONE thing: the exact line where a reinforcement
learning action enters a SOFA simulation. Everything SofaGym does is a
generalisation of `self.pressure.value.value = [p]` below.

Run:
    cd ~/sofa
    pixi run -e supported-plugins runSofa thesis-scenes/pneumatic_02_ramp.py
Then press Animate. Watch the terminal, not just the viewport.
"""
import os
import sys

import Sofa
import Sofa.Core

# runSofa does not guarantee the scene's own directory is importable.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pneumatic_01_minimal import createScene as buildBaseScene

P_START = 0.0
P_STEP = 0.05     # per timestep
P_MAX = 12.0      # tune this: too high and the solver will blow up
REPORT_EVERY = 20


class PressureRamp(Sofa.Core.Controller):
    """A SOFA controller = a Python object that gets called every timestep.

    onAnimateBeginEvent fires BEFORE the physics solve, which is exactly where
    an RL agent's action must be written. onAnimateEndEvent fires after, which
    is where you would read the observation and compute the reward.
    """

    def __init__(self, *a, **kw):
        Sofa.Core.Controller.__init__(self, *a, **kw)
        self.node = kw['node']
        self.p = P_START
        self.k = 0

    def onAnimateBeginEvent(self, event):
        # ---- THIS IS THE ACTION INTERFACE -------------------------------
        # For your robot, `value` becomes a vector with one entry per chamber.
        # a_t = [dP_chamber1, dP_chamber2, ...] -> integrate -> write here.
        self.p = min(self.p + P_STEP, P_MAX)
        self.node.robot.cavity.pressure.value.value = [self.p]
        # -----------------------------------------------------------------

    def onAnimateEndEvent(self, event):
        # ---- THIS IS THE OBSERVATION INTERFACE --------------------------
        self.k += 1
        if self.k % REPORT_EVERY:
            return
        c = self.node.robot.cavity.pressure
        tip = self.node.robot.tetras.position.value[0]   # placeholder "tip" node
        print(f'step {self.k:5d}  p = {self.p:7.3f}   '
              f'V0 = {c.initialCavityVolume.value:10.3f}   '
              f'V = {c.cavityVolume.value:10.3f}   '
              f'node0 = [{tip[0]:7.3f} {tip[1]:7.3f} {tip[2]:7.3f}]')
        # -----------------------------------------------------------------


def createScene(root):
    buildBaseScene(root)
    root.addObject(PressureRamp(name='ramp', node=root))
    return root
