"""Plot the validation runs (needs the .npz files written by the two tests).
    python3 plugins/ViscoOgden/tests/plot_validation.py  ->  tests/validation.png
"""
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
h = np.load(os.path.join(HERE, 'homogeneous_test.npz'))
c = np.load(os.path.join(HERE, 'creep_test.npz'))
SOFA, REF = '#d55e00', '#222222'

fig, ax = plt.subplots(1, 3, figsize=(13, 3.8), constrained_layout=True)
a = ax[0]
a.plot(h['step_t'], h['step_sofa'][:, 0, 0], color=SOFA, lw=2.5, label='SOFA ViscoOgden')
a.plot(h['step_t'], h['step_ref'][:, 0, 0], color=REF, ls='--', lw=1, label='NumPy reference')
a.set(title='Step stretch 1.25 held: stress relaxation', xlabel='time [s]', ylabel=r'$S_{11}$ [Pa]')
a.legend(frameon=False)

a = ax[1]
last = h['step_t'] >= 2.0
a.plot(h['shear_g'][last], h['shear_sofa'][last, 0, 1], color=SOFA, lw=2.5, label='viscoelastic')
a.plot(h['shear_g'][last], h['shear_ref'][last, 0, 1], color=REF, ls='--', lw=1, label='reference')
a.plot(h['shear_g'][last], h['shear_el'][last, 0, 1], color='#0072b2', lw=1.5, label='viscous = 0')
a.set(title='Cyclic shear (1 Hz): hysteresis', xlabel=r'shear $\gamma$', ylabel=r'$S_{12}$ [Pa]')
a.legend(frameon=False)

a = ax[2]
a.plot(c['t'], -c['instant'], color='#0072b2', lw=1.2, label=r'hyperelastic, $(1+\Sigma\beta)\mu$')
a.plot(c['t'], -c['longterm'], color='#009e73', lw=1.2, label=r'hyperelastic, long-term $\mu$')
a.plot(c['t'], -c['visco'], color=SOFA, lw=2.5, label='viscoelastic (creep)')
a.set(title='Column under 200 Pa: creep', xlabel='time [s]', ylabel='top compression [mm]')
a.legend(frameon=False)
for a in ax:
    a.spines[['top', 'right']].set_visible(False)
fig.savefig(os.path.join(HERE, 'validation.png'), dpi=130)
print('wrote', os.path.join(HERE, 'validation.png'))
