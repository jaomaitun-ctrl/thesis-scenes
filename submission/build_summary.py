"""Build summary.pdf (2 pages) from figures/ and figures/eval_table.json (run the notebook first).
Needs reportlab.  AUTHOR can be set in the environment."""
import json
import os

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (Image, KeepTogether, Paragraph, SimpleDocTemplate, Spacer, Table,
                                TableStyle)

AUTHOR = os.environ.get('AUTHOR', '')
INK, MUTED, RULE = colors.HexColor('#0b0b0b'), colors.HexColor('#52514e'), colors.HexColor('#d8d7d2')

ss = getSampleStyleSheet()
body = ParagraphStyle('body', parent=ss['BodyText'], fontName='Helvetica', fontSize=8.9, leading=11.6,
                      textColor=INK, spaceAfter=4, alignment=TA_LEFT)
h1 = ParagraphStyle('h1', parent=body, fontName='Helvetica-Bold', fontSize=14.5, leading=17.5, spaceAfter=2)
sub = ParagraphStyle('sub', parent=body, fontSize=8.6, textColor=MUTED, spaceAfter=8)
h2 = ParagraphStyle('h2', parent=body, fontName='Helvetica-Bold', fontSize=10.2, leading=13, spaceBefore=5, spaceAfter=3)
cap = ParagraphStyle('cap', parent=body, fontSize=7.8, leading=9.6, textColor=MUTED, spaceAfter=6)
cell = ParagraphStyle('cell', parent=body, fontSize=7.8, leading=9.4, spaceAfter=0)
cellb = ParagraphStyle('cellb', parent=cell, fontName='Helvetica-Bold')
bullet = ParagraphStyle('bullet', parent=body, leftIndent=10, bulletIndent=2, spaceAfter=2)

P = lambda t, s=body: Paragraph(t, s)
B = lambda t: Paragraph(t, bullet, bulletText='•')


def table(rows, widths, header=True):
    data = [[Paragraph(str(c), cellb if (header and i == 0) else cell) for c in r] for i, r in enumerate(rows)]
    t = Table(data, colWidths=widths, hAlign='LEFT')
    t.setStyle(TableStyle([('VALIGN', (0, 0), (-1, -1), 'TOP'), ('LINEBELOW', (0, 0), (-1, 0), 0.6, INK),
                           ('LINEBELOW', (0, 1), (-1, -1), 0.3, RULE), ('TOPPADDING', (0, 0), (-1, -1), 2),
                           ('BOTTOMPADDING', (0, 0), (-1, -1), 2), ('LEFTPADDING', (0, 0), (-1, -1), 3),
                           ('RIGHTPADDING', (0, 0), (-1, -1), 3)]))
    return t


def fig(fn, width_cm):
    from reportlab.lib.utils import ImageReader
    w, h = ImageReader(fn).getSize()
    return Image(fn, width=width_cm * cm, height=width_cm * cm * h / w)


ev = json.load(open('figures/eval_table.json'))
v1, v2 = ev['rows']
W = A4[0] - 3.2 * cm
story = [
    P('Deep Reinforcement Learning for Soft-Robot Navigation in the Brain Ventricles', h1),
    P((AUTHOR + ' · ' if AUTHOR else '') + 'MEng, Biomedical Engineering, Mahidol University · '
      'project summary · code, notebook and logs: <font face="Courier">thesis-scenes/submission</font>', sub),

    P('1. Goal', h2),
    P('Endoscopic access to the lateral ventricles (e.g. for intraventricular haemorrhage) uses a straight '
      'rigid instrument inserted at Kocher\'s point. A soft continuum robot could follow the curved lumen '
      'instead, but steering it by hand is hard. The goal of this project is an <b>AI controller that '
      'steers a two-section pneumatic soft robot from the Kocher entry through the frontal horn to a target '
      '80 mm along a planned path</b>, keeping the whole body inside the ventricle and close to the path. '
      'The controller is learned with deep reinforcement learning (RL) in a physics simulation of the '
      'patient\'s anatomy.'),

    P('2. Data and how they are used', h2),
    table([['Data', 'Content', 'Role in the learning system'],
           ['Planned paths (40 patients × left/right)', 'MRI-derived centre paths, mm, LPS frame; start at the frontal-horn entry F',
            'All 80: offline robot design (section lengths 25 + 55 mm, curvature limit) and a nominal curvature schedule. '
            'Patient 44 right: reference trajectory and goal of the RL task'],
           ['Ventricle surface mesh (patient 44 R)', 'Closed STL, 2 500 triangles',
            'Simulator wall (contacts, clearance observations) and the failure test "robot outside the lumen"'],
           ['3D Slicer fiducials', 'Coronal suture (OC) and ventricle entry (F)',
            'Kocher point K and the rigid-sheath axis K → F (K lies 0.3 mm off the simulated axis)']],
          [3.6 * cm, 4.6 * cm, W - 8.2 * cm]),
    Spacer(1, 3),
    P('The data are not a supervised training set: the policy learns from <b>interaction</b> with a simulator '
      'built from them, so every training sample (state, action, reward) is generated online. One anatomy is '
      'used for training and testing; results therefore measure control quality, not cross-patient generalisation.'),

    P('3. Method', h2),
    P('<b>Simulator.</b> SOFA v26.06: beam-FEM rod (41 nodes) pushed out of a rigid sheath; frictional wall '
      'contact solved as a linear complementarity problem (Gauss-Seidel, tolerance 1e-6). The tolerance matters: '
      'an under-converged solver sends the robot into a different ventricle branch at a bifurcation. Chamber '
      'pressures come from a Yeoh hyperelastic model of the actuators.'),
    P('<b>MDP.</b> One RL step = 0.1 s. <i>Observation</i> (35): tip-to-goal and tip-to-path vectors, tip '
      'direction, depth, curvature commands, 6 body points, wall clearances, contact force. <i>Action</i> (5, '
      'continuous): insertion 0–1 mm (forward only) and a residual change of each bending-curvature command. '
      'This is <b>residual RL</b>: the policy corrects a depth-scaled nominal schedule fitted to the path, '
      'which alone ends 45 mm off target in the wrong branch. <i>Reward</i>: '
      'r = 0.2·Δdepth − 0.002·e<sub>tip</sub> − w<sub>shape</sub>·e<sub>shape</sub> − 20·force − 0.01, '
      '+10·exp(−e<sub>goal</sub>/10) at full insertion, −10 and termination when leaving the lumen. '
      'Success = full insertion with goal error ≤ 3 mm.'),
    P('<b>Learning algorithm.</b> Proximal Policy Optimization (stable-baselines3): an actor-critic with a '
      'Gaussian policy and a value network (MLP 128-128 each). PPO maximises the clipped surrogate objective '
      'E[min(ρ<sub>t</sub>A<sub>t</sub>, clip(ρ<sub>t</sub>, 1±0.2)A<sub>t</sub>)], ρ<sub>t</sub> = π<sub>θ</sub>/π<sub>old</sub>, '
      'so each update stays close to the data-collecting policy; advantages A<sub>t</sub> use GAE (γ = 0.99, '
      'λ = 0.95). Rollouts of 256 steps × 5 parallel simulators, 10 epochs per update, batch 256, lr 3e-4, '
      'rewards normalised online. Two reward designs are compared: <b>v1</b> (w<sub>shape</sub> = 0, tip only) '
      'and <b>v2</b> (w<sub>shape</sub> = 0.01 1/mm, reward shaping on the body-to-path distance).'),
    P('<b>Engineering.</b> Five SOFA environments run in separate processes (the scene is configured at import). '
      'Each reset rebuilds the scene and the worker processes leak memory; a first 200k-step run was killed by '
      'the out-of-memory killer. Training now runs one rollout per update call and respawns the simulator '
      'workers past 1.5 GB, keeping policy, optimiser and normaliser state, so no steps are lost; a supervisor '
      'resumes from the newest checkpoint after a crash. Training ran on a 6-core Linux server (~60 steps/s, '
      '~1 h per 200k steps; GPU not needed).'),
]

story += [
    KeepTogether([P('4. Results', h2),
                  fig('figures/learning_curves.png', W / cm),
                  P('Figure 1. Training curves (10k-step bins, stochastic policy). Both reward designs escape the wrong '
                    'branch within ~40k steps. Only v2 reduces the body-to-path distance; it also reaches the goal more accurately.', cap)]),
    table([['Policy (200k steps)', 'Success, last 20k training steps', 'Goal error, deterministic (mm)',
            'Body off path, deterministic (mm)', 'Success, 30 test episodes', 'Goal error, test median [p90] (mm)',
            'Body off path, test (mm)'],
           ['v1 · tip reward', v1[1], v1[4], v1[5], v1[6], v1[7], v1[8]],
           ['v2 · tip + shape', v2[1], v2[4], v2[5], v2[6], v2[7], v2[8]],
           ['nominal schedule (no RL)', '—', '45.35', '8.26', '0%', '—', '—']],
          [3.2 * cm] + [(W - 3.2 * cm) / 6] * 6),
    P('Table 1. Test episodes: 30 new episodes per policy with sampled actions (seed 7), same anatomy. '
      'Deterministic = the mean action.', cap),
    fig('figures/trajectories.png', W / cm),
    P('Figure 2. Recorded SOFA insertions (deterministic policies): tip paths (thin) and final robot bodies (wide bands) '
      'against the planned path. In the side view the v1 body bows ~6 mm off the path; v2 stays within ~3 mm.', cap),
    P('<b>Findings.</b> (1) Residual RL on a physically fitted schedule makes the task learnable in ~10<super>5</super> '
      'simulator steps; both policies succeed in ≥ 90 % of test episodes with no wall exits. (2) Reward shaping '
      'with a body term is the most effective design change: it <b>halves the body-to-path distance</b> '
      '(4.6 → 2.5 mm) <b>and improves the task metric</b> (goal error 1.2 → 0.7 mm deterministic; 90 % → 97 % '
      'success), because a body on the path keeps the tip in a reachable configuration. (3) The learned '
      'policy runs live in SOFA\'s own GUI and reproduces the training rollout exactly (0.67 mm).'),

    P('5. Limitations and next steps', h2),
    B('Single anatomy, rigid wall, no sensor noise or domain randomisation, small test set (30 episodes per policy).'),
    B('Next: train on many of the 80 patient paths and test on held-out patients; randomise wall stiffness, '
      'friction, actuator gain and registration error; add the history-dependent (poroviscoelastic) wall and '
      'compare policies with and without contact-history information; non-learned baselines.'),

    P('6. Reproducibility', h2),
    P('Repository <font face="Courier">thesis-scenes</font> (branch <font face="Courier">rl-shape-reward</font>): '
      'environment <font face="Courier">rl/ventricle_env.py</font>, training <font face="Courier">rl/train_ppo.py</font>, '
      '<font face="Courier">rl/train_supervised.sh</font>; notebook <font face="Courier">submission/ventricle_rl_ppo.ipynb</font> '
      'regenerates every figure and table from the included logs and rollouts without SOFA. '
      '<font face="Courier">submission/ENVIRONMENT.md</font> pins SOFA (commit a4d4962a, v26.06), Python 3.12.13, '
      'torch 2.7.1, stable-baselines3 2.6.0 and gymnasium 1.1.1, with build and run commands. Patient data are private.'),
]

doc = SimpleDocTemplate('summary.pdf', pagesize=A4, leftMargin=1.6 * cm, rightMargin=1.6 * cm,
                        topMargin=1.4 * cm, bottomMargin=1.3 * cm, title='Deep RL soft-robot navigation - summary',
                        author=AUTHOR or 'MEng project')
doc.build(story)
import pypdfium2  # noqa: E402
print("pages:", len(pypdfium2.PdfDocument("summary.pdf")))
