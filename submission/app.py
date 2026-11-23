"""
Gradio demo: soft-robot navigation in the brain ventricles with PPO (patient 44, right).

    pip install -r requirements-app.txt
    python app.py                      # http://127.0.0.1:7860 ; add --share for a public link

Tabs
  Replay   3D replay of recorded SOFA insertions (v2, v1, nominal schedule): pick a policy and
           drag the step slider or press Play. Works anywhere; no SOFA needed.
  Results  training curves and the evaluation table from the notebook.
  Live     runs SOFA now with a chosen policy (deterministic or sampled, any seed) and shows the
           result. Needs the SOFA build (submission/ENVIRONMENT.md); hidden otherwise.
"""
import html
import json
import os
import shutil
import struct
import subprocess
import sys
import tempfile

import gradio as gr
import numpy as np
import plotly.graph_objects as go

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
RES = os.path.join(HERE, 'results')
DATA = os.path.join(REPO, 'data', 'patients_dataset')
RUNS = {  # label -> (recorded rollout, model, colour)
    'PPO v2 · tip + shape reward': ('unity_v2.npz', 'ppo_p44_server_v2_shape/final_model.zip', '#eb6834'),
    'PPO v1 · tip reward': ('unity_v1.npz', 'ppo_p44_server_v1/model_190k.zip', '#2a78d6'),
    'Nominal schedule · no policy': ('unity_baseline.npz', None, '#8a8a86'),
}
PATH_INK, WALL_RGB = '#0b0b0b', '#3fb6d8'


# ---------------------------------------------------------------- data
def read_stl(fn):
    b = open(fn, 'rb').read()
    n = struct.unpack('<I', b[80:84])[0]
    a = np.frombuffer(b[84:84 + 50 * n], dtype=np.dtype([('n', '<f4', 3), ('v', '<f4', (3, 3)), ('a', '<u2')]))
    tri = a['v'].reshape(-1, 3).astype(float)
    uniq, inv = np.unique(np.round(tri, 3), axis=0, return_inverse=True)
    return uniq, inv.reshape(-1, 3)


def read_fcsv_last(fn):
    rows = [l for l in open(fn).read().splitlines() if l and not l.startswith('#')]
    return np.array([float(v) for v in rows[-1].split(',')[1:4]])


VERTS, FACES = read_stl(os.path.join(DATA, 'ventricles', 'patient44_right.stl'))
KDIR = os.path.join(DATA, 'kochers', 'patient44')
K = sorted([read_fcsv_last(os.path.join(KDIR, f)) for f in os.listdir(KDIR) if f.startswith('OC')], key=lambda p: p[0])[0]


def load(fn):
    z = np.load(fn)
    rods = z['rods_0'] if 'rods_0' in z else None
    return dict(rods=rods, depth=z['depth_0'] if 'depth_0' in z else None, path=z['path'], goal=z['goal'],
                F=z['F'], d=z['d'] if 'd' in z else None, L=float(z['L_rigid_mm']) if 'L_rigid_mm' in z else 80.0,
                goal_err=float(z['goal_err_0']))


REC = {k: load(os.path.join(RES, 'rollouts', f)) for k, (f, _, _) in RUNS.items()}


def metrics(rec, step):
    rods, path = rec['rods'], rec['path']
    tip = rods[step, -1]
    s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(rods[step], axis=0), axis=1))]
    ins = rods[step][s >= s[-1] - rec['depth'][step] + 1e-6]
    dev = np.min(np.linalg.norm(ins[:, None] - path[None], axis=2), axis=1).mean() if len(ins) else 0.0
    return rec['depth'][step], np.linalg.norm(tip - rec['goal']), dev


# ---------------------------------------------------------------- 3D figure
def figure(labels, step=None, animate=True):
    ref = REC[labels[0]] if labels else REC[next(iter(REC))]
    xyz = lambda p: dict(x=p[..., 0], y=p[..., 1], z=p[..., 2])
    tr = [go.Mesh3d(**xyz(VERTS), i=FACES[:, 0], j=FACES[:, 1], k=FACES[:, 2], color=WALL_RGB, opacity=0.16,
                    name='ventricle wall', hoverinfo='skip', showlegend=True),
          go.Scatter3d(**xyz(ref['path']), mode='lines', line=dict(color=PATH_INK, width=5), name='planned path')]
    base = ref['F'] - ref['d'] * ref['L'] if ref['d'] is not None else ref['F']
    tr.append(go.Scatter3d(**xyz(np.array([base, ref['F']])), mode='lines', line=dict(color='#b8c0cc', width=14),
                           opacity=0.5, name='rigid sheath'))
    for p, name, col, sym in [(K, 'Kocher point K', '#e0a800', 'diamond'), (ref['F'], 'entry F', '#1baf7a', 'square'),
                              (ref['goal'], 'goal (80 mm)', '#e34948', 'circle')]:
        tr.append(go.Scatter3d(x=[p[0]], y=[p[1]], z=[p[2]], mode='markers', marker=dict(size=6, color=col, symbol=sym),
                               name=name))
    n_static = len(tr)
    nf = max(len(REC[l]['rods']) for l in labels) if labels else 1
    f0 = nf - 1 if step is None else min(step, nf - 1)

    def robot(l, f):
        r = REC[l]['rods']
        f = min(f, len(r) - 1)
        col = RUNS[l][2]
        return [go.Scatter3d(**xyz(r[f]), mode='lines', line=dict(color=col, width=12), name=l),
                go.Scatter3d(**xyz(r[:f + 1, -1]), mode='lines', line=dict(color=col, width=3), opacity=0.7,
                             name=l + ' · tip trail', showlegend=False)]

    for l in labels:
        tr += robot(l, f0)
    fig = go.Figure(tr)
    if animate and labels:
        idx = list(range(n_static, len(tr)))
        fig.frames = [go.Frame(data=[t for l in labels for t in robot(l, f)], traces=idx, name=str(f))
                      for f in range(nf)]
        fig.update_layout(
            updatemenus=[dict(type='buttons', x=0.0, y=0.0, xanchor='left', yanchor='bottom', direction='left',
                              buttons=[dict(label='▶ Play', method='animate',
                                            args=[None, dict(frame=dict(duration=80, redraw=True), fromcurrent=False,
                                                             transition=dict(duration=0))]),
                                       dict(label='❚❚', method='animate',
                                            args=[[None], dict(frame=dict(duration=0), mode='immediate')])])],
            sliders=[dict(active=f0, x=0.12, len=0.88, y=0.0, currentvalue=dict(prefix='RL step '),
                          steps=[dict(method='animate', label=str(f),
                                      args=[[str(f)], dict(mode='immediate', frame=dict(duration=0, redraw=True))])
                                 for f in range(nf)])])
    fig.update_layout(height=620, margin=dict(l=0, r=0, t=10, b=0), legend=dict(x=0.01, y=0.99),
                      scene=dict(aspectmode='data', xaxis_title='L (mm)', yaxis_title='P (mm)', zaxis_title='S (mm)',
                                 camera=dict(eye=dict(x=-1.4, y=-1.2, z=0.8))))
    return fig


def as_html(fig):
    """gr.Plot drops Plotly animation frames, so Play would do nothing: render the figure as a
    self-contained page in an iframe, where Plotly's own Play button and slider work."""
    page = fig.to_html(include_plotlyjs='cdn', full_html=True, auto_play=False, config=dict(displaylogo=False))
    return (f'<iframe srcdoc="{html.escape(page, quote=True)}" '
            f'style="width:100%;height:{fig.layout.height + 20}px;border:0"></iframe>')


def replay(labels):
    labels = labels or []
    rows = []
    for l in labels:
        rec = REC[l]
        dep, tg, dev = metrics(rec, len(rec['rods']) - 1)
        rows.append([l, f'{dep:.0f}', f'{rec["goal_err"]:.2f}', f'{dev:.2f}',
                     'yes' if rec['goal_err'] <= 3 else 'no'])
    return as_html(figure(labels)), rows


# ---------------------------------------------------------------- live SOFA
def sofa_runner():
    if os.path.exists('/srv/train/sofa/.pixi/envs/supported-plugins/bin/python') and \
            os.path.exists(os.path.join(REPO, 'rl', 'run_server.sh')):
        return os.path.join(REPO, 'rl', 'run_server.sh')
    mac = os.path.join(os.path.expanduser('~'), 'Developer', 'sofa', '.pixi', 'envs', 'supported-plugins', 'bin', 'python-sofa')
    if os.path.exists(mac):
        return os.path.join(REPO, 'rl', 'run.sh')
    return None


RUNNER = sofa_runner()


def live(label, mode, seed, progress=gr.Progress()):
    model = RUNS[label][1]
    out = os.path.join(tempfile.mkdtemp(), 'live.npz')
    args = [RUNNER, 'rl/rollout.py', '--record-rod', '--out', out, '--seed', str(int(seed))]
    if model is None:
        args += ['--baseline']
    else:
        args += ['--ckpt', os.path.join(RES, model)]
        args += ['--deterministic'] if mode.startswith('Deterministic') else ['--episodes', '1']
    progress(0.1, desc='SOFA running (about 10–60 s)…')
    p = subprocess.run(args, cwd=REPO, capture_output=True, text=True, timeout=900)
    if not os.path.exists(out):
        raise gr.Error('SOFA run failed: ' + (p.stdout + p.stderr)[-600:])
    REC['live'] = load(out)
    RUNS['live'] = (None, None, RUNS[label][2])
    rec = REC['live']
    dep, tg, dev = metrics(rec, len(rec['rods']) - 1)
    shutil.rmtree(os.path.dirname(out), ignore_errors=True)
    summary = (f'**{label}**, {mode.lower()}, seed {int(seed)}: {len(rec["rods"]) - 1} RL steps, inserted {dep:.0f} mm, '
               f'goal error **{rec["goal_err"]:.2f} mm** ({"success" if rec["goal_err"] <= 3 else "no success"}), '
               f'robot body {dev:.2f} mm from the path.')
    return as_html(figure(['live'])), summary


# ---------------------------------------------------------------- UI
ev = json.load(open(os.path.join(HERE, 'figures', 'eval_table.json')))
with gr.Blocks(title='Soft-robot ventricle navigation (PPO + SOFA)') as demo:
    gr.Markdown(
        '# Soft-robot navigation in the brain ventricles with deep RL\n'
        'A two-section pneumatic soft robot enters at Kocher\'s point (sheath along K → F) and is steered by a PPO '
        'policy to a target 80 mm along the planned path of patient 44 (right lateral ventricle), simulated in SOFA. '
        'Compare a reward on the tip only (**v1**) with one that also scores the robot body\'s distance from the path '
        '(**v2**).')
    with gr.Tab('Replay'):
        pick = gr.CheckboxGroup(list(RUNS), value=list(RUNS)[:2], label='Insertions to show (recorded in SOFA, deterministic policies)')
        plot = gr.HTML()
        table = gr.Dataframe(headers=['run', 'inserted (mm)', 'goal error (mm)', 'body off path (mm)', 'success (≤ 3 mm)'],
                             interactive=False)
        pick.change(replay, pick, [plot, table])
        demo.load(replay, pick, [plot, table])
    with gr.Tab('Results'):
        gr.Image(os.path.join(HERE, 'figures', 'learning_curves.png'), label='Training curves', show_label=True)
        gr.Dataframe(value=ev['rows'], headers=ev['header'], interactive=False)
        gr.Markdown('train = last 20k training steps; det = deterministic policy; eval = 30 new episodes with '
                    'sampled actions (seed 7). Errors in mm.')
        gr.Image(os.path.join(HERE, 'figures', 'trajectories.png'), label='Trajectories')
    with gr.Tab('Live SOFA', visible=RUNNER is not None):
        gr.Markdown('Runs a fresh SOFA simulation with the chosen policy on this machine and replays it. '
                    'Sampled actions with different seeds show how robust the policy is.')
        with gr.Row():
            lab = gr.Dropdown(list(RUNS), value=list(RUNS)[0], label='Controller')
            mode = gr.Radio(['Deterministic (mean action)', 'Sampled actions'], value='Deterministic (mean action)', label='Actions')
            seed = gr.Number(value=1, precision=0, label='Seed')
            go_btn = gr.Button('Run SOFA', variant='primary')
        out_md = gr.Markdown()
        out_plot = gr.HTML()
        go_btn.click(live, [lab, mode, seed], [out_plot, out_md])

if __name__ == '__main__':
    demo.launch(server_name=os.environ.get('GRADIO_SERVER_NAME', '127.0.0.1'),
                server_port=int(os.environ.get('GRADIO_SERVER_PORT', 7860)), share='--share' in sys.argv)
