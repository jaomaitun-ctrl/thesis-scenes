# Thesis: deep RL navigation of a soft robot in the brain ventricles

This is the MEng thesis project (Biomedical Engineering, Mahidol University), titled *Deep
Reinforcement Learning for Adaptive Navigation of Soft Robots in Dynamic Ventricular
Neurosurgery*.

A two-section pneumatic soft continuum robot enters at Kocher's point. A PPO policy steers it
through the lateral ventricle, in a SOFA simulation of MRI-derived patient anatomy.

## Layout

| Folder | What it is |
|---|---|
| [`prelim/`](prelim/) | **Preliminary results.** Patient 44, rigid ventricle wall, PPO v1/v2, and the course submission (notebook, summary, Gradio demo). Self-contained: its own `rl/`, `tools/`, `data/`, `poses/` and `walls/`. See [prelim/README.md](prelim/README.md). |
| [`plugins/ViscoOgden/`](plugins/ViscoOgden/) | SOFA plugin: finite-strain viscoelastic Ogden tissue with internal relaxation variables. Shared by the thesis work. |
| *(next)* | Main thesis work: a deformable, history-dependent ventricle wall, several patients, domain randomisation, and the policy comparison (no contact history, true internal variables, estimated internal variables). |

The preliminary work exactly as submitted is the git tag `prelim-submission`, from before
this restructure:

```bash
git checkout prelim-submission
```

## Running

Everything runs on the SOFA v26.06 pixi build in `~/Developer/sofa`. The ViscoOgden plugin
is installed into that environment.

- **Preliminary scenes and RL:** run them from `prelim/`, for example
  `prelim/rl/run.sh rl/train_ppo.py ...`. Paths inside `prelim/` are relative to that folder.
- **Building or reinstalling the plugin:** see [plugins/ViscoOgden/README.md](plugins/ViscoOgden/README.md).
- **SOFA.app** (macOS launcher): opens `.py`, `.scn` and `.xml` scenes. Its Python path
  includes `prelim/`, `prelim/tools` and `prelim/.pydeps`.

The lab server keeps its own copy at `/srv/train/thesis-scenes`, which still has the old flat
layout of `prelim/`.
