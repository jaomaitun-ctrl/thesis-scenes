# Environment and reproduction

Three levels of reproduction, from cheapest to full:

| Level | What it reproduces | Needs |
|---|---|---|
| **A** | Every table and figure in the notebook and summary, from the saved training logs and rollouts in `results/` | Python ≥ 3.10 + `requirements-analysis.txt` |
| **B** | The evaluation rollouts (`results/rollouts/*.npz`) from the saved models | SOFA build (below) + `requirements-rl.txt` |
| **C** | Training both policies from scratch | Same as B; ~1 h per run on 6 CPU cores |

## Level A (analysis only)

```bash
cd thesis-scenes/submission
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements-analysis.txt
jupyter notebook ventricle_rl_ppo.ipynb      # Run All; RUN_SOFA stays False
```

## Exact software used for the reported runs

| Component | Version |
|---|---|
| OS (training server) | Ubuntu 24.04.5 LTS, 6 CPU cores, 11 GB RAM (GPU not used) |
| OS (development) | macOS 14 (Apple silicon) |
| SOFA | v26.06, `sofa-framework/sofa` commit `a4d4962aed1e37cd9027fbb4b9070127ad06f6de` |
| SOFA plugin pins | `applications/plugins/CMakeLists.txt`: SoftRobots.Inverse and Cosserat `GIT_REF v26.06` (instead of `main`) |
| Build | pixi 0.84.2, environment `supported-plugins` (Release); conda-forge toolchain |
| Python (SOFA's) | 3.12.13 (conda-forge), numpy 2.5.0 |
| RL | torch 2.7.1+cpu, stable-baselines3 2.6.0, gymnasium 1.1.1 |
| Scene components | SOFA core only (`Sofa.Component.*`): BeamFEMForceField rod, FreeMotionAnimationLoop + BlockGaussSeidelConstraintSolver (LCP tol 1e-6), FixedProjectiveConstraint sheath |

## Build SOFA (levels B and C)

```bash
git clone https://github.com/sofa-framework/sofa.git && cd sofa
git checkout a4d4962aed1e37cd9027fbb4b9070127ad06f6de
# pin the two fetched plugins to v26.06 (see table above), then:
curl -fsSL https://pixi.sh/install.sh | bash
CMAKE_BUILD_PARALLEL_LEVEL=5 pixi run -e supported-plugins install     # 1-2 h on 6 cores
```

Then, in `thesis-scenes`, install the RL packages into `.pydeps` as described at the top of
`requirements-rl.txt`. Run scripts with SOFA's interpreter:

- macOS: `rl/run.sh <script>` (uses `python-sofa`; plain `python` crashes when importing Sofa)
- Linux: `rl/run_server.sh <script>` (plain env python; edit the SOFA path at the top)

## Data

Patient data come from the senior project's MRI-derived dataset (40 patients × left/right):
planned paths `paths/<patient>_<side>.csv`, ventricle meshes `ventricles/<patient>_<side>.stl`,
and Slicer fiducials `kochers/<patient>/` (coronal suture `OC*`, ventricle entry `F*`).
This study trains and tests on **patient 44, right side** only. That patient's files are in
the repository, shared with the data owners' permission, in the dataset's own layout under
`data/patients_dataset/` (path, closed ventricle mesh, Slicer fiducials). The collision wall with
the entry hole is `walls/patient44_right_entryhole.stl` (made by `tools/ventricle_wall.py`), and
`poses/design_patient44_right_*.json` refer to the path relative to the repository root.
The other 79 patients' data are not included; ask the project authors for access.

## Commands that produced the results

```bash
# v1: tip-only reward (crashed out of memory at 198.4k steps; checkpoint 190k is used)
rl/run_server.sh rl/train_ppo.py --name ppo_p44_server_v1 --steps 200000 --envs 5 \
    --n-steps 256 --ckpt 10000 --max-episode-steps 250
# v2: + shape term; current memory-aware launcher (v2 itself ran as 2 x 100k chunks)
rl/train_supervised.sh ppo_p44_server_v2_shape 200000 --envs 5 --n-steps 256 \
    --ckpt 10000 --max-episode-steps 250 --w-shape 0.01
# evaluation rollouts
rl/run_server.sh rl/rollout.py --ckpt <model.zip> --deterministic --record-rod --out v2_det.npz
rl/run_server.sh rl/rollout.py --ckpt <model.zip> --episodes 30 --seed 7 --out eval30_v2.npz
# watch a policy drive SOFA live
runSofa -l SofaPython3 -g imgui rl/policy_scene.py
```

Seeds: PPO `seed=0`; evaluation `--seed 7`. SOFA itself is deterministic for a given action
sequence. Sampled-action results vary with the seed; the deterministic policy rollout does not.
