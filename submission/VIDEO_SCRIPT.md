# 20-minute video: script and shot list

Record the screen with QuickTime (File > New Screen Recording) or OBS, with a microphone.
Times are targets; the total is 20:00. **[Show]** says what is on screen, and the rest is what to say.

| # | Time | Section |
|---|---|---|
| 1 | 0:00–1:30 | Problem and goal |
| 2 | 1:30–4:00 | Data |
| 3 | 4:00–7:30 | RL formulation |
| 4 | 7:30–10:00 | Simulator and engineering |
| 5 | 10:00–11:30 | Notebook walkthrough: setup and data |
| 6 | 11:30–14:30 | Results |
| 7 | 14:30–18:00 | Gradio demo and live SOFA |
| 8 | 18:00–19:30 | Limitations and next steps |
| 9 | 19:30–20:00 | Reproducibility and close |

---

## 1. Problem and goal (0:00–1:30)
**[Show]** `summary.pdf`, page 1 (title and section 1).
- Ventricular endoscopy enters at Kocher's point with a straight rigid instrument; the ventricle is curved.
- A pneumatic soft continuum robot could follow the lumen, but it is hard to steer by hand.
- Goal: an AI controller that steers the robot from the Kocher entry to a target 80 mm along a
  planned path, keeping the body inside the ventricle and close to the path.
- Approach: deep reinforcement learning (PPO) in a SOFA physics simulation of a real patient's anatomy.

## 2. Data (1:30–4:00)
**[Show]** notebook Section 2 (data table and the printed path statistics), then the web viewer
`http://10.33.99.110:8080` rotated to show the ventricle, K, F and the goal.
- Dataset: MRI-derived planned paths for 40 patients × left/right; ventricle meshes; 3D Slicer fiducials.
- All 80 paths were used offline to size the robot (25 + 55 mm sections) and to fit a nominal curvature schedule.
- Patient 44 right is the RL task: its path is the reference, its mesh is the wall, and its fiducials give
  Kocher point K and the sheath axis (K lies 0.3 mm from the simulated axis, 40.9 mm before F).
- Key point: this is not supervised learning. The data define a simulator, and the policy generates its
  own experience by interacting with it.
- Say clearly: one anatomy is used for training and testing (a limitation, covered in section 8).

## 3. RL formulation (4:00–7:30)
**[Show]** notebook Section 3 (MDP and the reward equation).
- RL loop: the agent observes a state, acts, and receives a reward; it learns to maximise expected return.
- Observation (35): tip-to-goal and tip-to-path vectors, tip direction, depth, curvature commands,
  6 body points, wall clearance, contact force.
- Action (5): insertion 0–1 mm plus residual curvature changes. **Residual RL**: the policy corrects the
  fitted schedule instead of learning from scratch. The schedule alone ends 45 mm off target in the wrong branch.
- Reward terms, one by one: progress, tip tracking, shape (v2 only), contact force, time,
  terminal bonus, failure penalty. Success means goal error ≤ 3 mm.
- PPO: actor-critic; clipped surrogate objective (ratio clipped to 1 ± 0.2) keeps each update close to
  the policy that collected the data; GAE advantages (γ 0.99, λ 0.95); rewards normalised online.
- The experiment: v1 rewards the tip only; v2 adds w = 0.01 × distance of the body from the path (reward shaping).

## 4. Simulator and engineering (7:30–10:00)
**[Show]** notebook Section 4; briefly `rl/ventricle_env.py` (`step`, `apply_action`) and `rl/train_ppo.py`.
- SOFA beam-FEM rod, rigid sheath, wall contact as an LCP. The solver tolerance changes which branch the
  robot enters, so simulator settings are part of the task definition.
- 5 simulators in parallel processes, about 60 steps/s on a 6-core server; 200k steps take about 1 hour.
- Failure story: the first run was killed by the out-of-memory killer at 198.4k steps because the
  workers leak memory. Fix: respawn the workers once they pass 1.5 GB, keeping the policy and optimiser
  state, so no steps are lost; a supervisor resumes from the last checkpoint after a crash.

## 5. Notebook walkthrough: setup and data (10:00–11:30)
**[Show]** scroll the notebook from the top: environment cell, versions, `RUN_SOFA = False`, data cell.
- Every figure regenerates from the logs in `results/`; training cells are guarded by `RUN_SOFA`.

## 6. Results (11:30–14:30)
**[Show]** notebook 6.1 (learning curves), 6.2 (table), 6.3 (trajectories), 6.4 (PPO diagnostics).
- Both policies escape the wrong branch within about 40k steps.
- v1 keeps improving the tip, but its body stays about 4.7 mm off the path for the whole run.
- v2 halves the body deviation (4.6 → 2.5 mm deterministic) and also improves the tip
  (1.2 → 0.7 mm; 90 % → 97 % success on 30 test episodes). Explain why: a body on the path keeps
  the tip where the goal is reachable.
- Diagnostics: exploration (action std) decreases smoothly, and value explained variance is about 0.97, so
  training is healthy.

## 7. Gradio demo and live SOFA (14:30–18:00)
**[Show]** the Gradio app (`python app.py`, or on the server `http://10.33.99.110:7860`):
1. **Replay** tab: tick v2 and v1, press ▶ Play, and drag the step slider to show the robot leaving the
   sheath at F and the v1 body bowing off the path; read the table (0.67 vs 1.21 mm; 2.4 vs 4.5 mm body).
   Tick "Nominal schedule" to show the wrong-branch failure.
2. **Results** tab: the same curves and table as the notebook.
3. **Live SOFA** tab: run v2 deterministic (0.67 mm on the server), then sampled actions with two seeds,
   to show the policy is robust and not just replaying one trajectory. Mention that each run is a new
   physics simulation, taking 10–60 s.
4. Optional: runSofa with `rl/policy_scene.py` (SOFA's own GUI; noVNC `http://10.33.99.110:6080`).

## 8. Limitations and next steps (18:00–19:30)
**[Show]** `summary.pdf` section 5.
- One anatomy, rigid wall, no noise or domain randomisation, 30 test episodes per policy.
- Next: train on many patients with held-out test patients; randomise wall stiffness, friction, actuator gain
  and registration; history-dependent poroviscoelastic wall; compare policies with and without
  contact-history information; non-learned baselines.

## 9. Reproducibility and close (19:30–20:00)
**[Show]** `ENVIRONMENT.md`.
- Three levels: analysis from logs (no SOFA), evaluation with the saved models, retraining.
  All versions are pinned. Thank the audience.

---
**Before recording:** start the Gradio app (tmux session `gradio` on the server), the viewers (`rl/server/sofa_gui.sh start` on the server for noVNC; the web
viewer runs in tmux session `web`), open the notebook already executed, and close unrelated windows.
Do not show the VNC password on screen.
