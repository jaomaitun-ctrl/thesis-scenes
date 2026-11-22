# Submission: deep RL navigation of a soft robot in the brain ventricles

| Requirement | File |
|---|---|
| Jupyter notebook with code and explanations | `ventricle_rl_ppo.ipynb` (executed; rebuild with `build_notebook.py`) |
| Written summary, 1–2 pages | `summary.pdf` (2 pages; rebuild with `build_summary.py`) |
| Video, 20 minutes | `VIDEO_SCRIPT.md` (timed script and shot list; recording and upload are still to do) |
| Reproducible environment and dependencies | `ENVIRONMENT.md`, `requirements-analysis.txt`, `requirements-rl.txt` |

`results/` holds the training logs (SB3 Monitor and progress CSVs), the trained models
(`ppo_p44_server_v1/model_190k.zip`, `ppo_p44_server_v2_shape/final_model.zip`) and recorded SOFA
rollouts (`rollouts/*.npz`, including 30-episode test sets). `figures/` is written by the notebook.

Quick start (no SOFA needed):

```bash
pip install -r requirements-analysis.txt
jupyter notebook ventricle_rl_ppo.ipynb
```

To rebuild the PDF, run the notebook first, then `pip install reportlab pypdfium2 && python build_summary.py`.

Patient data (patient 44's path and wall) are included in `results/`. Keep the repository private.
