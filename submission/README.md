# Submission: deep RL navigation of a soft robot in the brain ventricles

| Requirement | File |
|---|---|
| Jupyter notebook with code and explanations | `ventricle_rl_ppo.ipynb` (executed; rebuild with `build_notebook.py`) |
| Written summary, 1–2 pages | `summary.pdf` (2 pages; rebuild with `build_summary.py`) |
| Gradio demo | `app.py` (Replay, Results, Live SOFA); `requirements-app.txt` |
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

Patient 44's planned path and ventricle wall are included in `results/` and in the figures, shared with the data owners' permission. The full 80-path dataset is not included.
