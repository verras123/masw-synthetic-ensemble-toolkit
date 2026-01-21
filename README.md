# MASW Synthetic Ensemble Toolkit

Public, self-contained **synthetic MASW** demo to generate dispersion energy grids (f-c), build an ensemble, pick a baseline dispersion curve, apply a DOI mask, and run a simple inversion workflow to estimate **Vs(z)** with uncertainty summaries.

This repository is **synthetic-only** and does **not** require any external datasets.

---

## What this project does

Given a synthetic dispersion energy image (frequency x phase-velocity), the script:

- Generates **synthetic f-c energy grids** (seeded / reproducible)
- Builds a **dispersion ensemble** and stacks energy
- Performs **ridge picking** (baseline dispersion curve)
- Applies a **DOI (Depth of Investigation)** mask and coverage diagnostics
- Runs an inversion workflow to estimate **Vs(z)** and **Vs_z(z)** (time-averaged)
- Produces uncertainty summaries:
  - **P10-P90**
  - **Median**
  - **CoV (std/mean)**
  - **N(z) coverage after DOI**

---

## Repository structure

## Repository structure

```
.
├── MASW_Ensemble_Toolkit_Synthetic.py
├── requirements.txt
├── LICENSE
├── .gitignore
├── figures/
│   ├── stacked_energy_and_pick.png
│   ├── picked_dispersion_curve.png
│   ├── dispersion_ensemble_corrected.png
│   ├── ensemble_summary_full_depth.png
│   └── ensemble_summary_zoom.png
└── outputs/
    └── (auto-generated at runtime)
```

## Quick start

### Option A (recommended): run in a virtual environment

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux / macOS
source .venv/bin/activate

pip install -r requirements.txt
python MASW_Ensemble_Toolkit_Synthetic.py
```

### Option B: run directly

```bash
pip install -r requirements.txt
python MASW_Ensemble_Toolkit_Synthetic.py
```

## Output files

After running, the script creates an `outputs/` folder containing:

- Stacked MASW energy + picked ridge (baseline)
- Picked dispersion curve (baseline)
- Dispersion ensemble (corrected)
- Vs(z) ensemble summary (full depth)
- Vs(z) ensemble summary (zoom)

## Notes

- This is a synthetic-only demo and does not use any external datasets.
- Results are deterministic (seeded) unless you change the random seed in the script.
- If you want to regenerate figures, just re-run the script.

## License

This project is released under the MIT License.
