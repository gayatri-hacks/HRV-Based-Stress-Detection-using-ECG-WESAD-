# HRV-Based Stress Detection using ECG (WESAD)

This repository contains the **ECG → HRV → Stress detection pipeline** developed as part of the *HRV Analysis for Stress / Sleep Monitoring* project.

> Research software only — this model is not clinically validated and must not be used for medical diagnosis or treatment.

## Running the application

Use Python 3.10–3.12 (TensorFlow does not currently support Python 3.14). A fresh clone already contains the small deployed model bundle as ordinary Git files; no Git LFS command is needed.

```bash
python3.11 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

`GET /health` checks that the API process is alive. `GET /ready` reports each model artifact and remains `ready: false` if an artifact is missing; the API never uses a dummy model. The stress endpoint accepts only a CSV with a numeric `signal` column and an `fs` form value. The protected upload limit defaults to 100 MB and is configurable with `MAX_ECG_UPLOAD_BYTES`. It uses 60-second HRV windows with a 30-second hop; the trained model configuration defines its timestep count and minimum duration.

In another terminal:

```bash
cd frontend
npm ci
cp .env.example .env.local
npm run dev
```

Set `VITE_API_URL` when the API is not running at `http://127.0.0.1:8000`. The UI intentionally offers only ECG CSV stress prediction: PPG, PDF/image extraction, and sleep-stage prediction are disabled because this repository has no matching genuine backend/model path.

## WESAD provenance and real-data demonstration

WESAD is licensed source data; obtain it from the [official WESAD page](https://ubi29.informatik.uni-siegen.de/usi/data_wesad.html), accept its terms, and place its subject directories outside Git (for example `data/WESAD/`). Do not commit the raw archive or recordings. The chest ECG is sampled at 700 Hz.

### Reproducible real-data training

The original LFS model objects are no longer recoverable, so [`scripts/train_wesad_models.py`](scripts/train_wesad_models.py) trains replacement artifacts only from licensed raw WESAD files:

```bash
.venv/bin/python scripts/train_wesad_models.py --raw-dir ~/Downloads/WESAD --epochs 25
```

It processes contiguous baseline and stress segments separately, uses 60-second windows with a 30-second hop, and discards any window that cannot yield valid HRV features. This avoids the original notebook's discontinuity caused by removing labels before resampling. It holds out S2/S3/S4 for final testing and S5 for validation; no subject occurs in more than one split. The former 30-step sequence would require 930 seconds and cannot fit a pure stress region. The replacement LSTM uses ten steps (330 seconds). Model selection uses validation stress F1 at a fixed 0.5 threshold only: logistic regression (0.095) was selected over the LSTM (0.000). Final S2/S3/S4 results are reported only after that decision. Exact feature order, split, configuration, and results are tracked in [`app/models/ecg_stress_model_config.json`](app/models/ecg_stress_model_config.json) and [`app/models/ecg_stress_evaluation.csv`](app/models/ecg_stress_evaluation.csv).

Current genuine held-out window metrics (S2/S3/S4) are: logistic regression accuracy 0.800, stress F1 0.622, precision 0.903, recall 0.475, ROC-AUC 0.879. The LSTM was retained as an evaluated artifact but is not deployed because its held-out stress F1 was 0.359. These results are experimental, subject to protocol confounding, and not clinical validation.

Export a genuine full-session recording (the aligned WESAD `label` column is preserved; do not concatenate label-selected regions because that would make discontinuous ECG samples look adjacent):

```bash
python scripts/extract_wesad_ecg.py --raw-dir data/WESAD --subject S2 --output /tmp/S2_ecg.csv
```

The script output is an actual recording and includes `sample`, `time_seconds`, `signal`, and WESAD `label`. `data/processed/` contains derived feature/RR artifacts. `data/synthetic/` and `src/synthetic_data.ipynb` are test/demo fixtures only and are not valid evidence of real-data inference.

For a fresh-clone real-data demonstration, first obtain licensed WESAD data, extract an ECG CSV, then run:

```bash
WESAD_ECG_SMOKE_CSV=/tmp/S2_ecg.csv pytest -m integration -q
```

## Verification

```bash
pytest -m 'not integration' -q
cd frontend && npm run lint && npx tsc --noEmit && npm test && npm run build
```

The GitHub Actions workflow runs unit/contract tests with `python -m pytest -m "not integration"` and frontend checks. The genuine WESAD smoke test is intentionally opt-in because licensed recordings do not belong in CI. The deployed model bundle is committed as ordinary Git files, not LFS pointers.

This branch focuses **exclusively on ECG-based stress detection**, covering:

* Raw ECG preprocessing
* R-peak detection
* RR (NN) interval extraction
* HRV feature engineering
* Subject-independent baseline models
* Reproducible artifacts for downstream deep learning models

The output of this work serves as the **foundation** for the primary temporal models (LSTM / GRU) developed by other team members.

---

## Dataset

- We use the **[WESAD (WEarable Stress and Affect Detection)](https://ubi29.informatik.uni-siegen.de/usi/data_wesad.html)** dataset.


**Source:**
PhysioNet – WESAD Dataset
(see `data/WESAD/wesad_readme.pdf` for official documentation)

### Relevant Details

* Chest-worn ECG (RespiBAN)
* Sampling rate: **700 Hz**
* Labels provided at signal level
* Stress protocol conducted in controlled lab settings

### Label Mapping (as per official README)

| Label | Meaning                          |
| ----- | -------------------------------- |
| 0     | Undefined / transition (ignored) |
| 1     | Baseline                         |
| 2     | Stress                           |
| 3     | Amusement (ignored here)         |
| 4     | Meditation (ignored here)        |

For this work, we perform **binary stress classification**:

* **Non-stress:** Baseline (label = 1)
* **Stress:** Stress (label = 2)

---

## Project Pipeline Overview

```
Raw ECG (700 Hz)
   ↓
ECG Cleaning + Downsampling (250 Hz)
   ↓
R-Peak Detection
   ↓
RR / NN Interval Extraction
   ↓
Windowed HRV Feature Extraction
   ↓
Baseline ML Models (LogReg, RF)
```

---

## Directory Structure

```
.
├── app.py                         # Streamlit ECG/label exploration app
├── approach.md                    # High-level project approach
├── requirements.txt               # Python dependencies
│
├── data
│   ├── WESAD                      # Raw WESAD dataset (per-subject folders)
│   │   ├── S2, S3, ..., S17
│   │   └── wesad_readme.pdf
│   │
│   └── processed
│       ├── wesad/                 # RR-level intermediate artifacts
│       │   ├── S2_rr.npz
│       │   ├── S3_rr.npz
│       │   └── ...
│       └── wesad_hrv_features.csv # Final HRV feature dataset
│
├── runs
│   ├── logreg_coefficients.csv    # Logistic regression coefficients
│   ├── rf_importance.csv          # Random forest feature importance
│   └── models
│       ├── logreg_baseline.joblib
│       ├── rf_baseline.joblib
│       ├── hrv_feature_list.json
│       └── baseline_metrics.joblib
│
├── src
│   ├── eda.ipynb                  # Dataset exploration
│   ├── rr_processing.ipynb        # ECG → R-peaks → RR extraction
│   ├── hrv_feature_extraction.ipynb
│   └── baseline_model.ipynb       # Baseline model training & evaluation
│
└── utils
    └── wesad_loader.py            # Dataset loading utilities
```

---

## Streamlit Data Explorer

`app.py` provides an interactive Streamlit app for:

* Selecting any subject (e.g., S17)
* Selecting labels (baseline, stress)
* Visualizing ECG segments
* Overlaying detected R-peaks

This tool was used extensively for:

* Data sanity checks
* Verifying label alignment
* Validating R-peak quality

Run with:

```bash
streamlit run app.py
```

---

## RR Interval Processing

Notebook: `src/rr_processing.ipynb`

Steps:

1. Load raw ECG from `Sx.pkl`
2. Filter to baseline + stress labels
3. Downsample ECG from 700 Hz → 250 Hz
4. Clean ECG signal
5. Detect R-peaks using NeuroKit2
6. Compute RR (NN) intervals
7. Remove physiologically invalid beats
8. Save per-subject RR artifacts

### Output (per subject)

Saved as:

```
data/processed/wesad/Sx_rr.npz
```

Contains:

* `rpeaks` – R-peak indices
* `rr_intervals` – cleaned NN intervals (seconds)
* `rr_labels` – stress/baseline labels aligned to RR
* `fs` – sampling rate (250 Hz)

This RR-level data is the **canonical interface** for all downstream models.

---

## HRV Feature Extraction

Notebook: `src/hrv_feature_extraction.ipynb`

### Windowing Strategy

* Window size: **60 seconds**
* Overlap: **50%**
* Mixed-label windows are discarded

### Extracted HRV Features (Interpretable)

* HRV_MeanNN
* HRV_SDNN
* HRV_RMSSD
* HRV_pNN50
* HRV_LF
* HRV_HF
* HRV_LFHF

Each row corresponds to **one time window**.

### Output

```
data/processed/wesad_hrv_features.csv
```

This dataset is:

* Subject-independent
* ML-ready
* Used by both baseline and primary models

---

## Baseline Model Training

Notebook: `src/baseline_model.ipynb`

### Models

* Logistic Regression (with standardization)
* Random Forest

### Evaluation Protocol

* **Subject-wise cross-validation** (GroupKFold)
* Prevents subject leakage
* Metrics:

  * Accuracy
  * F1-score
  * ROC-AUC

### Results (Average)

**Logistic Regression**

* Accuracy ≈ 0.80
* F1 ≈ 0.71
* ROC-AUC ≈ 0.87

**Random Forest**

* Accuracy ≈ 0.80
* F1 ≈ 0.73
* ROC-AUC ≈ 0.82

### Feature Importance

Both models consistently identify:

* **Mean heart rate (HRV_MeanNN)** as the strongest stress indicator
* Short-term HRV (pNN50, RMSSD) as secondary predictors

---

## Saved Model Artifacts

Models are saved for **reproducibility and comparison only**.

They are **not intended to be extended or fine-tuned**.

Saved in:

```
runs/models/
```

Includes:

* Trained baseline models
* Feature list
* Cross-validation metrics

---

## Environment Notes

* Python 3.12
* NumPy pinned to `< 2.0` (NeuroKit2 compatibility)
* Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Summary

This branch provides:

* A complete ECG → HRV stress pipeline
* Physiologically validated features
* Strong, interpretable baseline models
* Clean, reusable artifacts for temporal deep learning models

It serves as the **ground truth preprocessing and baseline reference** for the overall project.

---
