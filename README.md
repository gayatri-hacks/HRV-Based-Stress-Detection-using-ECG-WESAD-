# HRV-Based Stress Detection with WESAD ECG

An educational research prototype that turns a 700 Hz WESAD chest-ECG CSV into HRV features and a binary logistic-regression stress estimate. It is **not clinically validated** and must not be used for diagnosis, treatment, or emergency decisions.

## Quick start

Use Python 3.10–3.12.

```bash
python3.11 -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

```bash
cd frontend
npm ci
npm run dev
```

Set `VITE_API_URL` if the API is not at `http://127.0.0.1:8000`.

## API contract

`POST /predict/ecg/stress` accepts multipart fields `file` and `fs`. The file must be a CSV containing a finite numeric `signal` column (or one numeric column). `fs` must be exactly `700`; the deployed model was trained on 700 Hz WESAD chest ECG and has not been validated at other rates.

The API rejects unsupported media types with 415, oversized uploads with 413, invalid input with 422, and absent/corrupt/incompatible model artifacts with 503. Uploads default to 100 MB and one hour of samples; set `MAX_ECG_UPLOAD_BYTES` or `MAX_ECG_SAMPLES` for a stricter deployment limit.

Inference uses 60-second windows with a 30-second hop. Every usable window is extracted, but—intentionally, to preserve the existing model semantics—the **final valid window** is scored. The response names that policy and reports its start time and the number of valid windows; it does not average unvalidated window predictions.

`GET /health` only means the process is alive. `GET /ready` additionally checks that the selected artifacts are present, not Git LFS pointers, loadable, dimension-compatible, correctly ordered, binary-labelled, and capable of returning finite normalized probabilities.

## Data and evaluation

WESAD is licensed data. Obtain it from the [official WESAD page](https://ubi29.informatik.uni-siegen.de/usi/data_wesad.html), accept its terms, and never commit raw recordings. The training script expects an external licensed directory:

```bash
.venv/bin/python scripts/train_wesad_models.py --raw-dir /path/to/WESAD --epochs 25
```

The committed configuration records contiguous baseline/stress segments, 60-second windows with 30-second hop, training-only scaling, S5 validation, and subject-disjoint S2/S3/S4 final test subjects. The selected logistic-regression artifact’s committed held-out metrics are accuracy 0.800, stress F1 0.622, precision 0.903, recall 0.475, and ROC-AUC 0.879. These are experimental held-out-window metrics, not clinical performance claims.

Run a genuine-data smoke test only after extracting an authorized recording:

```bash
python scripts/extract_wesad_ecg.py --raw-dir /path/to/WESAD --subject S2 --output /tmp/S2_ecg.csv
WESAD_ECG_SMOKE_CSV=/tmp/S2_ecg.csv pytest -m integration -q
```

Synthetic fixtures are for deterministic tests only and are never presented as WESAD evidence.

## Verification

```bash
pytest -m 'not integration' -q
cd frontend && npm run lint && npx tsc --noEmit && npm test && npm run build
```

See [architecture](docs/ARCHITECTURE.md), [engineering decisions](docs/ENGINEERING_DECISIONS.md), [verification](docs/VERIFICATION.md), and the [interview walkthrough](docs/INTERVIEW_WALKTHROUGH.md).
