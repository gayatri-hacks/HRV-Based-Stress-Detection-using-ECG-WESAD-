from __future__ import annotations
import os
from typing import Annotated
import numpy as np
import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from app.services.hrv_features import compute_hrv_features
from app.services.model_loader import WINDOW_HOP_SECONDS, WINDOW_SECONDS, ModelUnavailableError, artifact_status, load_stress_artifacts
from app.services.signal_processing import SignalValidationError, compute_rr_intervals, detect_r_peaks, load_signal_csv

MAX_FILE_BYTES = int(os.getenv("MAX_ECG_UPLOAD_BYTES", str(100 * 1024 * 1024)))
MIN_FS, MAX_FS = 50.0, 1000.0
origins = [x.strip() for x in os.getenv("CORS_ALLOW_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173").split(",") if x.strip()]
app = FastAPI(title="WESAD ECG stress API", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=origins, allow_credentials=False, allow_methods=["GET", "POST"], allow_headers=["*"])

@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}

@app.get("/ready")
def ready() -> dict:
    artifacts = artifact_status()
    return {"ready": all(value == "present" for value in artifacts.values()), "artifacts": artifacts}

@app.post("/predict/ecg/stress")
async def predict_ecg_stress(file: Annotated[UploadFile, File(...)], fs: Annotated[float, Form(...)]) -> dict:
    if not MIN_FS <= fs <= MAX_FS:
        raise HTTPException(422, detail=f"Sampling frequency must be between {MIN_FS:g} and {MAX_FS:g} Hz.")
    if file.content_type not in {"text/csv", "application/vnd.ms-excel", "application/csv", None, ""}:
        raise HTTPException(415, detail="Only ECG CSV uploads are supported.")
    contents = await file.read(MAX_FILE_BYTES + 1)
    if len(contents) > MAX_FILE_BYTES:
        raise HTTPException(413, detail=f"File exceeds the {MAX_FILE_BYTES // 1024 // 1024} MB upload limit.")
    try:
        signal = load_signal_csv(contents)
    except SignalValidationError as exc:
        raise HTTPException(422, detail=str(exc)) from exc
    try:
        artifacts = load_stress_artifacts()
    except ModelUnavailableError as exc:
        raise HTTPException(503, detail=str(exc)) from exc
    minimum_seconds = artifacts.minimum_duration_seconds
    if len(signal) / fs < minimum_seconds:
        raise HTTPException(422, detail=f"Need at least {minimum_seconds:g} seconds of ECG for the selected trained model.")
    window_samples, hop_samples = round(WINDOW_SECONDS*fs), round(WINDOW_HOP_SECONDS*fs)
    rows = []
    for start in range(0, len(signal)-window_samples+1, hop_samples):
        rr_ms = compute_rr_intervals(detect_r_peaks(signal[start:start+window_samples], fs), fs)
        try:
            features = compute_hrv_features(rr_ms)
            rows.append([features[name] for name in artifacts.feature_columns])
        except (ValueError, KeyError):
            continue
    if not rows:
        raise HTTPException(422, detail="Insufficient valid ECG windows after R-peak and HRV processing.")
    features = artifacts.scaler.transform(pd.DataFrame(rows[-1:], columns=artifacts.feature_columns))
    # Confidence is the probability of the semantic positive class (stress=1),
    # not simply whichever probability happens to occupy column 1.
    positive_index = int(np.flatnonzero(artifacts.model.classes_ == 1)[0])
    probability = float(artifacts.model.predict_proba(features)[0, positive_index])
    if not 0 <= probability <= 1:
        raise HTTPException(503, detail="Model returned an invalid probability.")
    prediction = int(probability >= .5)
    return {
        "prediction": prediction,
        "stress_probability": probability,
        "predicted_class_confidence": probability if prediction else 1.0 - probability,
        "windows_used": 1,
        "minimum_duration_seconds": minimum_seconds,
    }
