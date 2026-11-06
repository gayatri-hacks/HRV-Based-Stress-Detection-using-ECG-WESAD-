from __future__ import annotations
from io import BytesIO
import numpy as np
import pandas as pd
from scipy.signal import find_peaks

class SignalValidationError(ValueError):
    pass

def load_signal_csv(contents: bytes) -> np.ndarray:
    try:
        df = pd.read_csv(BytesIO(contents))
    except Exception as exc:
        raise SignalValidationError("Upload a UTF-8 CSV containing an ECG `signal` column.") from exc
    if df.empty:
        raise SignalValidationError("The CSV contains no ECG samples.")
    if "signal" in df.columns:
        values = df["signal"]
    elif len(df.columns) == 1:
        values = df.iloc[:, 0]
    else:
        raise SignalValidationError("CSV must contain a `signal` column (or exactly one numeric column).")
    signal = pd.to_numeric(values, errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(signal).all():
        raise SignalValidationError("ECG signal contains missing or non-numeric values.")
    if np.ptp(signal) == 0:
        raise SignalValidationError("ECG signal is constant; R peaks cannot be detected.")
    return signal

def detect_r_peaks(signal: np.ndarray, fs: float) -> np.ndarray:
    prominence = max(float(np.std(signal)) * 0.35, np.finfo(float).eps)
    return find_peaks(signal, distance=max(1, int(fs * 0.4)), prominence=prominence)[0]

def compute_rr_intervals(peaks: np.ndarray, fs: float) -> np.ndarray:
    return np.diff(peaks).astype(float) / fs * 1000.0
