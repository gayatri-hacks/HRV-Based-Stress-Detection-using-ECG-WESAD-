from __future__ import annotations
import numpy as np
from scipy.integrate import trapezoid
from scipy.signal import welch

def compute_hrv_features(rr_ms: np.ndarray) -> dict[str, float]:
    rr_ms = np.asarray(rr_ms, dtype=float)
    if len(rr_ms) < 4 or not np.isfinite(rr_ms).all():
        raise ValueError("At least four finite RR intervals are required for HRV features.")
    diff_rr = np.diff(rr_ms)
    if len(diff_rr) < 2:
        raise ValueError("Too few RR differences for HRV features.")
    mean_nn, sdnn = float(np.mean(rr_ms)), float(np.std(rr_ms, ddof=1))
    beat_time = np.r_[0.0, np.cumsum(rr_ms[:-1]) / 1000.0]
    grid = np.arange(0, beat_time[-1], 0.25)
    if len(grid) < 8:
        raise ValueError("RR duration is too short for frequency-domain HRV features.")
    freqs, psd = welch(np.interp(grid, beat_time, rr_ms), fs=4.0, nperseg=min(256, len(grid)))
    def band(lo: float, hi: float) -> float:
        mask = (freqs >= lo) & (freqs <= hi)
        return float(trapezoid(psd[mask], freqs[mask])) if mask.sum() >= 2 else 0.0
    lf, hf, diff_std = band(0.04, 0.15), band(0.15, 0.4), float(np.std(diff_rr, ddof=1))
    return {"HRV_MeanNN": mean_nn, "HRV_SDNN": sdnn, "HRV_RMSSD": float(np.sqrt(np.mean(diff_rr**2))), "HRV_pNN50": float(100*np.mean(np.abs(diff_rr)>50)), "HRV_LF": lf, "HRV_HF": hf, "HRV_LFHF": float(lf/hf) if hf else 0.0, "HRV_SD1": float(np.sqrt(.5)*diff_std), "HRV_SD2": float(np.sqrt(max(0., 2*sdnn**2-.5*diff_std**2)))}
