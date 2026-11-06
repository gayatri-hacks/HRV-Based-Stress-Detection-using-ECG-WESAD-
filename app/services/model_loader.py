"""Validated, lazy loading of the selected genuine WESAD model."""
from __future__ import annotations
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any
import json
import joblib

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = REPOSITORY_ROOT / "app" / "models"
MODEL_PATH, SCALER_PATH = MODEL_DIR / "ecg_stress_model.joblib", MODEL_DIR / "ecg_stress_scaler.pkl"
FEATURE_COLS_PATH, CONFIG_PATH = MODEL_DIR / "ecg_stress_feature_cols.pkl", MODEL_DIR / "ecg_stress_model_config.json"
WINDOW_SECONDS, WINDOW_HOP_SECONDS = 60, 30
class ModelUnavailableError(RuntimeError): pass
def _is_lfs_pointer(path: Path) -> bool:
    try: return path.read_bytes()[:64].startswith(b"version https://git-lfs.github.com/spec/v1")
    except OSError: return False
def artifact_status() -> dict[str, str]:
    paths = {"selected_model": MODEL_PATH, "scaler": SCALER_PATH, "feature_columns": FEATURE_COLS_PATH, "configuration": CONFIG_PATH}
    return {name: "missing" if not path.exists() else "git_lfs_pointer" if _is_lfs_pointer(path) else "present" for name, path in paths.items()}
@dataclass(frozen=True)
class StressArtifacts:
    model: Any; scaler: Any; feature_columns: tuple[str, ...]; minimum_duration_seconds: int
@lru_cache(maxsize=1)
def load_stress_artifacts() -> StressArtifacts:
    failures = [f"{key} ({value})" for key, value in artifact_status().items() if value != "present"]
    if failures: raise ModelUnavailableError("Trained WESAD artifacts are unavailable: " + ", ".join(failures) + ". Run scripts/train_wesad_models.py with licensed raw WESAD data.")
    try:
        config = json.loads(CONFIG_PATH.read_text())
        if config.get("selected_model") != "logistic_regression": raise ValueError("configuration does not select the supported logistic regression model")
        model, scaler, columns = joblib.load(MODEL_PATH), joblib.load(SCALER_PATH), tuple(joblib.load(FEATURE_COLS_PATH))
    except Exception as exc: raise ModelUnavailableError(f"Could not load genuine stress-model artifacts: {exc}") from exc
    if not columns or len(set(columns)) != len(columns): raise ModelUnavailableError("Feature-column artifact is empty or duplicated.")
    if getattr(scaler, "n_features_in_", len(columns)) != len(columns) or getattr(model, "n_features_in_", len(columns)) != len(columns): raise ModelUnavailableError("Model/scaler feature count does not match saved feature order.")
    if not hasattr(model, "predict_proba"): raise ModelUnavailableError("Selected model does not expose probabilities.")
    return StressArtifacts(model, scaler, columns, int(config["data"]["window_seconds"]))
