#!/usr/bin/env python3
"""Exit non-zero when any model artifact is an unresolved Git LFS pointer."""
from pathlib import Path
root = Path(__file__).resolve().parents[1]
paths = {
    "selected_model": root / "app/models/ecg_stress_model.joblib",
    "scaler": root / "app/models/ecg_stress_scaler.pkl",
    "feature_columns": root / "app/models/ecg_stress_feature_cols.pkl",
    "configuration": root / "app/models/ecg_stress_model_config.json",
}
status = {name: "missing" if not path.exists() else "git_lfs_pointer" if path.read_bytes()[:64].startswith(b"version https://git-lfs.github.com/spec/v1") else "present" for name, path in paths.items()}
for name, state in status.items(): print(f"{name}: {state}")
raise SystemExit(0 if all(value == "present" for value in status.values()) else 1)
