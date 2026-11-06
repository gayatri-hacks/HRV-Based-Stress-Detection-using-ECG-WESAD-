"""Opt-in smoke test: never mistakes the synthetic fixture for WESAD evidence."""
import os
from pathlib import Path
import pytest
from fastapi.testclient import TestClient
from app.main import app

@pytest.mark.integration
def test_real_wesad_ecg_prediction():
    recording = os.getenv("WESAD_ECG_SMOKE_CSV")
    if not recording or not Path(recording).is_file():
        pytest.skip("Set WESAD_ECG_SMOKE_CSV to a genuine extracted WESAD ECG CSV.")
    with Path(recording).open("rb") as handle:
        response = TestClient(app).post("/predict/ecg/stress", data={"fs": "700"}, files={"file": ("wesad.csv", handle, "text/csv")})
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["prediction"] in (0, 1)
    assert 0 <= payload["stress_probability"] <= 1
    assert 0 <= payload["predicted_class_confidence"] <= 1
    expected = payload["stress_probability"] if payload["prediction"] == 1 else 1 - payload["stress_probability"]
    assert payload["predicted_class_confidence"] == pytest.approx(expected)
