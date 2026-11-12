from fastapi.testclient import TestClient
from app.main import app
from app.services.model_loader import load_stress_artifacts
import numpy as np

client = TestClient(app)

def upload(contents: bytes, fs: str = "700", content_type: str = "text/csv"):
    return client.post("/predict/ecg/stress", data={"fs": fs}, files={"file": ("ecg.csv", contents, content_type)})

def test_health_and_readiness_expose_artifact_status():
    assert client.get("/health").json() == {"status": "ok"}
    response = client.get("/ready")
    assert response.status_code == 200
    assert "ready" in response.json()

def test_rejects_non_csv_and_invalid_sampling_rate():
    assert upload(b"signal\n1\n2", content_type="application/pdf").status_code == 415
    assert upload(b"signal\n1\n2", fs="5").status_code == 422
    assert upload(b"signal\n1\n2", fs="250").status_code == 422

def test_rejects_invalid_csv_before_model_loading():
    response = upload(b"a,b\n1,2\n")
    assert response.status_code == 422
    assert "signal" in response.json()["detail"]

def test_short_recording_is_rejected_before_prediction():
    response = upload(b"signal\n1\n2\n3\n")
    assert response.status_code == 422
    assert "Need at least" in response.json()["detail"]

def test_rejects_nan_and_constant_ecg_before_model_prediction():
    assert upload(b"signal\n1\nnan\n2\n").status_code == 422
    assert upload(b"signal\n" + b"1\n" * 50_000).status_code == 422

def test_prediction_reports_final_valid_window_metadata(monkeypatch):
    artifacts = load_stress_artifacts()
    monkeypatch.setattr("app.main.load_signal_csv", lambda *_args, **_kwargs: np.ones(700 * 150))
    monkeypatch.setattr("app.main.detect_r_peaks", lambda *_args: np.arange(0, 700 * 60, 700))
    monkeypatch.setattr("app.main.compute_hrv_features", lambda _rr: {name: 1.0 for name in artifacts.feature_columns})
    response = upload(b"signal\n1\n")
    assert response.status_code == 200
    payload = response.json()
    assert payload["selected_window"] == "final_valid"
    assert payload["valid_windows_found"] == 4
    assert payload["selected_window_start_seconds"] == 90
