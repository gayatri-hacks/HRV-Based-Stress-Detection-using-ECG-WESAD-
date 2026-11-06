import numpy as np
import pytest
from app.services.hrv_features import compute_hrv_features
from app.services.signal_processing import SignalValidationError, load_signal_csv

def test_load_signal_csv_accepts_signal_column():
    assert np.array_equal(load_signal_csv(b"signal\n1\n2\n3\n"), np.array([1., 2., 3.]))

def test_load_signal_csv_rejects_non_numeric_values():
    with pytest.raises(SignalValidationError): load_signal_csv(b"signal\n1\nnot-a-number\n")

def test_hrv_features_are_finite():
    features = compute_hrv_features(np.array([800, 820, 790, 810, 805, 815, 800, 820, 810, 805]))
    assert set(("HRV_MeanNN", "HRV_LFHF", "HRV_SD2")) <= features.keys()
    assert all(np.isfinite(value) for value in features.values())
