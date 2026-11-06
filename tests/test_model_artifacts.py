from app.services.model_loader import artifact_status, load_stress_artifacts

def test_selected_model_artifacts_are_real_and_compatible():
    assert set(artifact_status().values()) == {"present"}
    artifacts = load_stress_artifacts()
    assert artifacts.scaler.n_features_in_ == len(artifacts.feature_columns)
    assert artifacts.model.n_features_in_ == len(artifacts.feature_columns)
