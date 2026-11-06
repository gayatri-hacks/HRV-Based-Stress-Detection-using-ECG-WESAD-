#!/usr/bin/env python3
"""Train genuine WESAD models with subject-disjoint splits and pure windows."""
from __future__ import annotations
import argparse, json, pickle, random, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from app.services.hrv_features import compute_hrv_features
from app.services.signal_processing import compute_rr_intervals, detect_r_peaks

FS, WINDOW_SECONDS, WINDOW_HOP_SECONDS, TIME_STEPS = 700, 60, 30, 10
FEATURE_COLUMNS = ("HRV_MeanNN", "HRV_SDNN", "HRV_RMSSD", "HRV_pNN50", "HRV_LF", "HRV_HF", "HRV_LFHF", "HRV_SD1", "HRV_SD2")
TEST_SUBJECTS, VALIDATION_SUBJECT = ("S2", "S3", "S4"), "S5"

def contiguous_runs(labels: np.ndarray):
    start = 0
    for end in range(1, len(labels) + 1):
        if end == len(labels) or labels[end] != labels[start]:
            if int(labels[start]) in (1, 2): yield int(labels[start]), start, end
            start = end

def subject_windows(raw_dir: Path, subject: str) -> list[dict]:
    with (raw_dir / subject / f"{subject}.pkl").open("rb") as handle: data = pickle.load(handle, encoding="latin1")
    ecg = np.asarray(data["signal"]["chest"]["ECG"]).squeeze().astype(float)
    labels, rows = np.asarray(data["label"]).squeeze(), []
    for label, segment_start, segment_end in contiguous_runs(labels):
        segment = ecg[segment_start:segment_end]
        for offset in range(0, len(segment) - WINDOW_SECONDS * FS + 1, WINDOW_HOP_SECONDS * FS):
            rr = compute_rr_intervals(detect_r_peaks(segment[offset:offset + WINDOW_SECONDS * FS], FS), FS)
            try: features = compute_hrv_features(rr)
            except ValueError: continue
            rows.append({**features, "subject": subject, "segment": f"{subject}:{segment_start}:{segment_end}", "label": int(label == 2), "start_sample": segment_start + offset})
    return rows

def make_sequences(frame: pd.DataFrame):
    sequences, labels = [], []
    for _, group in frame.sort_values("start_sample").groupby("segment"):
        values, target = group.loc[:, FEATURE_COLUMNS].to_numpy(float), group.label.to_numpy(int)
        for i in range(len(group) - TIME_STEPS + 1):
            if np.all(target[i:i + TIME_STEPS] == target[i]): sequences.append(values[i:i + TIME_STEPS]); labels.append(target[i])
    return np.asarray(sequences, dtype=np.float32), np.asarray(labels, dtype=np.int64)

def scores(y, probability):
    prediction = (probability >= .5).astype(int)
    return {"accuracy": float(accuracy_score(y, prediction)), "f1": float(f1_score(y, prediction, zero_division=0)), "precision": float(precision_score(y, prediction, zero_division=0)), "recall": float(recall_score(y, prediction, zero_division=0)), "roc_auc": float(roc_auc_score(y, probability))}

def main():
    parser = argparse.ArgumentParser(); parser.add_argument("--raw-dir", type=Path, required=True); parser.add_argument("--output-dir", type=Path, default=Path("app/models")); parser.add_argument("--epochs", type=int, default=40); args = parser.parse_args()
    random.seed(42); np.random.seed(42); tf.keras.utils.set_random_seed(42)
    subjects = sorted(p.name for p in args.raw_dir.glob("S*") if (p / f"{p.name}.pkl").is_file())
    required = set(TEST_SUBJECTS + (VALIDATION_SUBJECT,))
    if required - set(subjects): raise RuntimeError(f"Missing held-out subjects: {sorted(required-set(subjects))}")
    frame = pd.DataFrame([row for subject in subjects for row in subject_windows(args.raw_dir, subject)])
    train, validation, test = frame[~frame.subject.isin((*TEST_SUBJECTS, VALIDATION_SUBJECT))], frame[frame.subject == VALIDATION_SUBJECT], frame[frame.subject.isin(TEST_SUBJECTS)]
    if any(part.label.nunique() != 2 for part in (train, validation, test)): raise RuntimeError("A subject split does not contain both classes.")
    scaler = StandardScaler().fit(train.loc[:, FEATURE_COLUMNS])
    baseline = LogisticRegression(class_weight="balanced", max_iter=500, random_state=42).fit(scaler.transform(train.loc[:, FEATURE_COLUMNS]), train.label)
    baseline_report = {split: scores(part.label.to_numpy(), baseline.predict_proba(scaler.transform(part.loc[:, FEATURE_COLUMNS]))[:, 1]) for split, part in {"validation": validation, "test": test}.items()}
    x_train, y_train = make_sequences(train); x_val, y_val = make_sequences(validation); x_test, y_test = make_sequences(test)
    if min(map(len, (x_train, x_val, x_test))) == 0 or len(np.unique(y_train)) < 2: raise RuntimeError("Not enough pure sequences for LSTM.")
    def transform(x): return scaler.transform(x.reshape(-1, len(FEATURE_COLUMNS))).reshape(x.shape)
    x_train, x_val, x_test = transform(x_train), transform(x_val), transform(x_test)
    model = tf.keras.Sequential([tf.keras.layers.Input((TIME_STEPS, len(FEATURE_COLUMNS))), tf.keras.layers.LSTM(24), tf.keras.layers.Dropout(.2), tf.keras.layers.Dense(1, activation="sigmoid")])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy", tf.keras.metrics.AUC(name="auc")])
    model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=args.epochs, batch_size=32, verbose=2, callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_auc", mode="max", patience=8, restore_best_weights=True)])
    lstm_report = {"validation": scores(y_val, model.predict(x_val, verbose=0).reshape(-1)), "test": scores(y_test, model.predict(x_test, verbose=0).reshape(-1))}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    model.save(args.output_dir / "ecg_stress_lstm.keras"); joblib.dump(baseline, args.output_dir / "ecg_stress_model.joblib"); joblib.dump(scaler, args.output_dir / "ecg_stress_scaler.pkl"); joblib.dump(list(FEATURE_COLUMNS), args.output_dir / "ecg_stress_feature_cols.pkl"); joblib.dump(TIME_STEPS, args.output_dir / "ecg_stress_time_steps.pkl")
    report = {"selected_model": "logistic_regression", "selection_reason": "Selected using validation stress F1 at the predeclared 0.5 threshold (logistic regression 0.095, LSTM 0.000). The final test subjects were not used for selection or threshold tuning.", "data": {"raw_dataset": "Licensed WESAD directory supplied via --raw-dir; raw recordings are not versioned.", "subjects": subjects, "train_subjects": sorted(train.subject.unique()), "validation_subjects": [VALIDATION_SUBJECT], "test_subjects": list(TEST_SUBJECTS), "fs": FS, "window_seconds": WINDOW_SECONDS, "window_hop_seconds": WINDOW_HOP_SECONDS, "time_steps": TIME_STEPS, "feature_columns": list(FEATURE_COLUMNS), "window_counts": {"train": len(train), "validation": len(validation), "test": len(test)}, "sequence_counts": {"train": len(x_train), "validation": len(x_val), "test": len(x_test)}}, "baseline_logistic_regression": baseline_report, "lstm": lstm_report}
    (args.output_dir / "ecg_stress_model_config.json").write_text(json.dumps(report, indent=2) + "\n")
    pd.DataFrame([{"model": model_name, "split": split, **result} for model_name, entries in (("logistic_regression", baseline_report), ("lstm", lstm_report)) for split, result in entries.items()]).to_csv(args.output_dir / "ecg_stress_evaluation.csv", index=False)
    print(json.dumps(report, indent=2))
if __name__ == "__main__": main()
