#!/usr/bin/env python3
"""Export genuine chest ECG and aligned WESAD labels to a CSV (never synthetic)."""
from __future__ import annotations
import argparse
from pathlib import Path
import sys
import pandas as pd
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.wesad_loader import FS, extract_ecg_and_labels, load_subject

def main() -> None:
    parser = argparse.ArgumentParser(description="Extract a licensed WESAD subject's 700 Hz chest ECG.")
    parser.add_argument("--raw-dir", required=True, type=Path, help="Directory containing S2/, S3/, … from WESAD.")
    parser.add_argument("--subject", required=True, help="Subject identifier, e.g. S2")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    data = load_subject(args.raw_dir, args.subject)
    ecg, labels = extract_ecg_and_labels(data)
    if len(ecg) != len(labels): raise ValueError("WESAD ECG/label alignment is invalid.")
    frame = pd.DataFrame({"sample": np.arange(len(ecg)), "time_seconds": np.arange(len(ecg))/FS, "signal": ecg, "label": labels})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.output, index=False)
    print(f"Wrote {len(frame)} genuine WESAD samples at {FS} Hz to {args.output}")
if __name__ == "__main__": main()
