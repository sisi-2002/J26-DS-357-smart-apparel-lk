#!/usr/bin/env python3
"""Load saved C3 model; predict at ONE historical 2015 test snapshot."""
import argparse
from pathlib import Path
from model_inference import format_result, predict_saved

if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--model', type=Path, default=Path('models/c3_7day_sigmoid.joblib'))
    ap.add_argument('--data-dir', type=Path, required=True)
    ap.add_argument('--machine-id', type=int, required=True)
    ap.add_argument('--prediction-time', type=str, required=True)
    a = ap.parse_args()
    try:
        result = predict_saved(a.model, a.data_dir, a.machine_id, a.prediction_time)
    except (ValueError, FileNotFoundError) as exc:
        ap.error(str(exc))
    print(format_result(result))
