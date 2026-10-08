#!/usr/bin/env python3
"""
Phase F.1: Energy Predictor Evaluation

Loads the trained energy predictor and evaluates it on the locked test set.
Computes all required metrics and compares against static baseline.
"""

import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Tuple
from collections import defaultdict
from dataclasses import dataclass, asdict

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, confusion_matrix
)
import joblib

# Constants
RESULTS_DIR = Path("data/evaluation/phase_f")
MODEL_DIR = Path("models/energy_predictor")
DATA_PATH = Path("data/processed/b2_dataset.json")

# Static baseline from Phase 0.5
STATIC_FP16_ENERGY = 0.0488
STATIC_INT4_ENERGY = 0.0493
STATIC_DIFF = STATIC_INT4_ENERGY - STATIC_FP16_ENERGY


# Feature constants (must match training)
NUMERIC_FEATURES = [
    "char_length", "word_count", "token_count_estimate",
    "has_question_mark", "has_code_indicator", "question_count", "sentence_count",
    "has_scientific_terms", "scientific_term_count", "has_reasoning_indicators",
    "reasoning_indicator_count", "has_code_keywords", "code_keyword_count",
    "avg_word_length", "unique_word_ratio", "has_numbers", "number_count",
    "requested_length_hint", "max_tokens_requested",
]


def load_test_data() -> Tuple[pd.DataFrame, np.ndarray, List[str]]:
    """Load test data and return X_test, y_test, test_ids."""
    # Load test split
    with open("data/evaluation/phase_f/energy_predictor_test_split.json") as f:
        split_info = json.load(f)
    test_ids = split_info["test_ids"]
    
    # Load B2 data
    with open("data/processed/b2_dataset.json") as f:
        data = json.load(f)
    
    # Build test set
    test_records = []
    for r in data:
        if r["prompt_id"] not in test_ids:
            continue
        if not r.get("measurement_status") == "complete":
            continue
        
        feat = r.get("features", {})
        row = {"prompt_id": r["prompt_id"]}
        
        for feat_name in NUMERIC_FEATURES:
            val = feat.get(feat_name, 0.0)
            
            # Apply same preprocessing as training (matching train_energy_predictor.py)
            if feat_name == "requested_length_hint":
                if val is None or val == "":
                    val = 0.0
                elif isinstance(val, str):
                    val = val.lower()
                    if val in ["short", "brief", "concise"]:
                        val = 1.0
                    elif val in ["detailed", "comprehensive"]:
                        val = 2.0
                    elif "sentence" in val:
                        import re
                        m = re.search(r'(\d+)', val)
                        if m:
                            val = float(m.group(1)) * 20.0
                        else:
                            val = 20.0
            elif feat_name == "max_tokens_requested":
                if val is None or (isinstance(val, float) and np.isnan(val)):
                    val = 0.0
            
            row[feat_name] = float(val)
        
        row["task_type"] = r.get("task_type", "unknown")
        row["complexity_level"] = r.get("complexity_level", 1)
        row["fp16_energy"] = r.get("fp16_energy_wh", 0.0)
        row["int4_energy"] = r.get("int4_energy_wh", 0.0)
        row["energy_diff"] = row["int4_energy"] - row["fp16_energy"]
        
        test_records.append(row)
    
    test_df = pd.DataFrame(test_records)
    test_ids_ordered = test_df["prompt_id"].tolist()
    
    return test_df, test_df["energy_diff"].values, test_ids_ordered


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute all evaluation metrics."""
    directional_acc = np.mean(np.sign(y_pred) == np.sign(y_true))
    
    rel_error = np.abs(y_pred - y_true) / (np.abs(y_true) + 1e-10)
    within_10 = np.mean(rel_error <= 0.10)
    
    y_true_sign = np.sign(y_true)
    y_pred_sign = np.sign(y_pred)
    cm = confusion_matrix(y_true_sign, y_pred_sign, labels=[-1, 1])
    tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)
    
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
        "directional_accuracy": float(directional_acc),
        "within_10_pct": float(within_10),
        "confusion_matrix": {
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)
        },
        "sign_agreement": float(np.mean(y_true_sign == y_pred_sign)),
    }


def static_baseline_predictions(y_true: np.ndarray) -> np.ndarray:
    """Static baseline always predicts the Phase 0.5 difference."""
    return np.full_like(y_true, STATIC_DIFF)


def main():
    print("=" * 70)
    print("PHASE F.1: ENERGY PREDICTOR EVALUATION")
    print("=" * 70)
    
    # Load model
    model_path = Path("models/energy_predictor/energy_predictor.joblib")
    if not model_path.exists():
        print(f"Error: Model not found at {model_path}")
        print("Run train_energy_predictor.py first.")
        return 1
    
    print(f"\nLoading model from {model_path}...")
    pipeline = joblib.load(model_path)
    
    # Load test data
    print("Loading test data...")
    test_df, y_test, test_ids = load_test_data()
    print(f"Test samples: {len(test_df)}")
    
    X_test = test_df[NUMERIC_FEATURES + ["task_type"]]
    y_test = test_df["energy_diff"].values
    
    # Predict
    print("Running predictions...")
    y_pred = pipeline.predict(X_test)
    
    # Compute metrics
    metrics = evaluate_predictions(y_test, y_pred)
    baseline_metrics = evaluate_predictions(y_test, static_baseline_predictions(y_test))
    
    # Print results
    print("\n" + "=" * 70)
    print("ENERGY PREDICTOR EVALUATION RESULTS")
    print("=" * 70)
    
    print(f"\nTest samples: {len(y_test)}")
    print(f"\nEnergy Predictor Performance:")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.6f}")
        else:
            print(f"  {k}: {v}")
    
    print(f"\nStatic Baseline (Phase 0.5 profile):")
    for k, v in baseline_metrics.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.6f}")
        else:
            print(f"  {k}: {v}")
    
    print(f"\nImprovement over Baseline:")
    for k in metrics:
        if k in baseline_metrics and isinstance(metrics[k], float) and isinstance(baseline_metrics[k], float):
            diff = metrics[k] - baseline_metrics[k]
            if k in ["mae", "rmse"]:
                print(f"  {k}: {diff:+.6f} (negative is better)")
            else:
                print(f"  {k}: {diff:+.6f}")
    
    # Success criterion
    print(f"\nSuccess Criterion:")
    mae_met = metrics["mae"] < 0.002
    dir_acc_met = metrics["directional_accuracy"] > 0.70
    print(f"  MAE < 0.002 Wh: {metrics['mae']:.6f} -> {'PASS' if mae_met else 'FAIL'}")
    print(f"  Directional Accuracy > 70%: {metrics['directional_accuracy']:.2%} -> {'PASS' if dir_acc_met else 'FAIL'}")
    print(f"  Overall: {'PASS' if (mae_met and dir_acc_met) else 'FAIL'}")
    
    # Confusion matrix
    y_true_sign = np.sign(y_test)
    y_pred_sign = np.sign(y_pred)
    cm = confusion_matrix(y_true_sign, y_pred_sign, labels=[-1, 1])
    tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)
    
    print(f"\nConfusion Matrix (lower energy config):")
    print(f"  True FP16 lower (TN): {tn}")
    print(f"  False FP16 lower (FP): {fp}")
    print(f"  False INT4 lower (FN): {fn}")
    print(f"  True INT4 lower (TP): {tp}")
    
    # Baseline confusion
    baseline_sign = np.sign(np.full_like(y_test, STATIC_DIFF))
    cm_baseline = confusion_matrix(y_true_sign, baseline_sign, labels=[-1, 1])
    tn_b, fp_b, fn_b, tp_b = cm_baseline.ravel() if cm_baseline.shape == (2, 2) else (0, 0, 0, 0)
    print(f"\nBaseline Confusion Matrix:")
    print(f"  TN: {tn_b}, FP: {fp_b}, FN: {fn_b}, TP: {tp_b}")
    
    # Per-task analysis
    print("\n" + "=" * 70)
    print("PER-TASK ANALYSIS")
    print("=" * 70)
    
    for task in sorted(test_df["task_type"].unique()):
        mask = test_df["task_type"] == task
        if mask.sum() < 2:
            continue
        y_t = y_test[mask]
        y_p = y_pred[mask]
        m = evaluate_predictions(y_t, y_p)
        print(f"  {task}: n={mask.sum()}, MAE={m['mae']:.6f}, DirAcc={m['directional_accuracy']:.2%}, diff={np.mean(y_pred[mask] - y_t):.6f} Wh")
    
    # Save evaluation results
    results = {
        "metrics": metrics,
        "baseline_metrics": baseline_metrics,
        "improvement": {
            k: float(metrics[k] - baseline_metrics[k]) if isinstance(metrics[k], float) and isinstance(baseline_metrics[k], float) else metrics[k]
            for k in metrics if k in baseline_metrics
        },
        "success_criterion": {
            "mae_threshold": 0.002,
            "directional_accuracy_threshold": 0.70,
            "mae_met": metrics["mae"] < 0.002,
            "directional_accuracy_met": metrics["directional_accuracy"] > 0.70,
            "overall_success": (metrics["mae"] < 0.002) and (metrics["directional_accuracy"] > 0.70),
        },
        "confusion_matrix": {
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)
        },
        "baseline_confusion": {
            "tn": int(tn_b), "fp": int(fp_b), "fn": int(fn_b), "tp": int(tp_b)
        },
    }
    
    # Save evaluation results
    output_dir = Path("data/evaluation/phase_f")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / "energy_predictor_evaluation.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\nEvaluation results saved to data/evaluation/phase_f/energy_predictor_evaluation.json")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())