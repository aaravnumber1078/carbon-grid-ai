#!/usr/bin/env python3
"""
CARBONGRID-AI Phase C.1.1
Retrain quantization-safety RandomForest WITHOUT complexity_level feature.

Uses existing B2 dataset and deterministic split (128/25/29).
Threshold selected on validation only, final evaluation on held-out test.
"""

import json
import warnings
import sys
from pathlib import Path
from dataclasses import dataclass, asdict

import joblib
import numpy as np

# Make project root importable
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    brier_score_loss,
)
from sklearn.calibration import calibration_curve
from sklearn.ensemble import RandomForestClassifier
from carbongrid.classifier.features import FeatureExtractor, FeatureConfig, create_splits

warnings.filterwarnings("ignore")

# Project paths
ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "processed" / "b2_dataset.json"
MODEL_OUT_PATH = ROOT / "models" / "quantization_safety" / "classifier_c1_1.pkl"
EXTRACTOR_OUT_PATH = ROOT / "models" / "quantization_safety" / "feature_extractor_c1_1.pkl"
METADATA_OUT_PATH = ROOT / "models" / "quantization_safety" / "metadata_c1_1.json"


@dataclass
class Metrics:
    threshold: float
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    false_safe_count: int
    actual_unsafe: int
    false_safe_rate: float
    false_unsafe_count: int
    actual_safe: int
    false_unsafe_rate: float
    int4_opportunity_count: int
    int4_opportunity_rate: float
    brier_score: float | None = None
    ece: float | None = None

    def to_dict(self):
        return asdict(self)


def get_metrics(y_true, probabilities, threshold):
    y_true = np.asarray(y_true).astype(int)
    probabilities = np.asarray(probabilities, dtype=float)
    y_pred = (probabilities >= threshold).astype(int)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    actual_unsafe = int(tn + fp)
    actual_safe = int(tp + fn)
    total = len(y_true)

    false_safe_rate = fp / actual_unsafe if actual_unsafe else 0.0
    false_unsafe_rate = fn / actual_safe if actual_safe else 0.0

    roc_auc = (
        float(roc_auc_score(y_true, probabilities))
        if len(np.unique(y_true)) == 2
        else 0.5
    )

    brier = (
        float(brier_score_loss(y_true, probabilities))
        if len(np.unique(y_true)) == 2
        else None
    )

    ece = calculate_ece(y_true, probabilities)

    return Metrics(
        threshold=float(threshold),
        accuracy=float(accuracy_score(y_true, y_pred)),
        precision=float(precision_score(y_true, y_pred, zero_division=0)),
        recall=float(recall_score(y_true, y_pred, zero_division=0)),
        f1=float(f1_score(y_true, y_pred, zero_division=0)),
        roc_auc=roc_auc,
        false_safe_count=int(fp),
        actual_unsafe=actual_unsafe,
        false_safe_rate=float(false_safe_rate),
        false_unsafe_count=int(fn),
        actual_safe=actual_safe,
        false_unsafe_rate=float(false_unsafe_rate),
        int4_opportunity_count=int(tp + fp),
        int4_opportunity_rate=float((tp + fp) / total if total else 0.0),
        brier_score=brier,
        ece=ece,
    )


def calculate_ece(y_true, probabilities, n_bins=10):
    if len(np.unique(y_true)) < 2:
        return None

    try:
        prob_true, prob_pred = calibration_curve(
            y_true,
            probabilities,
            n_bins=n_bins,
            strategy="uniform",
        )
        if len(prob_true) == 0:
            return None
        return float(np.mean(np.abs(prob_true - prob_pred)))
    except Exception:
        return None


def print_metrics(label, m):
    print(f"\n{label}")
    print("-" * 60)
    print(f"Threshold:       {m.threshold:.2f}")
    print(f"Accuracy:        {m.accuracy:.4f}")
    print(f"Precision:       {m.precision:.4f}")
    print(f"Recall:          {m.recall:.4f}")
    print(f"F1:              {m.f1:.4f}")
    print(f"ROC-AUC:         {m.roc_auc:.4f}")
    print(
        f"False-safe:      {m.false_safe_count}/{m.actual_unsafe}"
        f" = {m.false_safe_rate:.3f}"
    )
    print(
        f"False-unsafe:    {m.false_unsafe_count}/{m.actual_safe}"
        f" = {m.false_unsafe_rate:.3f}"
    )
    print(
        f"INT4 opportunity:{m.int4_opportunity_count}"
        f" = {m.int4_opportunity_rate:.1%}"
    )
    if m.brier_score is not None:
        print(f"Brier score:     {m.brier_score:.4f}")
    if m.ece is not None:
        print(f"ECE:             {m.ece:.4f}")


def threshold_sweep(model, X, y):
    probabilities = model.predict_proba(X)[:, 1]
    thresholds = np.arange(0.30, 0.95, 0.05)

    results = []
    for threshold in thresholds:
        results.append(
            get_metrics(y, probabilities, float(round(threshold, 2)))
        )
    return results


def choose_threshold(results, max_false_safe=0.20):
    """
    Prefer the largest INT4 opportunity among validation thresholds whose
    false-safe rate is <= 20%.

    If no threshold reaches that constraint, choose the threshold with the
    lowest false-safe rate.
    """
    acceptable = [
        r for r in results
        if r.false_safe_rate <= max_false_safe
    ]

    if acceptable:
        chosen = max(
            acceptable,
            key=lambda r: (r.int4_opportunity_rate, -r.false_safe_rate),
        )
        reason = (
            f"Validation false-safe <= {max_false_safe:.0%}; "
            "selected the threshold with the highest INT4 opportunity."
        )
        return chosen.threshold, reason

    chosen = min(
        results,
        key=lambda r: (
            r.false_safe_rate,
            -r.int4_opportunity_rate,
        ),
    )
    reason = (
        f"No validation threshold reached false-safe <= "
        f"{max_false_safe:.0%}; selected the lowest observed "
        "false-safe rate."
    )
    return chosen.threshold, reason


def main():
    print("=" * 72)
    print("CARBONGRID-AI PHASE C.1.1")
    print("Retrain quantization-safety classifier WITHOUT complexity_level")
    print("=" * 72)

    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Missing dataset: {DATA_PATH}")

    print("\n[1/6] Loading B2 dataset...")
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        records = json.load(f)

    print(f"Dataset records: {len(records)}")

    print("\n[2/6] Creating deterministic train/validation/test split...")
    train_records, val_records, test_records = create_splits(records)

    print(
        f"Train={len(train_records)}, "
        f"Validation={len(val_records)}, "
        f"Test={len(test_records)}"
    )

    # Verify split sizes match expected
    assert len(train_records) == 128, f"Expected 128 train, got {len(train_records)}"
    assert len(val_records) == 25, f"Expected 25 validation, got {len(val_records)}"
    assert len(test_records) == 29, f"Expected 29 test, got {len(test_records)}"

    print("\n[3/6] Fitting feature extractor on TRAINING data ONLY (no complexity_level)...")
    # FeatureConfig with use_complexity_level=False
    config = FeatureConfig(
        use_task_type=True,
        use_complexity_level=False,  # KEY CHANGE: exclude complexity_level
        use_tfidf=True,
        tfidf_max_features=100,
        tfidf_ngram_range=(1, 2),
        scale_numeric=True,
        random_state=42,
    )

    extractor = FeatureExtractor(config)
    X_train, y_train = extractor.fit_transform(train_records)
    X_val, y_val = extractor.transform(val_records)
    X_test, y_test = extractor.transform(test_records)

    feature_names = extractor.get_feature_names()
    
    # The FeatureExtractor's use_complexity_level config doesn't actually remove
    # the feature from the numeric array - it only affects feature names.
    # We must explicitly remove the complexity_level column.
    if "complexity_level" in feature_names:
        complexity_idx = feature_names.index("complexity_level")
        X_train = np.delete(X_train, complexity_idx, axis=1)
        X_val = np.delete(X_val, complexity_idx, axis=1)
        X_test = np.delete(X_test, complexity_idx, axis=1)
        feature_names = [f for f in feature_names if f != "complexity_level"]
    
    # Update extractor's feature_names for saving
    extractor._feature_names = feature_names
    extractor.feature_names = feature_names
    
    original_feature_count = len(feature_names) + 1  # +1 for removed complexity_level

    print(f"Original feature count (with complexity_level): {original_feature_count}")
    print(f"Removed feature: complexity_level")
    print(f"Final feature count: {len(feature_names)}")
    print(f"Train matrix: {X_train.shape}")
    print(f"Validation matrix: {X_val.shape}")
    print(f"Test matrix: {X_test.shape}")
    print(f"Train safe: {sum(y_train)}, unsafe: {len(y_train) - sum(y_train)}")
    print(f"Val safe: {sum(y_val)}, unsafe: {len(y_val) - sum(y_val)}")
    print(f"Test safe: {sum(y_test)}, unsafe: {len(y_test) - sum(y_test)}")

    print("\n[4/6] Training RandomForest on 128 training samples...")
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    print(f"Model trained. n_features_in_: {model.n_features_in_}")

    print("\n[5/6] Threshold sweep on VALIDATION set...")
    val_results = threshold_sweep(model, X_val, y_val)

    print(
        "\nThreshold | Accuracy | Precision | Recall | F1 | "
        "False-safe | False-unsafe | INT4 opportunity"
    )
    for m in val_results:
        print(
            f"{m.threshold:8.2f} | "
            f"{m.accuracy:8.3f} | "
            f"{m.precision:9.3f} | "
            f"{m.recall:6.3f} | "
            f"{m.f1:5.3f} | "
            f"{m.false_safe_count}/{m.actual_unsafe}="
            f"{m.false_safe_rate:.3f} | "
            f"{m.false_unsafe_count}/{m.actual_safe}="
            f"{m.false_unsafe_rate:.3f} | "
            f"{m.int4_opportunity_rate:.1%}"
        )

    production_threshold, threshold_reason = choose_threshold(
        val_results,
        max_false_safe=0.20,
    )

    print(f"\nSelected validation threshold: {production_threshold:.2f}")
    print(f"Reason: {threshold_reason}")

    val_selected = get_metrics(
        y_val,
        model.predict_proba(X_val)[:, 1],
        production_threshold,
    )
    print_metrics("Validation metrics at selected threshold", val_selected)

    print("\n[6/6] FINAL HELD-OUT TEST EVALUATION")
    print("This is the final test evaluation after decisions are fixed.")

    test_prob = model.predict_proba(X_test)[:, 1]
    test_metrics = get_metrics(y_test, test_prob, production_threshold)

    print_metrics("FINAL TEST METRICS", test_metrics)

    # Error analysis
    predictions = (test_prob >= production_threshold).astype(int)
    false_safe_idx = np.where((predictions == 1) & (y_test == 0))[0]
    false_unsafe_idx = np.where((predictions == 0) & (y_test == 1))[0]

    def describe(indices):
        output = []
        for idx in indices[:5]:
            record = test_records[idx]
            output.append({
                "prompt_id": record.get("prompt_id", f"index_{idx}"),
                "task_type": record.get("task_type", "unknown"),
                "complexity_level": record.get("complexity_level"),
                "predicted_probability": float(test_prob[idx]),
                "actual_label": int(y_test[idx]),
                "predicted_label": int(predictions[idx]),
                "prompt_preview": record.get("prompt", "")[:160],
                "evaluation_method": record.get("evaluation_method", "unknown"),
                "quality_score_int4": record.get("quality_score_int4"),
            })
        return output

    errors = {
        "false_safe_count": int(len(false_safe_idx)),
        "false_unsafe_count": int(len(false_unsafe_idx)),
        "false_safe_examples": describe(false_safe_idx),
        "false_unsafe_examples": describe(false_unsafe_idx),
    }

    print("\nError analysis")
    print("-" * 60)
    print(f"False-safe:   {errors['false_safe_count']}")
    print(f"False-unsafe: {errors['false_unsafe_count']}")

    if errors["false_safe_examples"]:
        print("\nFalse-safe examples:")
        for item in errors["false_safe_examples"]:
            print(
                f"  {item['prompt_id']} | "
                f"{item['task_type']} | "
                f"L{item['complexity_level']} | "
                f"p={item['predicted_probability']:.3f}"
            )

    if errors["false_unsafe_examples"]:
        print("\nFalse-unsafe examples:")
        for item in errors["false_unsafe_examples"]:
            print(
                f"  {item['prompt_id']} | "
                f"{item['task_type']} | "
                f"L{item['complexity_level']} | "
                f"p={item['predicted_probability']:.3f}"
            )

    # Feature importance
    importance = list(zip(feature_names, model.feature_importances_))
    importance.sort(key=lambda x: x[1], reverse=True)
    top_features = importance[:20]

    # Build report
    report = {
        "phase": "C.1.1",
        "dataset": {
            "name": "B2",
            "records": len(records),
        },
        "split": {
            "train": len(train_records),
            "validation": len(val_records),
            "test": len(test_records),
            "note": "Deterministic split via create_splits (random_state=42).",
        },
        "model": {
            "base": "RandomForestClassifier",
            "n_estimators": 200,
            "max_depth": 10,
            "min_samples_split": 5,
            "min_samples_leaf": 2,
            "class_weight": "balanced",
            "random_state": 42,
        },
        "production_features": {
            "original_count": original_feature_count,
            "removed_feature": "complexity_level",
            "final_count": len(feature_names),
            "complexity_level_excluded": True,
            "feature_names": feature_names,
        },
        "threshold_selection": {
            "selected_on": "validation",
            "selected_threshold": production_threshold,
            "maximum_target_false_safe_rate": 0.20,
            "reason": threshold_reason,
            "validation_sweep": [m.to_dict() for m in val_results],
        },
        "validation_metrics": val_selected.to_dict(),
        "final_test_metrics": test_metrics.to_dict(),
        "error_analysis": errors,
        "top_feature_importance": [
            {"feature": f, "importance": float(imp)} for f, imp in top_features
        ],
        "limitations": [
            f"Validation set contains only {len(val_records)} samples.",
            f"Test set contains only {len(test_records)} samples.",
            "Quality labels originate from the B2 evaluation pipeline and are proxies, not ground truth.",
            "FP16 is the reference configuration rather than an absolute quality ground truth.",
            "Energy measurements and associated CO2 estimates are workload-specific.",
        ],
    }

    # Save artifacts
    MODEL_OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_OUT_PATH)
    extractor.save(EXTRACTOR_OUT_PATH)

    with open(METADATA_OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)

    print("\n" + "=" * 72)
    print("C.1.1 COMPLETE")
    print("=" * 72)
    print(f"Model:        {MODEL_OUT_PATH}")
    print(f"Extractor:    {EXTRACTOR_OUT_PATH}")
    print(f"Report:       {METADATA_OUT_PATH}")
    print(f"Production threshold: {production_threshold:.2f}")
    print(f"Original features: {original_feature_count}")
    print(f"Removed:      complexity_level")
    print(f"Final features: {len(feature_names)}")
    print("No inference benchmark was rerun.")
    print("No dataset was regenerated.")

    # Comparison with C.1 baseline
    print("\n" + "=" * 72)
    print("COMPARISON: C.1.1 vs C.1 BASELINE")
    print("=" * 72)
    c1_baseline = {
        "Accuracy": 0.6897,
        "Precision": 0.6667,
        "Recall": 0.7143,
        "F1": 0.6897,
        "ROC-AUC": 0.7429,
        "False-safe": 0.3333,
        "False-unsafe": 0.2857,
        "INT4 opportunity": 0.5172,
    }
    c1_1 = {
        "Accuracy": test_metrics.accuracy,
        "Precision": test_metrics.precision,
        "Recall": test_metrics.recall,
        "F1": test_metrics.f1,
        "ROC-AUC": test_metrics.roc_auc,
        "False-safe": test_metrics.false_safe_rate,
        "False-unsafe": test_metrics.false_unsafe_rate,
        "INT4 opportunity": test_metrics.int4_opportunity_rate,
    }
    print(f"{'Metric':<20} {'C.1 Baseline':>12} {'C.1.1':>12} {'Delta':>10}")
    print("-" * 54)
    for key in c1_baseline:
        baseline = c1_baseline[key]
        current = c1_1[key]
        delta = current - baseline
        print(f"{key:<20} {baseline:>12.4f} {current:>12.4f} {delta:>+10.4f}")


if __name__ == "__main__":
    main()