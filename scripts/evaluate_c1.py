#!/usr/bin/env python3
"""
CarbonGrid-AI - Phase C.1
Corrected evaluation of the existing quantization-safety classifier.

IMPORTANT:
- Uses the existing B2 dataset and existing classifier/feature-extractor artifacts.
- Does NOT regenerate data.
- Does NOT run FP16/INT4 inference.
- Threshold is selected using validation only.
- Held-out test metrics are reported after the production decision is fixed.
- complexity_level is excluded from the production feature set.
"""

import json
import warnings
import sys
from pathlib import Path
from dataclasses import dataclass, asdict

import joblib
import numpy as np

# Make the project root importable BEFORE importing the carbongrid package.
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
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from carbongrid.classifier.features import FeatureExtractor, FeatureConfig, create_splits

warnings.filterwarnings("ignore")

# Project paths
ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "processed" / "b2_dataset.json"
MODEL_PATH = ROOT / "models" / "quantization_safety" / "classifier.pkl"
EXTRACTOR_PATH = ROOT / "models" / "quantization_safety" / "feature_extractor.pkl"
OUT_DIR = ROOT / "models" / "quantization_safety"
REPORT_PATH = OUT_DIR / "metadata_c1.json"
CALIBRATED_PATH = OUT_DIR / "calibrated_classifier.pkl"


def load_project_features():
    """Return the project's existing split function."""
    return create_splits


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
    best conservative trade-off. This is a validation-set decision only.
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


def fit_platt_on_validation(raw_model, X_val, y_val):
    """
    Platt/sigmoid calibration using validation predictions.

    The raw classifier remains available. Calibration is adopted only if
    validation Brier score improves over the raw probabilities.
    """
    raw_val_prob = raw_model.predict_proba(X_val)[:, 1]

    # Logistic regression maps the raw RF score to a calibrated probability.
    # It is deliberately simple because the validation set is small.
    calibrator = LogisticRegression(
        solver="lbfgs",
        max_iter=1000,
        random_state=42,
    )

    try:
        calibrator.fit(raw_val_prob.reshape(-1, 1), y_val)
    except Exception as exc:
        print(f"Platt calibration could not be fitted: {exc}")
        return None

    calibrated_val_prob = calibrator.predict_proba(
        raw_val_prob.reshape(-1, 1)
    )[:, 1]

    raw_brier = brier_score_loss(y_val, raw_val_prob)
    calibrated_brier = brier_score_loss(y_val, calibrated_val_prob)

    print("\nCalibration check on validation set")
    print("-" * 60)
    print(f"Raw Brier:       {raw_brier:.6f}")
    print(f"Platt Brier:     {calibrated_brier:.6f}")

    if calibrated_brier < raw_brier:
        print("Decision: Platt calibration improves validation Brier score.")
        return calibrator

    print("Decision: keep raw RandomForest probabilities.")
    return None


def calibrated_probability(model, calibrator, X):
    raw = model.predict_proba(X)[:, 1]
    if calibrator is None:
        return raw
    return calibrator.predict_proba(raw.reshape(-1, 1))[:, 1]


def run_ablation(X_val, y_val, feature_names):
    """
    Descriptive validation-only ablation.

    This intentionally trains small RF models on the validation subset only
    to compare feature groups descriptively. These models are NOT used as the
    production model and do NOT touch the held-out test set.
    """
    names = list(feature_names)
    complexity_idx = (
        names.index("complexity_level")
        if "complexity_level" in names
        else None
    )
    task_idx = names.index("task_type") if "task_type" in names else None

    tfidf_idx = [
        i for i, n in enumerate(names) if n.startswith("tfidf_")
    ]

    all_no_complexity = [
        i for i in range(len(names))
        if i != complexity_idx
    ]

    basic_no_tfidf = [
        i for i in range(len(names))
        if i != complexity_idx and i not in tfidf_idx
    ]

    task_only = [task_idx] if task_idx is not None else []

    configs = {
        "task_only": task_only,
        "basic_no_tfidf": basic_no_tfidf,
        "all_no_complexity": all_no_complexity,
        "all_features": list(range(len(names))),
    }

    results = {}

    print("\nValidation feature ablation (descriptive only)")
    print("-" * 90)

    for name, indices in configs.items():
        if not indices:
            continue

        X_sub = X_val[:, indices]

        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )
        model.fit(X_sub, y_val)

        p = model.predict_proba(X_sub)[:, 1]
        m = get_metrics(y_val, p, 0.50)

        results[name] = {
            "features": len(indices),
            **m.to_dict(),
        }

        print(
            f"{name:20s} | "
            f"features={len(indices):3d} | "
            f"acc={m.accuracy:.3f} | "
            f"F1={m.f1:.3f} | "
            f"ROC-AUC={m.roc_auc:.3f} | "
            f"false-safe={m.false_safe_rate:.3f}"
        )

    return results


def error_analysis(model, calibrator, X_test, y_test, records, threshold):
    probabilities = calibrated_probability(model, calibrator, X_test)
    predictions = (probabilities >= threshold).astype(int)

    false_safe = np.where((predictions == 1) & (y_test == 0))[0]
    false_unsafe = np.where((predictions == 0) & (y_test == 1))[0]

    def describe(indices):
        output = []
        for idx in indices[:5]:
            record = records[idx]
            output.append(
                {
                    "prompt_id": record.get("prompt_id", f"index_{idx}"),
                    "task_type": record.get("task_type", "unknown"),
                    "complexity_level": record.get("complexity_level"),
                    "predicted_probability": float(probabilities[idx]),
                    "actual_label": int(y_test[idx]),
                    "predicted_label": int(predictions[idx]),
                    "prompt_preview": record.get("prompt", "")[:160],
                    "evaluation_method": record.get(
                        "evaluation_method", "unknown"
                    ),
                    "quality_score_int4": record.get(
                        "quality_score_int4"
                    ),
                }
            )
        return output

    return {
        "false_safe_count": int(len(false_safe)),
        "false_unsafe_count": int(len(false_unsafe)),
        "false_safe_examples": describe(false_safe),
        "false_unsafe_examples": describe(false_unsafe),
    }


def main():
    print("=" * 72)
    print("CARBONGRID-AI PHASE C.1")
    print("Corrected quantization-safety classifier evaluation")
    print("=" * 72)

    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Missing dataset: {DATA_PATH}")
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Missing classifier: {MODEL_PATH}")
    if not EXTRACTOR_PATH.exists():
        raise FileNotFoundError(f"Missing feature extractor: {EXTRACTOR_PATH}")

    print("\n[1/7] Loading existing artifacts...")
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        records = json.load(f)

    raw_model = joblib.load(MODEL_PATH)
    create_splits = load_project_features()

    print(f"Dataset records: {len(records)}")
    print(f"Classifier: {type(raw_model).__name__}")

    print("\n[2/7] Reconstructing existing train/validation/test split...")
    train_records, val_records, test_records = create_splits(records)

    print(
        f"Train={len(train_records)}, "
        f"Validation={len(val_records)}, "
        f"Test={len(test_records)}"
    )

    print("\n[3/7] Restoring the FITTED feature extractor artifact...")
    # feature_extractor.pkl is a dictionary containing the already-fitted
    # FeatureConfig, scaler, task encoder, TF-IDF vectorizer, feature names,
    # and fitted flag. Reconstruct the FeatureExtractor without fitting it.
    saved_extractor = joblib.load(EXTRACTOR_PATH)

    if not isinstance(saved_extractor, dict):
        raise TypeError(
            "Expected feature_extractor.pkl to contain a dict, "
            f"got {type(saved_extractor).__name__}."
        )

    required_keys = {
        "config",
        "scaler",
        "task_encoder",
        "tfidf",
        "feature_names",
        "fitted",
    }
    missing = required_keys - set(saved_extractor.keys())
    if missing:
        raise KeyError(
            f"feature_extractor.pkl is missing keys: {sorted(missing)}"
        )

    # FeatureExtractor.transform() expects these fitted attributes.
    extractor = FeatureExtractor(saved_extractor["config"])
    extractor.scaler = saved_extractor["scaler"]
    extractor.task_encoder = saved_extractor["task_encoder"]
    extractor.tfidf = saved_extractor["tfidf"]
    extractor._feature_names = saved_extractor["feature_names"]
    extractor._fitted = bool(saved_extractor["fitted"])

    X_train, y_train = extractor.transform(train_records)
    X_val, y_val = extractor.transform(val_records)
    X_test, y_test = extractor.transform(test_records)

    feature_names = list(saved_extractor["feature_names"])
    complexity_present = "complexity_level" in feature_names

    print(f"Feature count: {len(feature_names)}")
    print(f"complexity_level present: {complexity_present}")
    print(f"Train matrix: {X_train.shape}")
    print(f"Validation matrix: {X_val.shape}")
    print(f"Test matrix: {X_test.shape}")

    # The existing classifier was trained with the existing 123-feature
    # artifact. Do not silently remove complexity_level here, because doing so
    # would change the feature dimensionality and invalidate the stored model.
    expected = getattr(raw_model, "n_features_in_", None)
    if expected is not None and expected != X_val.shape[1]:
        raise RuntimeError(
            "Feature dimension mismatch: existing classifier expects "
            f"{expected} features, but the restored extractor produces "
            f"{X_val.shape[1]} features."
        )

    X_val_prod = X_val
    X_test_prod = X_test
    production_features = feature_names

    print(
        "NOTE: the existing stored classifier uses all "
        f"{len(feature_names)} features, including complexity_level "
        "if present. Complexity exclusion is evaluated separately in "
        "the validation ablation; it cannot be applied to this stored "
        "classifier without retraining."
    )
    print(f"Validation size: {len(y_val)} (small; estimates are unstable)")
    print(f"Test size: {len(y_test)}")

    print("\n[4/7] Selecting threshold using VALIDATION ONLY...")
    val_results = threshold_sweep(raw_model, X_val_prod, y_val)

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

    print("\n[5/7] Testing Platt/sigmoid calibration on validation...")
    calibrator = fit_platt_on_validation(
        raw_model,
        X_val_prod,
        y_val,
    )

    if calibrator is not None:
        calibration_name = "Platt/sigmoid"
    else:
        calibration_name = "None (raw RandomForest)"

    val_prob = calibrated_probability(
        raw_model, calibrator, X_val_prod
    )
    val_selected = get_metrics(
        y_val,
        val_prob,
        production_threshold,
    )
    print_metrics(
        "Validation metrics at selected threshold",
        val_selected,
    )

    print("\n[6/7] Running validation-only feature ablation...")
    ablation_results = run_ablation(
        X_val,
        y_val,
        feature_names,
    )

    print("\n[7/7] FINAL HELD-OUT TEST EVALUATION")
    print("This is the final test evaluation after decisions are fixed.")

    test_prob = calibrated_probability(
        raw_model,
        calibrator,
        X_test_prod,
    )

    test_metrics = get_metrics(
        y_test,
        test_prob,
        production_threshold,
    )

    print_metrics(
        "FINAL TEST METRICS",
        test_metrics,
    )

    errors = error_analysis(
        raw_model,
        calibrator,
        X_test_prod,
        y_test,
        test_records,
        production_threshold,
    )

    print("\nError analysis")
    print("-" * 60)
    print(
        f"False-safe:   {errors['false_safe_count']}"
    )
    print(
        f"False-unsafe: {errors['false_unsafe_count']}"
    )

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

    report = {
        "phase": "C.1",
        "dataset": {
            "name": "B2",
            "records": len(records),
        },
        "split": {
            "train": len(train_records),
            "validation": len(val_records),
            "test": len(test_records),
            "note": "Existing project split reconstructed with create_splits.",
        },
        "model": {
            "base": type(raw_model).__name__,
            "calibration": calibration_name,
        },
        "production_features": {
            "count": len(production_features),
            "complexity_level_excluded": "complexity_level" not in production_features,
            "note": "Existing classifier evaluated with its original feature set; excluding complexity_level requires retraining.",
        },
        "threshold_selection": {
            "selected_on": "validation",
            "selected_threshold": production_threshold,
            "maximum_target_false_safe_rate": 0.20,
            "reason": threshold_reason,
            "validation_sweep": [
                m.to_dict() for m in val_results
            ],
        },
        "validation_metrics": val_selected.to_dict(),
        "ablation_validation_only": ablation_results,
        "final_test_metrics": test_metrics.to_dict(),
        "error_analysis": errors,
        "limitations": [
            f"Validation set contains only {len(val_records)} samples.",
            f"Test set contains only {len(test_records)} samples.",
            "Quality labels originate from the B2 evaluation pipeline and are proxies, not ground truth.",
            "FP16 is the reference configuration rather than an absolute quality ground truth.",
            "Energy measurements and associated CO2 estimates are workload-specific.",
            "complexity_level remains in the existing stored classifier; a separate ablation evaluates exclusion, and a production model without it would require retraining.",
        ],
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)

    if calibrator is not None:
        joblib.dump(calibrator, CALIBRATED_PATH)

    print("\n" + "=" * 72)
    print("C.1 COMPLETE")
    print("=" * 72)
    print(f"Report: {REPORT_PATH}")
    if calibrator is not None:
        print(f"Platt calibrator: {CALIBRATED_PATH}")
    print(f"Production threshold: {production_threshold:.2f}")
    print(f"Calibration: {calibration_name}")
    print("No inference benchmark was rerun.")
    print("No dataset was regenerated.")


if __name__ == "__main__":
    main()
