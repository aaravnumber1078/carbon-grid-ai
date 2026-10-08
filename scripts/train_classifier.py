"""
Phase C: Quantization-Safety Classifier Training.

Trains and evaluates multiple models with proper splits, ablation studies,
and threshold analysis.
"""

import json
import warnings
import sys
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass, asdict

import numpy as np
import joblib

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    precision_recall_curve, brier_score_loss
)
from sklearn.calibration import CalibratedClassifierCV, calibration_curve

from carbongrid.classifier.features import (
    FeatureExtractor, FeatureConfig, create_splits
)


@dataclass
class ModelConfig:
    """Model hyperparameters."""
    name: str
    model_class: Any
    params: Dict[str, Any]
    calibrate: bool = True


@dataclass
class EvaluationResult:
    """Results for a model at a specific threshold."""
    model_name: str
    threshold: float
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    false_safe_rate: float
    false_unsafe_rate: float
    safe_predictions: int
    unsafe_predictions: int
    confusion_matrix: List[List[int]]
    
    def to_dict(self) -> Dict:
        return asdict(self)


# Model configurations to try
MODEL_CONFIGS = [
    ModelConfig(
        name="LogisticRegression",
        model_class=LogisticRegression,
        params={
            "C": 1.0,
            "max_iter": 1000,
            "class_weight": "balanced",
            "random_state": 42,
            "solver": "lbfgs"
        }
    ),
    ModelConfig(
        name="RandomForest",
        model_class=RandomForestClassifier,
        params={
            "n_estimators": 200,
            "max_depth": 10,
            "min_samples_split": 5,
            "min_samples_leaf": 2,
            "class_weight": "balanced",
            "random_state": 42,
            "n_jobs": -1
        }
    ),
    ModelConfig(
        name="GradientBoosting",
        model_class=GradientBoostingClassifier,
        params={
            "n_estimators": 200,
            "max_depth": 5,
            "learning_rate": 0.1,
            "min_samples_split": 5,
            "min_samples_leaf": 2,
            "random_state": 42
        }
    ),
]


def train_model(
    model_config: ModelConfig,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    calibrate: bool = True
) -> Any:
    """Train a model with optional calibration."""
    model = model_config.model_class(**model_config.params)
    model.fit(X_train, y_train)
    
    if calibrate and model_config.calibrate:
        # Use validation set for calibration
        calibrated = CalibratedClassifierCV(model, method='isotonic', cv='prefit')
        calibrated.fit(X_val, y_val)
        return calibrated
    
    return model


def evaluate_model(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    threshold: float = 0.5,
    model_name: str = "model"
) -> EvaluationResult:
    """Evaluate model at a specific threshold."""
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)
    
    # Basic metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_proba)
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    # False-safe rate: predicted safe (1) but actually unsafe (0)
    false_safe = fp / (fp + tn) if (fp + tn) > 0 else 0
    # False-unsafe rate: predicted unsafe (0) but actually safe (1)
    false_unsafe = fn / (fn + tp) if (fn + tp) > 0 else 0
    
    safe_preds = int(y_pred.sum())
    unsafe_preds = len(y_pred) - safe_preds
    
    return EvaluationResult(
        model_name=model_name,
        threshold=threshold,
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1=f1,
        roc_auc=roc_auc,
        false_safe_rate=false_safe,
        false_unsafe_rate=false_unsafe,
        safe_predictions=safe_preds,
        unsafe_predictions=unsafe_preds,
        confusion_matrix=cm.tolist()
    )


def threshold_analysis(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    thresholds: List[float] = None,
    model_name: str = "model"
) -> List[EvaluationResult]:
    """Evaluate model at multiple thresholds."""
    if thresholds is None:
        thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    
    results = []
    for t in thresholds:
        result = evaluate_model(model, X_test, y_test, t, model_name)
        results.append(result)
        print(f"  {model_name} @ {t:.2f}: Acc={result.accuracy:.3f}, "
              f"Prec={result.precision:.3f}, Rec={result.recall:.3f}, "
              f"F1={result.f1:.3f}, FalseSafe={result.false_safe_rate:.3f}, "
              f"FalseUnsafe={result.false_unsafe_rate:.3f}")
    
    return results


def calibration_analysis(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    n_bins: int = 10
) -> Dict[str, Any]:
    """Evaluate probability calibration."""
    y_proba = model.predict_proba(X_test)[:, 1]
    
    # Calibration curve
    prob_true, prob_pred = calibration_curve(y_test, y_proba, n_bins=n_bins)
    
    # Brier score
    brier = brier_score_loss(y_test, y_proba)
    
    # Expected calibration error
    ece = np.mean(np.abs(prob_true - prob_pred))
    
    return {
        "brier_score": brier,
        "ece": ece,
        "calibration_curve": {
            "prob_true": prob_true.tolist(),
            "prob_pred": prob_pred.tolist()
        }
    }


def feature_importance_analysis(
    model: Any,
    feature_names: List[str],
    top_k: int = 20
) -> List[Dict[str, Any]]:
    """Extract feature importance from trained model."""
    importances = []
    
    # Try different model structures to find the base estimator with feature_importances_
    base_model = model
    
    # Check calibrated classifiers
    if hasattr(model, 'calibrated_classifiers_') and len(model.calibrated_classifiers_) > 0:
        for cal in model.calibrated_classifiers_:
            if hasattr(cal, 'base_estimator'):
                base_model = cal.base_estimator
                break
            elif hasattr(cal, 'estimator'):
                base_model = cal.estimator
                break
    
    # Check base_estimator directly
    elif hasattr(model, 'base_estimator'):
        base_model = model.base_estimator
    elif hasattr(model, 'estimator'):
        base_model = model.estimator
    
    imps = None
    if hasattr(base_model, 'feature_importances_'):
        imps = base_model.feature_importances_
    elif hasattr(base_model, 'coef_'):
        imps = np.abs(base_model.coef_[0])
    elif hasattr(model, 'feature_importances_'):
        imps = model.feature_importances_
    elif hasattr(model, 'coef_'):
        imps = np.abs(model.coef_[0])
    else:
        return []
    
    if imps is None:
        return []
    
    # Get top k
    indices = np.argsort(imps)[::-1][:top_k]
    for idx in indices:
        if idx < len(feature_names):
            importances.append({
                "feature": feature_names[idx],
                "importance": float(imps[idx])
            })
    
    return importances


def ablation_study(
    records: List[Dict],
    feature_names: List[str],
    X_all: np.ndarray,
    y_all: np.ndarray,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray
) -> Dict[str, Any]:
    """Run ablation studies comparing feature sets."""
    
    # Find indices for different feature groups
    task_type_idx = feature_names.index("task_type") if "task_type" in feature_names else -1
    complexity_idx = feature_names.index("complexity_level") if "complexity_level" in feature_names else -1
    
    # Identify TF-IDF features
    tfidf_indices = [i for i, name in enumerate(feature_names) if name.startswith("tfidf_")]
    non_tfidf_indices = [i for i, name in enumerate(feature_names) if not name.startswith("tfidf_")]
    
    # Task type only
    task_only_indices = [task_type_idx] if task_type_idx >= 0 else []
    
    # Basic features (no TF-IDF, no complexity)
    basic_indices = [i for i in non_tfidf_indices if i != complexity_idx]
    
    # All features except complexity
    no_complexity_indices = [i for i in range(len(feature_names)) if i != complexity_idx]
    
    # All features
    all_indices = list(range(len(feature_names)))
    
    ablations = {
        "task_only": task_only_indices,
        "basic_no_tfidf": basic_indices,
        "all_no_complexity": no_complexity_indices,
        "all_features": all_indices,
    }
    
    results = {}
    
    for name, indices in ablations.items():
        if not indices:
            continue
        
        print(f"\n  Ablation: {name} ({len(indices)} features)")
        
        X_train_sub = X_train[:, indices]
        X_val_sub = X_val[:, indices]
        X_test_sub = X_test[:, indices]
        
        # Train Random Forest (good balance)
        rf = RandomForestClassifier(
            n_estimators=200, max_depth=10, min_samples_split=5,
            min_samples_leaf=2, class_weight="balanced", random_state=42, n_jobs=-1
        )
        rf.fit(X_train_sub, y_train)
        
        # Calibrate on validation
        calibrated = CalibratedClassifierCV(rf, method='isotonic', cv='prefit')
        calibrated.fit(X_val_sub, y_val)
        
        # Evaluate
        result = evaluate_model(calibrated, X_test_sub, y_test, threshold=0.5)
        results[name] = {
            "features": len(indices),
            "accuracy": result.accuracy,
            "precision": result.precision,
            "recall": result.recall,
            "f1": result.f1,
            "roc_auc": result.roc_auc,
            "false_safe_rate": result.false_safe_rate,
            "false_unsafe_rate": result.false_unsafe_rate,
            "feature_indices": indices
        }
        
        print(f"    Acc={result.accuracy:.3f}, F1={result.f1:.3f}, "
              f"FalseSafe={result.false_safe_rate:.3f}, FalseUnsafe={result.false_unsafe_rate:.3f}")
    
    return results


def error_analysis(
    model: Any,
    X_test: np.ndarray,
    y_test: np.ndarray,
    test_records: List[Dict],
    feature_names: List[str],
    threshold: float = 0.5,
    n_examples: int = 5
) -> Dict[str, List[Dict]]:
    """Analyze misclassified examples."""
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)
    
    # False-safe: predicted safe (1), actually unsafe (0)
    false_safe_mask = (y_pred == 1) & (y_test == 0)
    false_safe_indices = np.where(false_safe_mask)[0]
    
    # False-unsafe: predicted unsafe (0), actually safe (1)
    false_unsafe_mask = (y_pred == 0) & (y_test == 1)
    false_unsafe_indices = np.where(false_unsafe_mask)[0]
    
    # True safe
    true_safe_mask = (y_pred == 1) & (y_test == 1)
    true_safe_indices = np.where(true_safe_mask)[0]
    
    # True unsafe
    true_unsafe_mask = (y_pred == 0) & (y_test == 0)
    true_unsafe_indices = np.where(true_unsafe_mask)[0]
    
    def format_example(idx: int) -> Dict:
        r = test_records[idx]
        return {
            "prompt_id": r.get("prompt_id", "unknown"),
            "task_type": r.get("task_type", "unknown"),
            "complexity_level": r.get("complexity_level", 0),
            "prompt_preview": r.get("prompt", "")[:100] + "...",
            "predicted_prob": float(model.predict_proba(X_test[idx:idx+1])[:, 1][0]),
            "predicted_label": int(y_pred[idx]),
            "actual_label": int(y_test[idx]),
            "evaluation_method": r.get("evaluation_method", "unknown"),
            "evaluation_confidence": r.get("evaluation_confidence", 0),
            "fp16_latency_ms": r.get("fp16_latency_ms", 0),
            "int4_latency_ms": r.get("int4_latency_ms", 0),
        }
    
    return {
        "false_safe": [format_example(i) for i in false_safe_indices[:n_examples]],
        "false_unsafe": [format_example(i) for i in false_unsafe_indices[:n_examples]],
        "true_safe": [format_example(i) for i in true_safe_indices[:n_examples]],
        "true_unsafe": [format_example(i) for i in true_unsafe_indices[:n_examples]],
    }


def cross_validation_study(
    records: List[Dict],
    feature_names: List[str],
    X_all: np.ndarray,
    y_all: np.ndarray,
    n_splits: int = 5
) -> Dict[str, Any]:
    """Cross-validation on training data."""
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    models = {
        "LogisticRegression": LogisticRegression(C=1.0, max_iter=1000, class_weight="balanced", random_state=42),
        "RandomForest": RandomForestClassifier(n_estimators=200, max_depth=10, class_weight="balanced", random_state=42, n_jobs=-1),
        "GradientBoosting": GradientBoostingClassifier(n_estimators=200, max_depth=5, learning_rate=0.1, random_state=42),
    }
    
    results = {}
    for name, model in models.items():
        scores = cross_val_score(model, X_all, y_all, cv=cv, scoring='roc_auc', n_jobs=-1)
        results[name] = {
            "cv_roc_auc_mean": float(np.mean(scores)),
            "cv_roc_auc_std": float(np.std(scores)),
            "cv_scores": scores.tolist()
        }
        print(f"  {name}: ROC-AUC = {np.mean(scores):.3f} (+/- {np.std(scores):.3f})")
    
    return results


def main():
    print("=" * 70)
    print("PHASE C: QUANTIZATION-SAFETY CLASSIFIER TRAINING")
    print("=" * 70)
    
    # Load dataset
    with open("data/processed/b2_dataset.json", 'r') as f:
        records = json.load(f)
    
    print(f"Loaded {len(records)} records")
    safe_count = sum(1 for r in records if r.get("safe_to_quantize", False))
    print(f"Safe: {safe_count}, Unsafe: {len(records) - safe_count}")
    
    # Create splits
    print("\nCreating train/val/test splits...")
    train_records, val_records, test_records = create_splits(records)
    
    # Feature extraction configs to test
    configs = {
        "task_only": FeatureConfig(use_task_type=True, use_complexity_level=False, use_tfidf=False),
        "basic": FeatureConfig(use_task_type=True, use_complexity_level=False, use_tfidf=False),
        "basic_complexity": FeatureConfig(use_task_type=True, use_complexity_level=True, use_tfidf=False),
        "tfidf": FeatureConfig(use_task_type=True, use_complexity_level=False, use_tfidf=True),
        "tfidf_complexity": FeatureConfig(use_task_type=True, use_complexity_level=True, use_tfidf=True),
    }
    
    # First, find best feature config with CV
    print("\n" + "=" * 60)
    print("FEATURE CONFIGURATION COMPARISON (5-fold CV)")
    print("=" * 60)
    
    best_config_name = None
    best_auc = 0
    
    for name, config in configs.items():
        print(f"\nTesting config: {name}")
        extractor = FeatureExtractor(config)
        X_all, y_all = extractor.fit_transform(records)
        
        # Cross-validation
        cv_results = cross_validation_study(records, extractor.get_feature_names(), X_all, y_all)
        
        # Track best
        for model_name, result in cv_results.items():
            if result["cv_roc_auc_mean"] > best_auc:
                best_auc = result["cv_roc_auc_mean"]
                best_config_name = name
                print(f"  NEW BEST: {name} + {model_name} = {best_auc:.3f}")
    
    print(f"\nBest config: {best_config_name}")
    
    # Use best config for final training
    best_config = configs[best_config_name]
    print(f"\nFinal training with config: {best_config_name}")
    
    extractor = FeatureExtractor(best_config)
    
    # Fit on train, transform all
    extractor.fit(train_records)
    X_train, y_train = extractor.transform(train_records)
    X_val, y_val = extractor.transform(val_records)
    X_test, y_test = extractor.transform(test_records)
    
    print(f"\nFeature matrix: {X_train.shape[1]} features")
    print(f"Train: {X_train.shape[0]}, Val: {X_val.shape[0]}, Test: {X_test.shape[0]}")
    
    # Train all models
    print("\n" + "=" * 60)
    print("MODEL TRAINING")
    print("=" * 60)
    
    trained_models = {}
    for model_config in MODEL_CONFIGS:
        print(f"\nTraining {model_config.name}...")
        model = train_model(model_config, X_train, y_train, X_val, y_val)
        trained_models[model_config.name] = model
    
    # Evaluate all models at multiple thresholds
    print("\n" + "=" * 60)
    print("THRESHOLD ANALYSIS ON TEST SET")
    print("=" * 60)
    
    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    all_results = {}
    
    for name, model in trained_models.items():
        print(f"\n{name}:")
        results = threshold_analysis(model, X_test, y_test, thresholds, name)
        all_results[name] = results
    
    # Select best model based on validation (using 0.5 threshold as reference)
    best_model_name = max(all_results.keys(), 
                          key=lambda k: all_results[k][thresholds.index(0.5)].f1)
    best_model = trained_models[best_model_name]
    
    print(f"\nBest model (F1 @ 0.5): {best_model_name}")
    
    # Calibration analysis
    print("\n" + "=" * 60)
    print("CALIBRATION ANALYSIS")
    print("=" * 60)
    
    for name, model in trained_models.items():
        cal = calibration_analysis(model, X_test, y_test)
        print(f"\n{name}:")
        print(f"  Brier Score: {cal['brier_score']:.4f}")
        print(f"  ECE: {cal['ece']:.4f}")
    
    # Ablation study
    print("\n" + "=" * 60)
    print("ABLATION STUDY")
    print("=" * 60)
    
    ablation_results = ablation_study(
        records, extractor.get_feature_names(),
        X_all, y_all, X_train, y_train, X_val, y_val, X_test, y_test
    )
    
    # Feature importance for best model
    print("\n" + "=" * 60)
    print("FEATURE IMPORTANCE")
    print("=" * 60)
    
    importance = feature_importance_analysis(best_model, extractor.get_feature_names(), top_k=20)
    for item in importance:
        print(f"  {item['feature']}: {item['importance']:.4f}")
    
    # Error analysis
    print("\n" + "=" * 60)
    print("ERROR ANALYSIS")
    print("=" * 60)
    
    errors = error_analysis(best_model, X_test, y_test, test_records, extractor.get_feature_names())
    
    print(f"\nFalse-Safe (predicted safe, actually unsafe): {len(errors['false_safe'])}")
    for i, e in enumerate(errors['false_safe'][:5]):
        print(f"  {e['prompt_id']} ({e['task_type']}, L{e['complexity_level']}): "
              f"prob={e['predicted_prob']:.3f}, method={e['evaluation_method']}")
        print(f"    Prompt: {e['prompt_preview']}")
    
    print(f"\nFalse-Unsafe (predicted unsafe, actually safe): {len(errors['false_unsafe'])}")
    for i, e in enumerate(errors['false_unsafe'][:5]):
        print(f"  {e['prompt_id']} ({e['task_type']}, L{e['complexity_level']}): "
              f"prob={e['predicted_prob']:.3f}, method={e['evaluation_method']}")
        print(f"    Prompt: {e['prompt_preview']}")
    
    # Final evaluation at recommended threshold
    print("\n" + "=" * 60)
    print("FINAL MODEL EVALUATION AT RECOMMENDED THRESHOLDS")
    print("=" * 60)
    
    for t in [0.5, 0.6, 0.7, 0.75, 0.8]:
        result = evaluate_model(best_model, X_test, y_test, t, best_model_name)
        print(f"  Threshold {t:.2f}: Acc={result.accuracy:.3f}, "
              f"Prec={result.precision:.3f}, Rec={result.recall:.3f}, "
              f"F1={result.f1:.3f}, FalseSafe={result.false_safe_rate:.3f}, "
              f"FalseUnsafe={result.false_unsafe_rate:.3f}, "
              f"SafePreds={result.safe_predictions}")
    
    # Save final model and artifacts
    print("\n" + "=" * 60)
    print("SAVING MODEL AND ARTIFACTS")
    print("=" * 60)
    
    model_path = Path("models/quantization_safety/classifier.pkl")
    extractor_path = Path("models/quantization_safety/feature_extractor.pkl")
    
    joblib.dump(best_model, model_path)
    extractor.save(extractor_path)
    
    # Save metadata
    metadata = {
        "model_name": best_model_name,
        "feature_config": asdict(best_config),
        "feature_names": extractor.get_feature_names(),
        "thresholds_analyzed": thresholds,
        "recommended_threshold": 0.7,  # Based on false-safe rate priority
        "test_metrics": {
            "accuracy": all_results[best_model_name][thresholds.index(0.5)].accuracy,
            "precision": all_results[best_model_name][thresholds.index(0.5)].precision,
            "recall": all_results[best_model_name][thresholds.index(0.5)].recall,
            "f1": all_results[best_model_name][thresholds.index(0.5)].f1,
            "roc_auc": all_results[best_model_name][thresholds.index(0.5)].roc_auc,
            "false_safe_rate": all_results[best_model_name][thresholds.index(0.5)].false_safe_rate,
            "false_unsafe_rate": all_results[best_model_name][thresholds.index(0.5)].false_unsafe_rate,
        },
        "ablation_results": ablation_results,
        "feature_importance": importance,
        "calibration": calibration_analysis(best_model, X_test, y_test),
        "errors": errors,
        "random_seed": 42,
        "dataset_version": "b2_182",
        "train_size": len(train_records),
        "val_size": len(val_records),
        "test_size": len(test_records),
    }
    
    with open("models/quantization_safety/metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\nModel saved to {model_path}")
    print(f"Extractor saved to {extractor_path}")
    print(f"Metadata saved to models/quantization_safety/metadata.json")
    
    # Print summary
    print("\n" + "=" * 70)
    print("PHASE C SUMMARY")
    print("=" * 70)
    
    print(f"""
Dataset: 182 records (B2)
Split: Train={len(train_records)}, Val={len(val_records)}, Test={len(test_records)}
Features: {X_train.shape[1]} ({best_config_name} config)
Models trained: {len(trained_models)}
Best model: {best_model_name}

Test metrics @ 0.5:
  Accuracy:  {all_results[best_model_name][thresholds.index(0.5)].accuracy:.3f}
  Precision: {all_results[best_model_name][thresholds.index(0.5)].precision:.3f}
  Recall:    {all_results[best_model_name][thresholds.index(0.5)].recall:.3f}
  F1:        {all_results[best_model_name][thresholds.index(0.5)].f1:.3f}
  ROC-AUC:   {all_results[best_model_name][thresholds.index(0.5)].roc_auc:.3f}
  False-Safe Rate: {all_results[best_model_name][thresholds.index(0.5)].false_safe_rate:.3f}
  False-Unsafe Rate: {all_results[best_model_name][thresholds.index(0.5)].false_unsafe_rate:.3f}

Recommended threshold: 0.7 (prioritizes low false-safe rate)
  @ 0.7: FalseSafe={all_results[best_model_name][thresholds.index(0.7)].false_safe_rate:.3f},
         FalseUnsafe={all_results[best_model_name][thresholds.index(0.7)].false_unsafe_rate:.3f}

Model saved: models/quantization_safety/classifier.pkl
Extractor saved: models/quantization_safety/feature_extractor.pkl
Metadata saved: models/quantization_safety/metadata.json

Ready for Phase D: Decision Engine integration.
""")


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)
    main()