#!/usr/bin/env python3
"""
Phase F.1: Offline Request-Level Energy Predictor Training

Predicts: energy_difference = INT4_energy_Wh - FP16_energy_Wh

Uses ONLY B2 dataset (182 samples with measured FP16/INT4 energy).
Locked 20% test set (36 samples, stratified by task_type + complexity_level, seed=42).
Training on 146 samples with nested 5x5 CV for ElasticNet hyperparameter selection.

No Phase F data used. No TF-IDF. No safety probability. No classifier features.
"""

import sys
import json
import re
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from collections import defaultdict

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from sklearn.model_selection import (
    train_test_split, KFold, cross_val_score, GridSearchCV
)
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import ElasticNet
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    accuracy_score, confusion_matrix
)
from sklearn.base import clone

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Constants
RANDOM_SEED = 42
TEST_SIZE = 0.2
N_OUTER_FOLDS = 5
N_INNER_FOLDS = 5
DATA_PATH = Path("data/processed/b2_dataset.json")
OUTPUT_DIR = Path("models/energy_predictor")
RESULTS_DIR = Path("data/evaluation/phase_f")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Static baseline from Phase 0.5
STATIC_FP16_ENERGY = 0.0488  # Wh
STATIC_INT4_ENERGY = 0.0493  # Wh
STATIC_DIFF = STATIC_INT4_ENERGY - STATIC_FP16_ENERGY  # 0.0005 Wh

# Features available at decision time (from B2 features) - only those present in dataset
NUMERIC_FEATURES = [
    "char_length", "word_count", "token_count_estimate",
    "has_question_mark", "has_code_indicator", "question_count", "sentence_count",
    "has_scientific_terms", "scientific_term_count", "has_reasoning_indicators",
    "reasoning_indicator_count", "has_code_keywords", "code_keyword_count",
    "avg_word_length", "unique_word_ratio", "has_numbers", "number_count",
    "requested_length_hint", "max_tokens_requested",
]

TASK_TYPES = [
    "factual", "extraction", "explanation", "reasoning",
    "coding", "summarization", "scientific", "creative",
]


def load_b2_data() -> pd.DataFrame:
    """Load B2 dataset and extract features + targets."""
    with open(DATA_PATH) as f:
        data = json.load(f)

    records = []
    for r in data:
        if not r.get("measurement_status") == "complete":
            continue
        
        # Extract numeric features from B2 features dict
        feat = r.get("features", {})
        row = {"prompt_id": r["prompt_id"]}
        
        # Numeric features - handle special cases
        for feat_name in NUMERIC_FEATURES:
            val = feat.get(feat_name, 0.0)
            
            # Handle special string features
            if feat_name == "requested_length_hint":
                # Convert string hints to numeric
                if val is None or val == "":
                    val = 0.0
                elif isinstance(val, str):
                    val = val.lower()
                    if val in ["short", "brief", "concise"]:
                        val = 1.0
                    elif val in ["detailed", "comprehensive"]:
                        val = 2.0
                    elif "sentence" in val:
                        # Extract number: "5 sentences" -> 5 * 20 = 100 tokens approx
                        import re
                        m = re.search(r'(\d+)', val)
                        if m:
                            val = float(m.group(1)) * 20.0
                        else:
                            val = 20.0
                    elif val is True:
                        val = 1.0
                    elif val is False:
                        val = 0.0
                elif feat_name == "max_tokens_requested":
                    if val is None or (isinstance(val, float) and np.isnan(val)):
                        val = 0.0
                
            # DEBUG
            print(f"DEBUG: feat_name={feat_name}, val={val}, type={type(val).__name__}")
            print(f"  About to check feat_name == 'max_tokens_requested': {feat_name == 'max_tokens_requested'}")
            if feat_name == "max_tokens_requested":
                print(f"  Inside max_tokens_requested block, val={val}, is None: {val is None}")
                if val is None or (isinstance(val, float) and np.isnan(val)):
                    val = 0.0
                    print(f"  Set val to 0.0")
            row[feat_name] = float(val)
            print(f"  After assignment: row['{feat_name}']={row.get(feat_name)}")
            
        # Task type
        row["task_type"] = r.get("task_type", "unknown")
        
        # Complexity level
        row["complexity_level"] = r.get("complexity_level", 1)
        
        # Targets
        row["fp16_energy"] = r.get("fp16_energy_wh", 0.0)
        row["int4_energy"] = r.get("int4_energy_wh", 0.0)
        row["energy_diff"] = row["int4_energy"] - row["fp16_energy"]
        
        records.append(row)

    df = pd.DataFrame(records)
    print(f"Loaded {len(df)} samples from B2 dataset")
    return df


def create_stratified_split(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, List[str], List[str]]:
    """
    Create stratified 80/20 split by (task_type, complexity_level).
    Returns train_df, test_df, train_ids, test_ids.
    """
    # Create stratification key
    df["strata"] = df["task_type"].astype(str) + "_" + df["complexity_level"].astype(str)
    
    # Check strata sizes
    strata_counts = df["strata"].value_counts()
    print(f"\nStrata distribution:")
    for s, c in strata_counts.items():
        print(f"  {s}: {c}")
    
    # Split
    train_df, test_df = train_test_split(
        df,
        test_size=TEST_SIZE,
        stratify=df["strata"],
        random_state=RANDOM_SEED
    )
    
    train_ids = train_df["prompt_id"].tolist()
    test_ids = test_df["prompt_id"].tolist()
    
    print(f"\nSplit: {len(train_df)} train, {len(test_df)} test")
    print(f"Train IDs: {train_ids}")
    print(f"Test IDs: {test_ids}")
    
    return train_df, test_df, train_ids, test_ids


def build_preprocessor() -> ColumnTransformer:
    """Build preprocessing pipeline for features."""
    # Numeric features: standardize
    numeric_transformer = Pipeline([
        ("scaler", StandardScaler())
    ])
    
    # Task type: one-hot encode
    categorical_transformer = Pipeline([
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer([
        ("num", numeric_transformer, NUMERIC_FEATURES),
        ("cat", categorical_transformer, ["task_type"]),
    ], remainder="drop")
    
    return preprocessor


def create_model_pipeline(alpha: float, l1_ratio: float) -> Pipeline:
    """Create ElasticNet pipeline with preprocessing."""
    return Pipeline([
        ("preprocessor", build_preprocessor()),
        ("regressor", ElasticNet(
            alpha=alpha,
            l1_ratio=l1_ratio,
            max_iter=10000,
            random_state=RANDOM_SEED,
            selection="cyclic"
        ))
    ])


def get_hyperparameter_grid() -> Dict[str, List]:
    """Return hyperparameter grid for ElasticNet."""
    return {
        "regressor__alpha": np.logspace(-4, 1, 20),  # 0.0001 to 10
        "regressor__l1_ratio": [0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0],  # Ridge to Lasso
    }


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute all evaluation metrics."""
    # Directional accuracy: sign of prediction matches sign of truth
    directional_acc = np.mean(np.sign(y_pred) == np.sign(y_true))
    
    # Within-10% accuracy
    rel_error = np.abs(y_pred - y_true) / (np.abs(y_true) + 1e-10)
    within_10 = np.mean(rel_error <= 0.10)
    
    # Confusion matrix for lower-energy configuration
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
        "sign_agreement": float(np.mean(np.sign(y_true) == np.sign(y_pred))),
    }


def static_baseline_predictions(y_true: np.ndarray) -> np.ndarray:
    """Static baseline always predicts the Phase 0.5 difference."""
    return np.full_like(y_true, STATIC_DIFF)


def run_nested_cv(train_df: pd.DataFrame) -> Tuple[Dict, Pipeline]:
    """
    Run nested 5x5 CV to select best hyperparameters.
    Returns best_params and best_pipeline fitted on full training data.
    """
    X = train_df[NUMERIC_FEATURES + ["task_type"]]
    y = train_df["energy_diff"]
    
    # Inner CV for hyperparameter selection
    inner_cv = KFold(n_splits=N_INNER_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    outer_cv = KFold(n_splits=N_OUTER_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    
    pipeline = Pipeline([
        ("preprocessor", build_preprocessor()),
        ("regressor", ElasticNet(
            max_iter=10000,
            random_state=RANDOM_SEED,
            selection="cyclic"
        ))
    ])
    
    param_grid = get_hyperparameter_grid()
    
    # Grid search with inner CV
    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=inner_cv,
        scoring="neg_mean_absolute_error",
        n_jobs=-1,
        verbose=0
    )
    
    # Nested CV: evaluate best params with outer CV
    outer_scores = cross_val_score(
        grid_search, X, y, cv=outer_cv,
        scoring="neg_mean_absolute_error", n_jobs=-1
    )
    
    print(f"\nNested CV MAE: {-outer_scores.mean():.6f} (+/- {outer_scores.std()*2:.6f})")
    
    # Fit on full training data with best params
    grid_search.fit(X, y)
    best_pipeline = grid_search.best_estimator_
    best_params = grid_search.best_params_
    
    print(f"Best params: {best_params}")
    
    return best_params, best_pipeline


def compute_baseline_metrics(y_true: np.ndarray) -> Dict[str, float]:
    """Compute metrics for static baseline."""
    y_pred = static_baseline_predictions(y_true)
    return evaluate_predictions(y_true, y_pred)


def main():
    print("=" * 70)
    print("PHASE F.1: ENERGY PREDICTOR TRAINING")
    print("=" * 70)
    
    # 1. Load data
    print("\n[1/8] Loading B2 dataset...")
    df = load_b2_data()
    print(f"Total samples: {len(df)}")
    
    # 2. Create locked test split
    print("\n[2/8] Creating locked test split (20%, stratified)...")
    train_df, test_df, train_ids, test_ids = create_stratified_split(df)
    
    # Save locked test split
    split_info = {
        "test_ids": test_ids,
        "train_ids": train_ids,
        "test_size": TEST_SIZE,
        "random_seed": RANDOM_SEED,
        "stratification": "task_type + complexity_level"
    }
    with open(RESULTS_DIR / "energy_predictor_test_split.json", "w") as f:
        json.dump(split_info, f, indent=2)
    print(f"Saved test split to {RESULTS_DIR / 'energy_predictor_test_split.json'}")
    
    # 3. Prepare features and targets
    X_train = train_df[NUMERIC_FEATURES + ["task_type"]]
    y_train = train_df["energy_diff"]
    X_test = test_df[NUMERIC_FEATURES + ["task_type"]]
    y_test = test_df["energy_diff"]
    
    # 4. Nested CV for hyperparameter selection
    print("\n[3/8] Running nested 5x5 CV for hyperparameter selection...")
    best_params, best_pipeline = run_nested_cv(train_df)
    
    # 5. Train final model on full training data
    print("\n[4/8] Training final model on full training set...")
    final_pipeline = create_model_pipeline(
        alpha=best_params["regressor__alpha"],
        l1_ratio=best_params["regressor__l1_ratio"]
    )
    final_pipeline.fit(
        train_df[NUMERIC_FEATURES + ["task_type"]],
        y_train
    )
    
    # 5. Evaluate on locked test set
    print("\n[5/8] Evaluating on locked test set...")
    y_pred = final_pipeline.predict(X_test)
    test_metrics = evaluate_predictions(y_test.values, y_pred)
    
    # 6. Static baseline on test set
    print("\n[6/8] Computing static baseline metrics...")
    baseline_metrics = compute_baseline_metrics(y_test.values)
    
    # 8. Training set metrics (for comparison)
    y_train_pred = final_pipeline.predict(X_train)
    train_metrics = evaluate_predictions(y_train.values, y_train_pred)
    
    # 9. Cross-validation on training set (outer CV scores)
    print("\n[7/8] Computing cross-validation scores on training set...")
    cv_pipeline = create_model_pipeline(
        alpha=best_params["regressor__alpha"],
        l1_ratio=best_params["regressor__l1_ratio"]
    )
    outer_cv = KFold(n_splits=N_OUTER_FOLDS, shuffle=True, random_state=RANDOM_SEED)
    cv_scores = cross_val_score(
        cv_pipeline,
        train_df[NUMERIC_FEATURES + ["task_type"]],
        y_train,
        cv=outer_cv,
        scoring="neg_mean_absolute_error",
        n_jobs=-1
    )
    cv_mae = -cv_scores.mean()
    cv_mae_std = cv_scores.std()
    
    # 8. Confusion matrix and directional analysis
    y_test_sign = np.sign(y_test.values)
    y_pred_sign = np.sign(y_pred)
    cm = confusion_matrix(y_test_sign, y_pred_sign, labels=[-1, 1])
    tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)
    
    # Save results
    print("\n[8/8] Saving results...")
    
    # Save model
    import joblib
    model_path = OUTPUT_DIR / "energy_predictor.joblib"
    joblib.dump(final_pipeline, model_path)
    print(f"Saved model to {model_path}")
    
    # Save predictions
    predictions = []
    for i, (_, row) in enumerate(test_df.iterrows()):
        predictions.append({
            "prompt_id": row["prompt_id"],
            "task_type": row["task_type"],
            "complexity_level": row["complexity_level"],
            "b2_safe_to_quantize": row.get("b2_safe_to_quantize", False),
            "true_energy_diff": float(y_test.iloc[i]),
            "predicted_energy_diff": float(y_pred[i]),
            "true_fp16_energy": float(row["fp16_energy"]),
            "true_int4_energy": float(row["int4_energy"]),
            "true_sign": int(np.sign(y_test.iloc[i])),
            "pred_sign": int(np.sign(y_pred[i])),
            "correct_direction": bool(np.sign(y_test.iloc[i]) == np.sign(y_pred[i])),
        })
    
    # Baseline predictions on test set
    baseline_pred = static_baseline_predictions(y_test.values)
    baseline_correct = np.mean(np.sign(baseline_pred) == np.sign(y_test.values))
    
    # Prepare results
    results = {
        "metadata": {
            "phase": "F.1",
            "timestamp": pd.Timestamp.now().isoformat(),
            "n_total": len(df),
            "n_train": len(train_df),
            "n_test": len(test_df),
            "test_ids": test_ids,
            "train_ids": train_ids,
            "random_seed": RANDOM_SEED,
            "stratification": "task_type + complexity_level",
            "model": "ElasticNet",
            "best_params": best_params,
        },
        "cv_performance": {
            "mean_mae": float(cv_mae),
            "std_mae": float(cv_mae_std),
            "folds": [-float(s) for s in cv_scores],
        },
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "baseline_metrics": baseline_metrics,
        "improvement_over_baseline": {
            "mae_reduction": baseline_metrics["mae"] - test_metrics["mae"],
            "rmse_reduction": baseline_metrics["rmse"] - test_metrics["rmse"],
            "r2_improvement": test_metrics["r2"] - baseline_metrics["r2"],
            "directional_acc_improvement": test_metrics["directional_accuracy"] - baseline_metrics["directional_accuracy"],
        },
        "success_criterion": {
            "mae_threshold": 0.002,
            "directional_accuracy_threshold": 0.70,
            "mae_met": test_metrics["mae"] < 0.002,
            "directional_accuracy_met": test_metrics["directional_accuracy"] > 0.70,
            "overall_success": (test_metrics["mae"] < 0.002) and (test_metrics["directional_accuracy"] > 0.70),
        },
        "test_predictions": predictions,
        "confusion_matrix": {
            "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)
        },
        "baseline_confusion": {
            "directional_accuracy": float(baseline_correct),
        },
        "feature_importance": {},
    }
    
    # Extract feature importance (coefficients)
    if hasattr(final_pipeline.named_steps["regressor"], "coef_"):
        coef = final_pipeline.named_steps["regressor"].coef_
        # Get feature names from preprocessor
        preprocessor = final_pipeline.named_steps["preprocessor"]
        cat_encoder = preprocessor.named_transformers_["cat"].named_steps["encoder"]
        cat_features = cat_encoder.get_feature_names_out(["task_type"]).tolist()
        num_features = NUMERIC_FEATURES
        all_features = num_features + cat_features
        if len(coef) == len(all_features):
            importance = dict(zip(all_features, coef.tolist()))
            results["feature_importance"] = {k: float(v) for k, v in sorted(importance.items(), key=lambda x: abs(x[1]), reverse=True)}
    
    # Save results
    results_path = RESULTS_DIR / "energy_predictor_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Saved results to {results_path}")
    
    # Print summary
    print("\n" + "=" * 70)
    print("ENERGY PREDICTOR TRAINING COMPLETE")
    print("=" * 70)
    print(f"\nCV Performance (5-fold): MAE = {cv_mae:.6f} +/- {cv_mae_std:.6f} Wh")
    print(f"\nTest Set Performance (n={len(test_df)}):")
    for k, v in test_metrics.items():
        print(f"  {k}: {v:.6f}" if isinstance(v, float) else f"  {k}: {v}")
    print(f"\nStatic Baseline (Phase 0.5):")
    for k, v in baseline_metrics.items():
        print(f"  {k}: {v:.6f}" if isinstance(v, float) else f"  {k}: {v}")
    print(f"\nImprovement over Baseline:")
    for k, v in results["improvement_over_baseline"].items():
        print(f"  {k}: {v:.6f}" if isinstance(v, float) else f"  {k}: {v}")
    print(f"\nSuccess Criterion:")
    for k, v in results["success_criterion"].items():
        print(f"  {k}: {v}")
    
    print(f"\nResults saved to {results_path}")
    print(f"Model saved to {model_path}")
    print(f"Test split saved to {RESULTS_DIR / 'energy_predictor_test_split.json'}")
    
    return results


if __name__ == "__main__":
    main()