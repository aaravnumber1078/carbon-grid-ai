#!/usr/bin/env python3
"""
Phase G Feasibility Assessment on 25-Prompt Pilot

Aggregates 3 repetitions per prompt/config, uses 21 pre-inference features,
evaluates FP16 and INT4 energy prediction separately with LOPO-CV.
Compares Ridge and ElasticNet against fold-specific training-mean baseline.
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple, Any
from collections import defaultdict
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

from sklearn.linear_model import Ridge, ElasticNet, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneOut
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ============================================================
# CONFIGURATION
# ============================================================

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# 21 approved pre-inference features (must match Phase G design)
FEATURE_NAMES = [
    "char_length", "word_count", "token_count_estimate",
    "sentence_count", "question_count", "has_question_mark",
    "has_code_indicator", "has_scientific_terms", "scientific_term_count",
    "has_reasoning_indicators", "reasoning_indicator_count",
    "has_code_keywords", "code_keyword_count",
    "avg_word_length", "unique_word_ratio",
    "has_numbers", "number_count",
    "requested_length_hint", "max_tokens_requested",
    "instruction_count", "constraint_count",
]

# Forbidden: category, task_type, prompt_id, output_tokens, latency, power, energy, response, etc.
# Only FEATURE_NAMES above are allowed as predictors.

PILOT_PROMPTS_PATH = Path("C:/CARBON GRID AI/data/evaluation/phase_g/pilot_prompts.json")
PILOT_RESULTS_PATH = Path("C:/CARBON GRID AI/data/evaluation/phase_g/pilot_results.json")
OUTPUT_DIR = Path("C:/CARBON GRID AI/data/evaluation/phase_g")
REPORT_PATH = OUTPUT_DIR / "phase_g_feasibility_report.json"
SCRIPT_PATH = Path("C:/CARBON GRID AI/scripts/phase_g_feasibility.py")


# ============================================================
# DATA LOADING & AGGREGATION
# ============================================================

def load_pilot_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load pilot prompts and results, aggregate to per-prompt per-config means."""
    
    # Load prompts with features
    with open(PILOT_PROMPTS_PATH) as f:
        pilot_data = json.load(f)
    prompts = pilot_data["pilot_prompts"]
    
    prompt_df = pd.DataFrame(prompts)
    print(f"Loaded {len(prompt_df)} pilot prompts")
    
    # Load raw runs
    with open(PILOT_RESULTS_PATH) as f:
        results_data = json.load(f)
    runs = results_data["runs"]
    
    runs_df = pd.DataFrame(runs)
    successful = runs_df[runs_df["success"] == True].copy()
    print(f"Successful runs: {len(successful)} / {len(runs_df)}")
    
    # Aggregate: mean energy per prompt per configuration
    agg = successful.groupby(["prompt_id", "configuration"]).agg(
        energy_wh=("energy_wh", "mean"),
        energy_std=("energy_wh", "std"),
        latency_ms=("latency_ms", "mean"),
        latency_std=("latency_ms", "std"),
        avg_power_w=("avg_power_w", "mean"),
        n_reps=("energy_wh", "count"),
    ).reset_index()
    
    print(f"Aggregated prompt-config pairs: {len(agg)}")
    print(f"  FP16: {len(agg[agg['configuration'] == 'fp16'])}")
    print(f"  INT4: {len(agg[agg['configuration'] == 'int4'])}")
    
    # Merge features with targets
    merged = agg.merge(prompt_df, on="prompt_id", how="left")
    
    # Verify all features present
    missing = [f for f in FEATURE_NAMES if f not in merged.columns]
    if missing:
        raise ValueError(f"Missing features: {missing}")
    
    # Check for missing values
    missing_vals = merged[FEATURE_NAMES].isnull().sum()
    if missing_vals.any():
        print(f"WARNING: Missing values in features:\n{missing_vals[missing_vals > 0]}")
    
    # Check for constant features
    constant = [f for f in FEATURE_NAMES if merged[f].nunique() == 1]
    if constant:
        print(f"WARNING: Constant features (will be dropped by scaler): {constant}")
    
    return merged, prompt_df


def prepare_config_data(merged: pd.DataFrame, config: str) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """Extract X, y for a specific configuration."""
    config_data = merged[merged["configuration"] == config].copy()
    config_data = config_data.sort_values("prompt_id").reset_index(drop=True)
    
    X = config_data[FEATURE_NAMES].values.astype(float)
    y = config_data["energy_wh"].values.astype(float)
    prompt_ids = config_data["prompt_id"].tolist()
    
    print(f"{config.upper()}: {len(y)} samples, energy range [{y.min():.6f}, {y.max():.6f}], mean={y.mean():.6f}")
    return X, y, prompt_ids


# ============================================================
# MODEL DEFINITIONS
# ============================================================

def get_models():
    """Return dict of model name -> (pipeline, param_grid for inner CV)."""
    
    models = {
        "Ridge": (
            Pipeline([("scaler", StandardScaler()), ("model", Ridge(random_state=RANDOM_SEED))]),
            {"model__alpha": np.logspace(-3, 3, 13)}
        ),
        "ElasticNet": (
            Pipeline([("scaler", StandardScaler()), ("model", ElasticNet(random_state=RANDOM_SEED, max_iter=10000))]),
            {"model__alpha": np.logspace(-3, 1, 9), "model__l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9]}
        ),
    }
    return models


# ============================================================
# LOPO-CV EVALUATION
# ============================================================

def run_lopo_cv(X: np.ndarray, y: np.ndarray, prompt_ids: List[str], config: str) -> Dict[str, Any]:
    """Run Leave-One-Prompt-Out CV for a configuration."""
    
    loo = LeaveOneOut()
    n_samples = len(y)
    
    models = get_models()
    
    # Storage for predictions
    results = {
        "config": config,
        "n_samples": n_samples,
        "models": {},
        "baseline_mae": [],
        "baseline_rmse": [],
        "baseline_r2": [],
    }
    
    # Per-sample predictions for each model
    pred_storage = {name: np.zeros(n_samples) for name in models}
    pred_storage["baseline"] = np.zeros(n_samples)
    
    fold = 0
    for train_idx, test_idx in loo.split(X):
        fold += 1
        if fold % 5 == 0:
            print(f"  {config} fold {fold}/{n_samples}")
        
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Fold-specific training-mean baseline
        train_mean = np.mean(y_train)
        pred_storage["baseline"][test_idx] = train_mean
        
        # For each model
        for name, (pipeline, param_grid) in models.items():
            # Inner CV for hyperparameter selection (nested)
            from sklearn.model_selection import GridSearchCV
            
            inner_cv = LeaveOneOut()  # or KFold(n_splits=min(5, len(train_idx)))
            if len(train_idx) >= 5:
                inner_cv = LeaveOneOut()  # LOPO again for inner
            else:
                inner_cv = 2
            
            grid_search = GridSearchCV(
                pipeline, param_grid,
                cv=inner_cv,
                scoring="neg_mean_absolute_error",
                n_jobs=-1,
                verbose=0
            )
            
            try:
                grid_search.fit(X_train, y_train)
                y_pred = grid_search.predict(X_test)
                pred_storage[name][test_idx] = y_pred
            except Exception as e:
                print(f"    WARNING: {name} failed on fold {fold}: {e}")
                pred_storage[name][test_idx] = train_mean  # fallback to baseline
    
    # Compute metrics for each model
    for name in list(models.keys()) + ["baseline"]:
        y_pred = pred_storage[name]
        mae = mean_absolute_error(y, y_pred)
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        r2 = r2_score(y, y_pred)
        
        results["models"][name] = {
            "mae": float(mae),
            "rmse": float(rmse),
            "r2": float(r2),
            "predictions": y_pred.tolist(),
            "errors": (y - y_pred).tolist(),
        }
    
    # Baseline metrics (aggregate)
    results["baseline_mae"] = results["models"]["baseline"]["mae"]
    results["baseline_rmse"] = results["models"]["baseline"]["rmse"]
    results["baseline_r2"] = results["models"]["baseline"]["r2"]
    
    return results, pred_storage


def run_simple_linear_regression(X: np.ndarray, y: np.ndarray, prompt_ids: List[str], config: str) -> Dict[str, Any]:
    """Run LOPO-CV with simple LinearRegression (no regularization)."""
    
    loo = LeaveOneOut()
    n_samples = len(y)
    y_pred_lr = np.zeros(n_samples)
    
    fold = 0
    for train_idx, test_idx in loo.split(X):
        fold += 1
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Standardize within fold
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        try:
            lr = LinearRegression()
            lr.fit(X_train_scaled, y_train)
            y_pred_lr[test_idx] = lr.predict(X_test_scaled)
        except Exception as e:
            print(f"    WARNING: LinearRegression failed on fold {fold}: {e}")
            y_pred_lr[test_idx] = np.mean(y_train)
    
    mae = mean_absolute_error(y, y_pred_lr)
    rmse = np.sqrt(mean_squared_error(y, y_pred_lr))
    r2 = r2_score(y, y_pred_lr)
    
    return {
        "config": config,
        "model": "LinearRegression",
        "mae": float(mae),
        "rmse": float(rmse),
        "r2": float(r2),
        "predictions": y_pred_lr.tolist(),
        "errors": (y - y_pred_lr).tolist(),
    }


# ============================================================
# ANALYSIS HELPERS
# ============================================================

def analyze_feature_importance(X: np.ndarray, y: np.ndarray, feature_names: List[str]) -> Dict[str, float]:
    """Fit Ridge on full data to get feature importance (for inspection only)."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    ridge = Ridge(alpha=1.0, random_state=RANDOM_SEED)
    ridge.fit(X_scaled, y)
    importance = dict(zip(feature_names, ridge.coef_))
    return {k: float(v) for k, v in sorted(importance.items(), key=lambda x: abs(x[1]), reverse=True)}


def analyze_influential_prompts(results: Dict, prompt_ids: List[str], y: np.ndarray) -> List[Dict]:
    """Identify prompts with large errors that may drive results."""
    model_errors = {}
    for name, m in results["models"].items():
        model_errors[name] = np.array(m["errors"])
    
    influential = []
    for i, pid in enumerate(prompt_ids):
        # Check if this prompt has large absolute error across models
        max_abs_error = max(np.abs(model_errors[name][i]) for name in model_errors if name != "baseline")
        if max_abs_error > 2 * np.std(y):
            influential.append({
                "prompt_id": pid,
                "true_energy": float(y[i]),
                "max_abs_error": float(max_abs_error),
                "errors": {name: float(model_errors[name][i]) for name in model_errors}
            })
    return influential


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 70)
    print("PHASE G FEASIBILITY ASSESSMENT - 25 PROMPT PILOT")
    print("=" * 70)
    
    # Load and aggregate data
    merged, prompt_df = load_pilot_data()
    
    # Prepare data for each config
    X_fp16, y_fp16, ids_fp16 = prepare_config_data(merged, "fp16")
    X_int4, y_int4, ids_int4 = prepare_config_data(merged, "int4")
    
    # Verify same prompts in both configs
    assert ids_fp16 == ids_int4, "Prompt ID mismatch between configs"
    prompt_ids = ids_fp16
    
    # Feature analysis
    print("\n--- Feature Analysis ---")
    fp16_importance = analyze_feature_importance(X_fp16, y_fp16, FEATURE_NAMES)
    int4_importance = analyze_feature_importance(X_int4, y_int4, FEATURE_NAMES)
    
    print("Top 5 features for FP16:")
    for f, v in list(fp16_importance.items())[:5]:
        print(f"  {f}: {v:.6f}")
    print("Top 5 features for INT4:")
    for f, v in list(int4_importance.items())[:5]:
        print(f"  {f}: {v:.6f}")
    
    # LOPO-CV for FP16
    print("\n--- FP16 LOPO-CV ---")
    fp16_results, fp16_preds = run_lopo_cv(X_fp16, y_fp16, prompt_ids, "fp16")
    
    # LOPO-CV for INT4
    print("\n--- INT4 LOPO-CV ---")
    int4_results, int4_preds = run_lopo_cv(X_int4, y_int4, prompt_ids, "int4")
    
    # Linear Regression (additional check)
    print("\n--- Linear Regression (LOPO-CV) ---")
    fp16_lr = run_simple_linear_regression(X_fp16, y_fp16, prompt_ids, "fp16")
    int4_lr = run_simple_linear_regression(X_int4, y_int4, prompt_ids, "int4")
    
    # Influential prompts analysis
    fp16_influential = analyze_influential_prompts(fp16_results, prompt_ids, y_fp16)
    int4_influential = analyze_influential_prompts(int4_results, prompt_ids, y_int4)
    
    # Compile final report
    report = {
        "metadata": {
            "phase": "G.1.2_feasibility",
            "timestamp": pd.Timestamp.now().isoformat(),
            "pilot_prompts": 25,
            "repetitions_aggregated": 3,
            "configs_evaluated": ["fp16", "int4"],
            "cv_method": "LeaveOneOut (LOPO-CV)",
            "features_used": FEATURE_NAMES,
            "n_features": len(FEATURE_NAMES),
            "n_samples_per_config": 25,
            "random_seed": RANDOM_SEED,
        },
        "fp16": fp16_results,
        "int4": int4_results,
        "linear_regression": {
            "fp16": fp16_lr,
            "int4": int4_lr,
        },
        "feature_importance": {
            "fp16": fp16_importance,
            "int4": int4_importance,
        },
        "influential_prompts": {
            "fp16": fp16_influential,
            "int4": int4_influential,
        },
        "summary": {
            "fp16": {
                "baseline_mae": fp16_results["baseline_mae"],
                "best_model": min(fp16_results["models"].keys(), key=lambda k: fp16_results["models"][k]["mae"]),
                "best_mae": min(fp16_results["models"][k]["mae"] for k in fp16_results["models"]),
                "beats_baseline": any(
                    fp16_results["models"][k]["mae"] < fp16_results["baseline_mae"] 
                    for k in fp16_results["models"] if k != "baseline"
                ),
            },
            "int4": {
                "baseline_mae": int4_results["baseline_mae"],
                "best_model": min(int4_results["models"].keys(), key=lambda k: int4_results["models"][k]["mae"]),
                "best_mae": min(int4_results["models"][k]["mae"] for k in int4_results["models"]),
                "beats_baseline": any(
                    int4_results["models"][k]["mae"] < int4_results["baseline_mae"] 
                    for k in int4_results["models"] if k != "baseline"
                ),
            },
        }
    }
    
    # Save report
    with open(REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2, default=str)
    
    print(f"\n{'=' * 70}")
    print("FEASIBILITY REPORT SAVED")
    print(f"{'=' * 70}")
    print(f"Report: {REPORT_PATH}")
    
    # Print summary
    print("\n--- SUMMARY ---")
    for config in ["fp16", "int4"]:
        r = report["summary"][config]
        print(f"\n{config.upper()}:")
        print(f"  Baseline MAE: {r['baseline_mae']:.6f} Wh")
        print(f"  Best model: {r['best_model']} (MAE={r['best_mae']:.6f} Wh)")
        print(f"  Beats baseline: {r['beats_baseline']}")
        
        # Detailed model comparison
        print(f"  Model comparison:")
        for name, m in report[config]["models"].items():
            print(f"    {name}: MAE={m['mae']:.6f}, RMSE={m['rmse']:.6f}, R2={m['r2']:.4f}")
    
    print(f"\nLinear Regression:")
    for config in ["fp16", "int4"]:
        lr = report["linear_regression"][config]
        print(f"  {config.upper()}: MAE={lr['mae']:.6f}, RMSE={lr['rmse']:.6f}, R2={lr['r2']:.4f}")
    
    if fp16_influential:
        print(f"\nInfluential FP16 prompts (>2 std): {len(fp16_influential)}")
        for p in fp16_influential[:3]:
            print(f"  {p['prompt_id']}: true={p['true_energy']:.6f}, max_err={p['max_abs_error']:.6f}")
    if int4_influential:
        print(f"\nInfluential INT4 prompts (>2 std): {len(int4_influential)}")
        for p in int4_influential[:3]:
            print(f"  {p['prompt_id']}: true={p['true_energy']:.6f}, max_err={p['max_abs_error']:.6f}")
    
    # Feasibility verdict
    fp16_beats = report["summary"]["fp16"]["beats_baseline"]
    int4_beats = report["summary"]["int4"]["beats_baseline"]
    
    print(f"\n{'=' * 70}")
    print("FEASIBILITY VERDICT")
    print(f"{'=' * 70}")
    print(f"FP16 beats baseline: {fp16_beats}")
    print(f"INT4 beats baseline: {int4_beats}")
    print(f"Both beat baseline: {fp16_beats and int4_beats}")
    
    if fp16_beats and int4_beats:
        print("\n✓ FEASIBLE: Both configs show predictive signal beyond static mean.")
        print("  → Consider 100-prompt expansion for robust evaluation.")
    else:
        print("\n✗ NOT FEASIBLE: Pre-inference features do not reliably predict energy.")
        print("  → Do not expand. Report negative result.")
    
    return report


if __name__ == "__main__":
    main()