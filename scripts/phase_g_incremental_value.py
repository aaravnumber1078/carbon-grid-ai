#!/usr/bin/env python3
"""
Phase G Incremental Value Experiment

Determines whether the 20 additional pre-inference features improve energy prediction
beyond max_tokens_requested alone. Uses the same LOPO-CV folds and 25-prompt pilot data.
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
from sklearn.model_selection import LeaveOneOut, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ============================================================
# CONFIGURATION
# ============================================================

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

# 21 approved pre-inference features
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

# Single feature baseline
MAX_TOKENS_IDX = FEATURE_NAMES.index("max_tokens_requested")

PILOT_PROMPTS_PATH = Path("C:/CARBON GRID AI/data/evaluation/phase_g/pilot_prompts.json")
PILOT_RESULTS_PATH = Path("C:/CARBON GRID AI/data/evaluation/phase_g/pilot_results.json")
OUTPUT_DIR = Path("C:/CARBON GRID AI/data/evaluation/phase_g")
REPORT_JSON_PATH = OUTPUT_DIR / "phase_g_incremental_value_report.json"
REPORT_MD_PATH = OUTPUT_DIR / "PHASE_G_INCREMENTAL_VALUE_REPORT.md"


# ============================================================
# DATA LOADING
# ============================================================

def load_pilot_data():
    """Load and aggregate pilot data."""
    with open(PILOT_PROMPTS_PATH) as f:
        pilot_data = json.load(f)
    prompts = pilot_data["pilot_prompts"]
    prompt_df = pd.DataFrame(prompts)

    with open(PILOT_RESULTS_PATH) as f:
        results_data = json.load(f)
    runs = results_data["runs"]
    runs_df = pd.DataFrame(runs)
    successful = runs_df[runs_df["success"] == True].copy()

    # Aggregate: mean energy per prompt per configuration
    agg = successful.groupby(["prompt_id", "configuration"]).agg(
        energy_wh=("energy_wh", "mean"),
    ).reset_index()

    merged = agg.merge(prompt_df, on="prompt_id", how="left")
    return merged


def prepare_config_data(merged, config):
    """Extract X, y for a specific configuration."""
    config_data = merged[merged["configuration"] == config].copy()
    config_data = config_data.sort_values("prompt_id").reset_index(drop=True)
    X = config_data[FEATURE_NAMES].values.astype(float)
    y = config_data["energy_wh"].values.astype(float)
    prompt_ids = config_data["prompt_id"].tolist()
    return X, y, prompt_ids


# ============================================================
# MODEL EVALUATION
# ============================================================

def evaluate_models_lopo(X, y, prompt_ids, config):
    """Run LOPO-CV for three model variants."""
    loo = LeaveOneOut()
    n_samples = len(y)
    
    # Model definitions
    models = {
        "LinearReg_maxTokens": (
            Pipeline([("scaler", StandardScaler()), ("model", LinearRegression())]),
            {}  # no hyperparameters
        ),
        "Ridge_full": (
            Pipeline([("scaler", StandardScaler()), ("model", Ridge(random_state=RANDOM_SEED))]),
            {"model__alpha": np.logspace(-3, 3, 13)}
        ),
        "ElasticNet_full": (
            Pipeline([("scaler", StandardScaler()), ("model", ElasticNet(random_state=RANDOM_SEED, max_iter=10000))]),
            {"model__alpha": np.logspace(-3, 1, 9), "model__l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9]}
        ),
    }
    
    # For single-feature baseline, we need a separate pipeline
    def make_single_feature_pipeline():
        return Pipeline([("scaler", StandardScaler()), ("model", LinearRegression())])
    
    # Storage
    pred_storage = {name: np.zeros(n_samples) for name in models}
    pred_storage["LinearReg_maxTokens"] = np.zeros(n_samples)
    
    fold = 0
    for train_idx, test_idx in loo.split(X):
        fold += 1
        
        X_train_full, X_test_full = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Single-feature baseline (max_tokens_requested only)
        X_train_one = X_train_full[:, MAX_TOKENS_IDX:MAX_TOKENS_IDX+1]
        X_test_one = X_test_full[:, MAX_TOKENS_IDX:MAX_TOKENS_IDX+1]
        
        scaler_one = StandardScaler()
        X_train_one_scaled = scaler_one.fit_transform(X_train_one)
        X_test_one_scaled = scaler_one.transform(X_test_one)
        
        lr_one = LinearRegression()
        lr_one.fit(X_train_one_scaled, y_train)
        pred_storage["LinearReg_maxTokens"][test_idx] = lr_one.predict(X_test_one_scaled)
        
        # Full-feature models with inner CV
        for name, (pipeline, param_grid) in models.items():
            inner_cv = LeaveOneOut() if len(train_idx) >= 5 else 2
            
            grid_search = GridSearchCV(
                pipeline, param_grid,
                cv=inner_cv,
                scoring="neg_mean_absolute_error",
                n_jobs=-1,
                verbose=0
            )
            
            try:
                grid_search.fit(X_train_full, y_train)
                y_pred = grid_search.predict(X_test_full)
                pred_storage[name][test_idx] = y_pred
            except Exception as e:
                print(f"    WARNING: {name} failed on fold {fold}: {e}")
                pred_storage[name][test_idx] = np.mean(y_train)
    
    # Compute metrics and per-prompt errors
    results = {}
    for name in list(models.keys()) + ["LinearReg_maxTokens"]:
        y_pred = pred_storage[name]
        mae = mean_absolute_error(y, y_pred)
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        r2 = r2_score(y, y_pred)
        
        # Count negative predictions
        n_negative = np.sum(y_pred < 0)
        
        # Per-prompt errors
        errors = y - y_pred
        
        results[name] = {
            "mae": float(mae),
            "rmse": float(rmse),
            "r2": float(r2),
            "n_negative_predictions": int(n_negative),
            "predictions": y_pred.tolist(),
            "errors": errors.tolist(),
        }
    
    # Paired error differences: extended vs baseline
    baseline_errors = np.array(results["LinearReg_maxTokens"]["errors"])
    extended_errors = {}
    for name in ["Ridge_full", "ElasticNet_full"]:
        ext_errors = np.array(results[name]["errors"])
        diff = ext_errors - baseline_errors  # positive = extended worse
        extended_errors[name] = {
            "mean_diff": float(np.mean(diff)),
            "median_diff": float(np.median(diff)),
            "n_better": int(np.sum(diff < 0)),  # extended better (lower error)
            "n_worse": int(np.sum(diff > 0)),
            "n_tied": int(np.sum(diff == 0)),
            "per_prompt_diff": diff.tolist(),
        }
    
    return results, extended_errors, pred_storage


def analyze_prompt_family_similarity(prompt_ids, prompt_df):
    """Check for near-duplicate prompts within pilot that could inflate LOPO-CV."""
    # We don't have text embeddings, but we can check for same template family
    # by looking at prompts with same category + input_band + output_band
    families = defaultdict(list)
    for pid in prompt_ids:
        row = prompt_df[prompt_df["prompt_id"] == pid].iloc[0]
        key = (row["category"], row["input_length_band"], row["output_length_band"])
        families[key].append(pid)
    
    duplicated = {k: v for k, v in families.items() if len(v) > 1}
    return duplicated


def main():
    print("=" * 70)
    print("PHASE G INCREMENTAL VALUE EXPERIMENT")
    print("=" * 70)
    
    merged = load_pilot_data()
    with open(PILOT_PROMPTS_PATH) as f:
        pilot_df = pd.DataFrame(json.load(f)["pilot_prompts"])
    
    # Prompt family check
    all_prompt_ids = merged["prompt_id"].unique().tolist()
    families = analyze_prompt_family_similarity(all_prompt_ids, pilot_df)
    print(f"\nPrompt families with multiple members: {len(families)}")
    for fam, pids in families.items():
        print(f"  {fam}: {pids}")
    
    # Evaluate each config
    all_results = {}
    all_paired = {}
    
    for config in ["fp16", "int4"]:
        print(f"\n--- {config.upper()} ---")
        X, y, prompt_ids = prepare_config_data(merged, config)
        print(f"Target range: [{y.min():.6f}, {y.max():.6f}], mean={y.mean():.6f}, std={y.std():.6f}")
        
        results, paired, preds = evaluate_models_lopo(X, y, prompt_ids, config)
        all_results[config] = results
        all_paired[config] = paired
        
        # Print summary
        print(f"\n  Model comparison ({config.upper()}):")
        for name, m in results.items():
            print(f"    {name}: MAE={m['mae']:.6f}, RMSE={m['rmse']:.6f}, R2={m['r2']:.4f}, neg_preds={m['n_negative_predictions']}")
        
        print(f"\n  Paired comparison (extended - baseline):")
        for name, diff in paired.items():
            print(f"    {name}: mean_diff={diff['mean_diff']:+.6f}, median={diff['median_diff']:+.6f}, better={diff['n_better']}, worse={diff['n_worse']}, tied={diff['n_tied']}")
    
    # Compile report
    report = {
        "metadata": {
            "phase": "G.1.2_incremental_value",
            "timestamp": pd.Timestamp.now().isoformat(),
            "pilot_prompts": 25,
            "repetitions_aggregated": 3,
            "configs_evaluated": ["fp16", "int4"],
            "cv_method": "LeaveOneOut (LOPO-CV)",
            "features_used": FEATURE_NAMES,
            "n_features": len(FEATURE_NAMES),
            "n_samples_per_config": 25,
            "random_seed": RANDOM_SEED,
            "prompt_families_with_duplicates": len(families),
        },
        "results": all_results,
        "paired_comparison": all_paired,
        "prompt_families": {k: v for k, v in families.items()},
    }
    
    # Save JSON
    with open(REPORT_JSON_PATH, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nJSON report saved: {REPORT_JSON_PATH}")
    
    # Generate Markdown report
    generate_markdown_report(report, REPORT_MD_PATH)
    print(f"Markdown report saved: {REPORT_MD_PATH}")
    
    # Decision
    print(f"\n{'=' * 70}")
    print("INCREMENTAL VALUE VERDICT")
    print(f"{'=' * 70}")
    
    for config in ["fp16", "int4"]:
        base_mae = all_results[config]["LinearReg_maxTokens"]["mae"]
        en_mae = all_results[config]["ElasticNet_full"]["mae"]
        diff = en_mae - base_mae
        paired = all_paired[config]["ElasticNet_full"]
        
        print(f"\n{config.upper()}:")
        print(f"  Baseline (maxTokens only): {base_mae:.6f} Wh")
        print(f"  Extended (21 features):    {en_mae:.6f} Wh")
        print(f"  Difference:                {diff:+.6f} Wh")
        print(f"  Per-prompt: better={paired['n_better']}, worse={paired['n_worse']}, tied={paired['n_tied']}")
        
        if diff < -0.001 and paired['n_better'] > paired['n_worse']:
            print(f"  -> Meaningful improvement")
        elif diff > 0.001 and paired['n_worse'] > paired['n_better']:
            print(f"  -> Extended model worse")
        else:
            print(f"  -> No meaningful improvement (inconclusive or negligible)")
    
    return report


def generate_markdown_report(report, path):
    """Generate concise Markdown report."""
    lines = [
        "# Phase G Incremental Value Report",
        "",
        f"**Date:** {report['metadata']['timestamp']}",
        f"**Pilot prompts:** {report['metadata']['pilot_prompts']} (25 prompts × 3 reps × 2 configs)",
        f"**CV method:** {report['metadata']['cv_method']}",
        f"**Features evaluated:** {report['metadata']['n_features']} pre-inference features",
        f"**Prompt families with duplicates:** {report['metadata']['prompt_families_with_duplicates']}",
        "",
        "## Model Comparison",
        "",
        "| Config | Model | MAE (Wh) | RMSE (Wh) | R² | Negative Predictions |",
        "|--------|-------|----------|-----------|-----|----------------------|",
    ]
    
    for config in ["fp16", "int4"]:
        for name, m in report["results"][config].items():
            lines.append(f"| {config.upper()} | {name} | {m['mae']:.6f} | {m['rmse']:.6f} | {m['r2']:.4f} | {m['n_negative_predictions']} |")
    
    lines.extend([
        "",
        "## Paired Comparison (Extended - Baseline)",
        "",
        "| Config | Extended Model | Mean Diff (Wh) | Median Diff (Wh) | Better | Worse | Tied |",
        "|--------|----------------|----------------|------------------|--------|-------|------|",
    ])
    
    for config in ["fp16", "int4"]:
        for name, diff in report["paired_comparison"][config].items():
            lines.append(f"| {config.upper()} | {name} | {diff['mean_diff']:+.6f} | {diff['median_diff']:+.6f} | {diff['n_better']} | {diff['n_worse']} | {diff['n_tied']} |")
    
    lines.extend([
        "",
        "## Negative Predictions",
        "",
        "Physically impossible negative energy predictions observed for ElasticNet_full:",
    ])
    
    for config in ["fp16", "int4"]:
        neg = report["results"][config]["ElasticNet_full"]["n_negative_predictions"]
        lines.append(f"- {config.upper()}: {neg}/25 predictions < 0 Wh")
    
    lines.extend([
        "",
        "## Prompt Family Overlap",
        "",
        f"Found {report['metadata']['prompt_families_with_duplicates']} template families with multiple prompts in the 25-prompt pilot.",
        "This may inflate LOPO-CV scores if similar prompts appear in both train and test folds.",
        "",
        "## Conclusion",
        "",
    ])
    
    # Decision logic
    fp16_base = report["results"]["fp16"]["LinearReg_maxTokens"]["mae"]
    fp16_ext = report["results"]["fp16"]["ElasticNet_full"]["mae"]
    int4_base = report["results"]["int4"]["LinearReg_maxTokens"]["mae"]
    int4_ext = report["results"]["int4"]["ElasticNet_full"]["mae"]
    
    fp16_diff = fp16_ext - fp16_base
    int4_diff = int4_ext - int4_base
    
    fp16_paired = report["paired_comparison"]["fp16"]["ElasticNet_full"]
    int4_paired = report["paired_comparison"]["int4"]["ElasticNet_full"]
    
    if fp16_diff < -0.001 and fp16_paired['n_better'] > fp16_paired['n_worse']:
        fp16_verdict = "IMPROVES"
    elif fp16_diff > 0.001 and fp16_paired['n_worse'] > fp16_paired['n_better']:
        fp16_verdict = "WORSENS"
    else:
        fp16_verdict = "NO MEANINGFUL IMPROVEMENT"
    
    if int4_diff < -0.001 and int4_paired['n_better'] > int4_paired['n_worse']:
        int4_verdict = "IMPROVES"
    elif int4_diff > 0.001 and int4_paired['n_worse'] > int4_paired['n_better']:
        int4_verdict = "WORSENS"
    else:
        int4_verdict = "NO MEANINGFUL IMPROVEMENT"
    
    lines.extend([
        f"- **FP16:** {fp16_verdict} (diff = {fp16_diff:+.6f} Wh, better={fp16_paired['n_better']}, worse={fp16_paired['n_worse']})",
        f"- **INT4:** {int4_verdict} (diff = {int4_diff:+.6f} Wh, better={int4_paired['n_better']}, worse={int4_paired['n_worse']})",
        "",
        "**Recommendation:**",
    ])
    
    if fp16_verdict == "IMPROVES" and int4_verdict == "IMPROVES":
        lines.append("The 20 additional features show consistent improvement. Consider 100-prompt expansion.")
    elif fp16_verdict == "WORSENS" or int4_verdict == "WORSENS":
        lines.append("Extended model performs worse. Use output-length baseline only.")
    else:
        lines.append("No meaningful improvement from extra features. Use simple output-length baseline (`energy ≈ a × max_tokens_requested + b`). Do not expand dataset for ML predictor.")
    
    lines.extend([
        "",
        "**Limitation:** n=25 is small; LOPO-CV variance is high. Prompt family overlap not fully controlled.",
        "R² alone does not imply practical utility for routing decisions.",
    ])
    
    with open(path, "w") as f:
        f.write("\n".join(lines))


if __name__ == "__main__":
    main()