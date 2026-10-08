#!/usr/bin/env python3
"""
Phase F: Analysis and Statistical Evaluation

Loads raw results and produces:
- Summary statistics
- Statistical tests
- Error analysis
- Visualizations (if matplotlib available)
"""

import sys
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict, Counter
from dataclasses import dataclass, asdict

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("Warning: scipy not available, statistical tests will be skipped")

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Warning: matplotlib not available, plots will be skipped")


@dataclass
class SummaryStats:
    """Summary statistics for a metric."""
    n: int
    mean: float
    median: float
    std: float
    iqr: float
    p25: float
    p75: float
    p95: Optional[float]
    min: float
    max: float


def load_results(results_path: Path) -> Dict[str, List[Dict]]:
    """Load raw results from JSON."""
    with open(results_path, 'r') as f:
        data = json.load(f)
    return data.get("results", {})


def compute_summary(values: List[float]) -> SummaryStats:
    """Compute summary statistics."""
    # Filter out non-numeric values
    numeric_values = []
    for v in values:
        try:
            numeric_values.append(float(v))
        except (ValueError, TypeError):
            continue
    
    if not numeric_values:
        return SummaryStats(0, 0, 0, 0, 0, 0, 0, None, 0, 0)
    
    arr = np.array(numeric_values)
    arr = arr[~np.isnan(arr)]
    
    if len(arr) == 0:
        return SummaryStats(0, 0, 0, 0, 0, 0, 0, None, 0, 0)
    
    p95 = float(np.percentile(arr, 95)) if len(arr) >= 20 else None
    
    return SummaryStats(
        n=len(arr),
        mean=float(np.mean(arr)),
        median=float(np.median(arr)),
        std=float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
        iqr=float(np.percentile(arr, 75) - np.percentile(arr, 25)),
        p25=float(np.percentile(arr, 25)),
        p75=float(np.percentile(arr, 75)),
        p95=p95,
        min=float(np.min(arr)),
        max=float(np.max(arr)),
    )


def wilcoxon_paired(x: List[float], y: List[float]) -> Dict[str, Any]:
    """Wilcoxon signed-rank test for paired samples."""
    if not SCIPY_AVAILABLE:
        return {"test": "wilcoxon", "available": False}
    
    # Filter to paired (same length)
    min_len = min(len(x), len(y))
    x_arr = np.array(x[:min_len])
    y_arr = np.array(y[:min_len])
    
    # Remove NaN pairs
    mask = ~(np.isnan(x_arr) | np.isnan(y_arr))
    x_clean = x_arr[mask]
    y_clean = y_arr[mask]
    
    if len(x_clean) < 10:
        return {"test": "wilcoxon", "n": len(x_clean), "skipped": "insufficient_pairs"}
    
    # Check for constant identical values
    diff = x_clean - y_clean
    if np.all(diff == 0):
        return {
            "test": "wilcoxon",
            "n": len(x_clean),
            "skipped": "constant_identical_values",
            "message": "Statistical test not applicable because both groups have constant identical values.",
            "median_difference": 0.0,
            "direction": "none"
        }
    
    try:
        stat, p = stats.wilcoxon(x_clean, y_clean, alternative='two-sided')
        # Effect size: rank-biserial correlation
        n = len(x_clean)
        r = 1 - (2 * stat) / (n * (n + 1) / 2)
        
        # Direction of difference
        median_diff = float(np.median(x_clean - y_clean))
        direction = "x_greater" if median_diff > 0 else "y_greater" if median_diff < 0 else "none"
        
        return {
            "test": "wilcoxon",
            "n": n,
            "statistic": float(stat),
            "p_value": float(p),
            "effect_size_r": float(r),
            "median_difference": median_diff,
            "direction": direction,
            "significant": p < 0.05,
        }
    except Exception as e:
        return {"test": "wilcoxon", "error": str(e)}


def analyze_policy_comparison(results: Dict[str, List[Dict]]) -> Dict[str, Any]:
    """Compare three policies across all metrics."""
    
    # Extract metrics by policy
    metrics_by_policy = {}
    for policy_name, records in results.items():
        metrics_by_policy[policy_name] = {
            "latency_ms": [r["latency_ms"] for r in records if r["success"]],
            "energy_wh": [r["energy_wh"] for r in records if r["success"] and r["energy_wh"] is not None],
            "co2_g": [r["co2_g"] for r in records if r["success"] and r["co2_g"] is not None],
            "quality_score": [r["quality_score"] for r in records if r["success"] and r["quality_score"] is not None],
            "selected_precision": [r["selected_precision"] for r in records if r["success"]],
            "decision_reason": [r["decision_reason"] for r in records if r["success"]],
            "fallback_used": [r.get("fallback_used", False) for r in records if r["success"]],
            "safety_probability": [r["safety_probability"] for r in records if r["success"] and r["safety_probability"] is not None],
        }
    
    # Summary statistics
    summary = {}
    for policy_name, metrics in metrics_by_policy.items():
        summary[policy_name] = {}
        for metric_name, values in metrics.items():
            if values:
                summary[policy_name][metric_name] = asdict(compute_summary(values))
    
    # Pairwise comparisons (CarbonGrid vs baselines)
    comparisons = {}
    
    # CarbonGrid vs Always FP16
    if "carbongrid" in metrics_by_policy and "always_fp16" in metrics_by_policy:
        cg = metrics_by_policy["carbongrid"]
        fp16 = metrics_by_policy["always_fp16"]
        
        comparisons["carbongrid_vs_always_fp16"] = {
            "latency_ms": wilcoxon_paired(cg["latency_ms"], fp16["latency_ms"]),
            "energy_wh": wilcoxon_paired(cg["energy_wh"], fp16["energy_wh"]),
            "co2_g": wilcoxon_paired(cg["co2_g"], fp16["co2_g"]),
            "quality_score": wilcoxon_paired(cg["quality_score"], fp16["quality_score"]),
        }
        
        # Relative changes
        for metric in ["latency_ms", "energy_wh", "co2_g", "quality_score"]:
            cg_vals = cg[metric]
            fp16_vals = fp16[metric]
            if cg_vals and fp16_vals:
                cg_med = np.median(cg_vals)
                fp16_med = np.median(fp16_vals)
                if fp16_med != 0:
                    pct_change = (cg_med - fp16_med) / fp16_med * 100
                    comparisons["carbongrid_vs_always_fp16"][f"{metric}_pct_change"] = pct_change
                    comparisons["carbongrid_vs_always_fp16"][f"{metric}_median_diff"] = cg_med - fp16_med
    
    # CarbonGrid vs Always INT4
    if "carbongrid" in metrics_by_policy and "always_int4" in metrics_by_policy:
        cg = metrics_by_policy["carbongrid"]
        int4 = metrics_by_policy["always_int4"]
        
        comparisons["carbongrid_vs_always_int4"] = {
            "latency_ms": wilcoxon_paired(cg["latency_ms"], int4["latency_ms"]),
            "energy_wh": wilcoxon_paired(cg["energy_wh"], int4["energy_wh"]),
            "co2_g": wilcoxon_paired(cg["co2_g"], int4["co2_g"]),
            "quality_score": wilcoxon_paired(cg["quality_score"], int4["quality_score"]),
        }
        
        for metric in ["latency_ms", "energy_wh", "co2_g", "quality_score"]:
            cg_vals = cg[metric]
            int4_vals = int4[metric]
            if cg_vals and int4_vals:
                cg_med = np.median(cg_vals)
                int4_med = np.median(int4_vals)
                if int4_med != 0:
                    pct_change = (cg_med - int4_med) / int4_med * 100
                    comparisons["carbongrid_vs_always_int4"][f"{metric}_pct_change"] = pct_change
                    comparisons["carbongrid_vs_always_int4"][f"{metric}_median_diff"] = cg_med - int4_med
    
    # Selection rates for CarbonGrid
    if "carbongrid" in metrics_by_policy:
        cg = metrics_by_policy["carbongrid"]
        precisions = cg["selected_precision"]
        reasons = cg["decision_reason"]
        fallbacks = cg.get("fallback_used", [])
        
        comparisons["carbongrid_selection"] = {
            "precision_distribution": {
                "fp16": precisions.count("fp16"),
                "int4": precisions.count("int4"),
            },
            "decision_reasons": dict(Counter(reasons)),
            "fallback_rate": sum(fallbacks) / len(fallbacks) if fallbacks else 0,
            "avg_safety_prob": np.mean(cg["safety_probability"]) if cg.get("safety_probability") else None,
        }
    
    return {
        "summary": summary,
        "comparisons": comparisons,
        "metadata": {
            "policies_compared": list(metrics_by_policy.keys()),
            "total_records_per_policy": {k: len(v["latency_ms"]) for k, v in metrics_by_policy.items()},
        }
    }


def analyze_by_strata(results: Dict[str, List[Dict]]) -> Dict[str, Any]:
    """Analyze results stratified by task_type, complexity, B2 safe label, separated by policy."""
    
    strata = {
        "by_policy_and_task_type": defaultdict(lambda: defaultdict(lambda: defaultdict(list))),
        "by_policy_and_complexity": defaultdict(lambda: defaultdict(lambda: defaultdict(list))),
        "by_policy_and_b2_safe": defaultdict(lambda: defaultdict(lambda: defaultdict(list))),
    }
    
    for policy_name, records in results.items():
        for r in records:
            if not r["success"]:
                continue
            
            task = r["task_type"]
            complexity = r["complexity_level"]
            b2_safe = r["b2_safe_to_quantize"]
            
            for metric in ["latency_ms", "energy_wh", "co2_g", "quality_score"]:
                val = r.get(metric)
                if val is not None:
                    strata["by_policy_and_task_type"][policy_name][task][metric].append(val)
                    strata["by_policy_and_complexity"][policy_name][complexity][metric].append(val)
                    strata["by_policy_and_b2_safe"][policy_name][b2_safe][metric].append(val)
    
    # Compute summaries per stratum
    stratified_summary = {}
    for stratum_name, stratum_data in strata.items():
        stratified_summary[stratum_name] = {}
        for policy_name, policy_data in stratum_data.items():
            stratified_summary[stratum_name][policy_name] = {}
            for group_name, metrics in policy_data.items():
                stratified_summary[stratum_name][policy_name][group_name] = {}
                for metric_name, values in metrics.items():
                    if values:
                        stratified_summary[stratum_name][policy_name][group_name][metric_name] = asdict(compute_summary(values))
    
    return stratified_summary


def error_analysis(results: Dict[str, List[Dict]]) -> Dict[str, Any]:
    """Identify and categorize post-hoc divergence cases."""
    
    divergences = {
        "carbongrid_posthoc_energy_divergence": [],
        "carbongrid_latency_violation": [],
        "safety_gate_rejections": [],
        "quality_degradation": [],
        "inference_failures": [],
    }
    
    if "carbongrid" not in results:
        return divergences
    
    cg_records = results["carbongrid"]
    fp16_records = {r["prompt_id"]: r for r in results.get("always_fp16", []) if r["success"]}
    int4_records = {r["prompt_id"]: r for r in results.get("always_int4", []) if r["success"]}
    
    for r in cg_records:
        if not r["success"]:
            divergences["inference_failures"].append({
                "prompt_id": r["prompt_id"],
                "error": r["error"],
            })
            continue
        
        prompt_id = r["prompt_id"]
        
        # Safety gate rejections
        if r.get("decision_reason") == "safety_gate_failed":
            divergences["safety_gate_rejections"].append({
                "prompt_id": prompt_id,
                "safety_prob": r.get("safety_probability"),
                "threshold": r.get("safety_threshold"),
            })
        
        # Latency violations
        if r.get("decision_reason") in ["int4_latency_violation", "fp16_latency_violation", "both_latency_violation"]:
            divergences["carbongrid_latency_violation"].append({
                "prompt_id": prompt_id,
                "reason": r["decision_reason"],
                "selected": r["selected_precision"],
                "latency_req": r.get("latency_requirement_ms"),
            })
        
        # Quality degradation (INT4 selected but quality < 0.7)
        if r["selected_precision"] == "int4" and r.get("quality_score") is not None:
            if r["quality_score"] < 0.7:
                divergences["quality_degradation"].append({
                    "prompt_id": prompt_id,
                    "quality_score": r["quality_score"],
                    "task_type": r["task_type"],
                })
        
        # Post-hoc energy divergence
        if prompt_id in fp16_records and prompt_id in int4_records:
            fp16_e = fp16_records[prompt_id]["energy_wh"]
            int4_e = int4_records[prompt_id]["energy_wh"]
            cg_e = r["energy_wh"]
            
            if fp16_e and int4_e and cg_e:
                # Decision-time estimates
                fp16_est = r.get("fp16_estimated_energy_wh")
                int4_est = r.get("int4_estimated_energy_wh")
                
                if r["selected_precision"] == "fp16" and int4_e < fp16_e:
                    divergences["carbongrid_posthoc_energy_divergence"].append({
                        "prompt_id": prompt_id,
                        "decision_time_fp16_estimate_wh": fp16_est,
                        "decision_time_int4_estimate_wh": int4_est,
                        "posthoc_measured_fp16_energy_wh": fp16_e,
                        "posthoc_measured_int4_energy_wh": int4_e,
                        "classifier_probability": r.get("safety_probability"),
                        "decision_reason": r.get("decision_reason"),
                        "selected_precision": r.get("selected_precision"),
                        "divergence_type": "selected_fp16_but_int4_lower_energy",
                    })
                elif r["selected_precision"] == "int4" and int4_e > fp16_e:
                    divergences["carbongrid_posthoc_energy_divergence"].append({
                        "prompt_id": prompt_id,
                        "decision_time_fp16_estimate_wh": fp16_est,
                        "decision_time_int4_estimate_wh": int4_est,
                        "posthoc_measured_fp16_energy_wh": fp16_e,
                        "posthoc_measured_int4_energy_wh": int4_e,
                        "classifier_probability": r.get("safety_probability"),
                        "decision_reason": r.get("decision_reason"),
                        "selected_precision": r.get("selected_precision"),
                        "divergence_type": "selected_int4_but_int4_higher_energy",
                    })
    
    return divergences


def generate_plots(results: Dict[str, List[Dict]], output_dir: Path):
    """Generate visualization plots if matplotlib available."""
    if not MATPLOTLIB_AVAILABLE:
        return
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract data
    policies = ["always_fp16", "always_int4", "carbongrid"]
    colors = {"always_fp16": "#1f77b4", "always_int4": "#ff7f0e", "carbongrid": "#2ca02c"}
    
    metrics = [
        ("latency_ms", "Latency (ms)"),
        ("energy_wh", "Energy (Wh)"),
        ("co2_g", "CO₂ (g)"),
        ("quality_score", "Quality Score"),
    ]
    
    for metric_name, metric_label in metrics:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Box plot
        ax = axes[0]
        data = []
        labels = []
        for policy in policies:
            vals = [r[metric_name] for r in results.get(policy, []) if r["success"] and r.get(metric_name) is not None]
            if vals:
                data.append(vals)
                labels.append(policy)
        
        if data:
            bp = ax.boxplot(data, labels=labels, patch_artist=True)
            for patch, policy in zip(bp['boxes'], labels):
                patch.set_facecolor(colors.get(policy, 'gray'))
            ax.set_ylabel(metric_label)
            ax.set_title(f"{metric_label} Distribution by Policy")
        
        # Bar chart with error bars (median + IQR)
        ax = axes[1]
        medians = []
        iqr_low = []
        iqr_high = []
        for policy in policies:
            vals = [r[metric_name] for r in results.get(policy, []) if r["success"] and r.get(metric_name) is not None]
            if vals:
                medians.append(np.median(vals))
                iqr_low.append(np.median(vals) - np.percentile(vals, 25))
                iqr_high.append(np.percentile(vals, 75) - np.median(vals))
            else:
                medians.append(0)
                iqr_low.append(0)
                iqr_high.append(0)
        
        x = range(len(policies))
        ax.bar(x, medians, yerr=[iqr_low, iqr_high], capsize=5,
               color=[colors[p] for p in policies], edgecolor='black')
        ax.set_xticks(x)
        ax.set_xticklabels(policies)
        ax.set_ylabel(f"Median {metric_label}")
        ax.set_title(f"Median {metric_label} by Policy (error bars = IQR)")
        
        plt.tight_layout()
        plt.savefig(output_dir / f"phase_f_{metric_name}.png", dpi=150)
        plt.close()
    
    # CarbonGrid precision selection pie chart
    if "carbongrid" in results:
        cg = [r for r in results["carbongrid"] if r["success"]]
        fp16_count = sum(1 for r in cg if r["selected_precision"] == "fp16")
        int4_count = sum(1 for r in cg if r["selected_precision"] == "int4")
        
        if fp16_count + int4_count > 0:
            fig, ax = plt.subplots(figsize=(6, 6))
            ax.pie([fp16_count, int4_count], labels=['FP16', 'INT4'], 
                   colors=['#1f77b4', '#ff7f0e'], autopct='%1.1f%%')
            ax.set_title("CarbonGrid Precision Selection")
            plt.savefig(output_dir / "phase_f_precision_selection.png", dpi=150)
            plt.close()
    
    # Decision reason distribution
    if "carbongrid" in results:
        reasons = [r["decision_reason"] for r in results["carbongrid"] if r["success"]]
        reason_counts = Counter(reasons)
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(list(reason_counts.keys()), list(reason_counts.values()), color='#2ca02c')
        ax.set_xlabel("Count")
        ax.set_title("CarbonGrid Decision Reasons")
        plt.tight_layout()
        plt.savefig(output_dir / "phase_f_decision_reasons.png", dpi=150)
        plt.close()
    
    print(f"Plots saved to {output_dir}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze Phase F results")
    parser.add_argument("--results", default="data/evaluation/phase_f/phase_f_raw_results.json")
    parser.add_argument("--output", default="data/evaluation/phase_f")
    parser.add_argument("--no-plots", action="store_true")
    
    args = parser.parse_args()
    
    results_path = Path(args.results)
    if not results_path.exists():
        print(f"Error: Results file not found: {results_path}")
        return 1
    
    print(f"Loading results from {results_path}...")
    results = load_results(results_path)
    
    print("\n=== Policy Comparison Analysis ===")
    analysis = analyze_policy_comparison(results)
    
    # Print summary
    print("\n--- Summary Statistics ---")
    for policy, metrics in analysis["summary"].items():
        print(f"\n{policy}:")
        for metric, stats in metrics.items():
            if isinstance(stats, dict) and "median" in stats:
                print(f"  {metric}: n={stats['n']}, median={stats['median']:.3f}, IQR=[{stats['p25']:.3f}, {stats['p75']:.3f}], mean={stats['mean']:.3f}±{stats['std']:.3f}")
    
    print("\n--- Pairwise Comparisons ---")
    for comp_name, comp_data in analysis["comparisons"].items():
        if comp_name in ["carbongrid_selection"]:
            print(f"\n{comp_name}:")
            for k, v in comp_data.items():
                print(f"  {k}: {v}")
        else:
            print(f"\n{comp_name}:")
            for metric, test_result in comp_data.items():
                if isinstance(test_result, dict) and "p_value" in test_result:
                    sig = "*" if test_result.get("significant") else ""
                    median_diff = test_result.get('median_difference', 'N/A')
                    direction = test_result.get('direction', 'N/A')
                    print(f"  {metric}: n={test_result.get('n', 'N/A')}, median_diff={median_diff:.3f}, direction={direction}, p={test_result['p_value']:.4f} {sig}, r={test_result.get('effect_size_r', 'N/A')}")
                elif isinstance(test_result, dict) and "skipped" in test_result:
                    print(f"  {metric}: {test_result['skipped']} - {test_result.get('message', '')}")
                elif isinstance(test_result, (int, float)):
                    if "_pct_change" in metric:
                        print(f"  {metric}: {test_result:.2f}%")
                    elif "_median_diff" in metric:
                        print(f"  {metric}: {test_result:.3f}")
                    else:
                        print(f"  {metric}: {test_result:.2f}")
    
    # Stratified analysis (by policy)
    print("\n=== Stratified Analysis (by Policy) ===")
    stratified = analyze_by_strata(results)
    for stratum_name, stratum_data in stratified.items():
        print(f"\n{stratum_name}:")
        for policy_name, policy_data in stratum_data.items():
            print(f"  {policy_name}:")
            for group_name, metrics in list(policy_data.items())[:5]:  # Show first 5
                print(f"    {group_name}:")
                for metric_name, stats in metrics.items():
                    if isinstance(stats, dict) and "median" in stats:
                        print(f"      {metric_name}: n={stats['n']}, median={stats['median']:.3f}")
            if len(policy_data) > 5:
                print(f"    ... and {len(policy_data) - 5} more groups")
    
    # Post-hoc divergence analysis
    print("\n=== Post-Hoc Divergence Analysis ===")
    divergences = error_analysis(results)
    for div_type, cases in divergences.items():
        if cases:
            print(f"\n{div_type}: {len(cases)} cases")
            for case in cases[:5]:
                print(f"  {case}")
            if len(cases) > 5:
                print(f"  ... and {len(cases) - 5} more")
    
    # Save analysis
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    analysis_path = output_dir / "phase_f_analysis.json"
    with open(analysis_path, 'w') as f:
        # Convert non-serializable objects
        def convert(obj, seen=None):
            if seen is None:
                seen = set()
            obj_id = id(obj)
            if obj_id in seen:
                return "<circular>"
            seen.add(obj_id)
            
            if isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, SummaryStats):
                return asdict(obj)
            elif isinstance(obj, dict):
                return {k: convert(v, seen) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert(v, seen) for v in obj]
            elif isinstance(obj, set):
                return [convert(v, seen) for v in obj]
            elif hasattr(obj, '__dict__'):
                return {k: convert(v, seen) for k, v in obj.__dict__.items()}
            elif isinstance(obj, (str, int, float, bool, type(None))):
                return obj
            return str(obj)
        
        json.dump(analysis, f, indent=2, default=convert)
    print(f"\nAnalysis saved to {analysis_path}")
    
    # Generate plots
    if not args.no_plots and MATPLOTLIB_AVAILABLE:
        print("\nGenerating plots...")
        generate_plots(results, output_dir)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())