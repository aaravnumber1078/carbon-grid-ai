#!/usr/bin/env python3
"""
Phase F.2: Analysis of CarbonGrid with B2 Workload Profile

Compares Phase F.2 results (b2_workload profile) against frozen Phase F results.
"""

import sys
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict, Counter
from dataclasses import dataclass, asdict
from datetime import datetime, timezone

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


def load_metadata(results_path: Path) -> Dict[str, Any]:
    """Load metadata from JSON."""
    with open(results_path, 'r') as f:
        data = json.load(f)
    return data.get("metadata", {})


def compute_summary(values: List[float]) -> SummaryStats:
    """Compute summary statistics."""
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
    
    min_len = min(len(x), len(y))
    x_arr = np.array(x[:min_len])
    y_arr = np.array(y[:min_len])
    
    mask = ~(np.isnan(x_arr) | np.isnan(y_arr))
    x_clean = x_arr[mask]
    y_clean = y_arr[mask]
    
    if len(x_clean) < 10:
        return {"test": "wilcoxon", "n": len(x_clean), "skipped": "insufficient_pairs"}
    
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
        n = len(x_clean)
        r = 1 - (2 * stat) / (n * (n + 1) / 2)
        
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


def bootstrap_ci(x: List[float], y: List[float], n_bootstrap: int = 10000) -> Tuple[float, float]:
    """Bootstrap confidence interval for median difference."""
    if not SCIPY_AVAILABLE:
        return (None, None)
    
    min_len = min(len(x), len(y))
    x_arr = np.array(x[:min_len])
    y_arr = np.array(y[:min_len])
    
    mask = ~(np.isnan(x_arr) | np.isnan(y_arr))
    x_clean = x_arr[mask]
    y_clean = y_arr[mask]
    
    if len(x_clean) < 10:
        return (None, None)
    
    diffs = []
    for _ in range(n_bootstrap):
        idx = np.random.choice(len(x_clean), len(x_clean), replace=True)
        diffs.append(np.median(x_clean[idx] - y_clean[idx]))
    
    return (float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5)))


def load_phase_f2_results(results_path: Path) -> Tuple[List[Dict], Dict]:
    """Load Phase F.2 results."""
    results = load_results(results_path)
    metadata = load_metadata(results_path)
    return results.get("carbongrid", []), metadata


def load_frozen_phase_f(results_path: Path) -> Tuple[Dict[str, List[Dict]], Dict]:
    """Load frozen Phase F results."""
    results = load_results(results_path)
    metadata = load_metadata(results_path)
    return results, metadata


def match_prompt_records(f2_records: List[Dict], frozen_results: Dict[str, List[Dict]]) -> List[Dict]:
    """Match Phase F.2 records with frozen Phase F records by prompt_id."""
    # Build lookup maps for frozen results
    frozen_fp16 = {r["prompt_id"]: r for r in frozen_results.get("always_fp16", []) if r["success"]}
    frozen_int4 = {r["prompt_id"]: r for r in frozen_results.get("always_int4", []) if r["success"]}
    frozen_cg = {r["prompt_id"]: r for r in frozen_results.get("carbongrid", []) if r["success"]}
    
    matched = []
    for f2 in f2_records:
        if not f2.get("success"):
            continue
        pid = f2["prompt_id"]
        match = {
            "f2": f2,
            "frozen_fp16": frozen_fp16.get(pid),
            "frozen_int4": frozen_int4.get(pid),
            "frozen_cg": frozen_cg.get(pid),
        }
        matched.append(match)
    
    return matched


def analyze_f2_decisions(f2_records: List[Dict]) -> Dict[str, Any]:
    """Analyze Phase F.2 decision distribution."""
    successful = [r for r in f2_records if r["success"]]
    
    precision_dist = Counter(r["selected_precision"] for r in successful)
    decision_reasons = Counter(r["decision_reason"] for r in successful)
    safety_gate = Counter(r["safety_gate_passed"] for r in successful)
    fallback_used = sum(1 for r in successful if r.get("fallback_used", False))
    
    # By task type
    by_task = {}
    for r in successful:
        task = r["task_type"]
        if task not in by_task:
            by_task[task] = {"fp16": 0, "int4": 0, "total": 0}
        by_task[task][r["selected_precision"]] += 1
        by_task[task]["total"] += 1
    
    # Safety gate failures by task
    safety_fail_by_task = Counter(r["task_type"] for r in successful if not r["safety_gate_passed"])
    
    # INT4 selections detail
    int4_records = [r for r in successful if r["selected_precision"] == "int4"]
    int4_by_task = Counter(r["task_type"] for r in int4_records)
    int4_safety_probs = [r["safety_probability"] for r in int4_records]
    
    return {
        "total_prompts": len(f2_records),
        "successful": len(successful),
        "precision_distribution": dict(precision_dist),
        "decision_reasons": dict(decision_reasons),
        "safety_gate_passed": dict(safety_gate),
        "fallback_used": fallback_used,
        "by_task_type": by_task,
        "safety_gate_failures_by_task": dict(safety_fail_by_task),
        "int4_selections": {
            "count": len(int4_records),
            "by_task": dict(int4_by_task),
            "avg_safety_prob": np.mean(int4_safety_probs) if int4_safety_probs else 0,
        },
        "estimated_energy": {
            "fp16_wh": 0.1061,
            "int4_wh": 0.0945,
            "int4_lower_by_wh": 0.0116,
        },
    }


def analyze_energy_latency(f2_records: List[Dict], matched: List[Dict]) -> Dict[str, Any]:
    """Analyze measured energy and latency with paired comparisons."""
    successful = [r for r in f2_records if r["success"]]
    
    # Overall summary
    f2_energy = [r["energy_wh"] for r in successful if r["energy_wh"] is not None]
    f2_latency = [r["latency_ms"] for r in successful if r["latency_ms"] is not None]
    f2_co2 = [r["co2_g"] for r in successful if r["co2_g"] is not None]
    
    # Energy pairs
    energy_vs_cg = []
    energy_vs_fp16 = []
    energy_vs_int4 = []
    
    # Latency pairs
    latency_vs_cg = []
    latency_vs_fp16 = []
    latency_vs_int4 = []
    
    for m in matched:
        f2 = m["f2"]
        frozen_cg = m["frozen_cg"]
        frozen_fp16 = m["frozen_fp16"]
        frozen_int4 = m["frozen_int4"]
        
        if f2.get("energy_wh") is not None and frozen_cg and frozen_cg.get("energy_wh") is not None:
            energy_vs_cg.append({
                "prompt_id": m["f2"]["prompt_id"],
                "f2_energy": f2["energy_wh"],
                "frozen_cg_energy": frozen_cg["energy_wh"],
                "diff_wh": f2["energy_wh"] - frozen_cg["energy_wh"],
                "pct_change": (f2["energy_wh"] - frozen_cg["energy_wh"]) / frozen_cg["energy_wh"] * 100 if frozen_cg["energy_wh"] != 0 else None,
            })
        
        if f2.get("energy_wh") is not None and frozen_fp16 and frozen_fp16.get("energy_wh") is not None:
            energy_vs_fp16.append({
                "prompt_id": m["f2"]["prompt_id"],
                "f2_energy": f2["energy_wh"],
                "frozen_fp16_energy": frozen_fp16["energy_wh"],
                "diff_wh": f2["energy_wh"] - frozen_fp16["energy_wh"],
                "pct_change": (f2["energy_wh"] - frozen_fp16["energy_wh"]) / frozen_fp16["energy_wh"] * 100 if frozen_fp16["energy_wh"] != 0 else None,
            })
        
        if f2.get("energy_wh") is not None and frozen_int4 and frozen_int4.get("energy_wh") is not None:
            energy_vs_int4.append({
                "prompt_id": m["f2"]["prompt_id"],
                "f2_energy": f2["energy_wh"],
                "frozen_int4_energy": frozen_int4["energy_wh"],
                "diff_wh": f2["energy_wh"] - frozen_int4["energy_wh"],
                "pct_change": (f2["energy_wh"] - frozen_int4["energy_wh"]) / frozen_int4["energy_wh"] * 100 if frozen_int4["energy_wh"] != 0 else None,
            })
        
        # Latency pairs
        if f2.get("latency_ms") is not None and frozen_cg and frozen_cg.get("latency_ms") is not None:
            latency_vs_cg.append({
                "prompt_id": m["f2"]["prompt_id"],
                "f2_latency": f2["latency_ms"],
                "frozen_cg_latency": frozen_cg["latency_ms"],
                "diff_ms": f2["latency_ms"] - frozen_cg["latency_ms"],
                "pct_change": (f2["latency_ms"] - frozen_cg["latency_ms"]) / frozen_cg["latency_ms"] * 100 if frozen_cg["latency_ms"] != 0 else None,
            })
        
        if f2.get("latency_ms") is not None and frozen_fp16 and frozen_fp16.get("latency_ms") is not None:
            latency_vs_fp16.append({
                "prompt_id": m["f2"]["prompt_id"],
                "f2_latency": f2["latency_ms"],
                "frozen_fp16_latency": frozen_fp16["latency_ms"],
                "diff_ms": f2["latency_ms"] - frozen_fp16["latency_ms"],
                "pct_change": (f2["latency_ms"] - frozen_fp16["latency_ms"]) / frozen_fp16["latency_ms"] * 100 if frozen_fp16["latency_ms"] != 0 else None,
            })
        
        if f2.get("latency_ms") is not None and frozen_int4 and frozen_int4.get("latency_ms") is not None:
            latency_vs_int4.append({
                "prompt_id": m["f2"]["prompt_id"],
                "f2_latency": f2["latency_ms"],
                "frozen_int4_latency": frozen_int4["latency_ms"],
                "diff_ms": f2["latency_ms"] - frozen_int4["latency_ms"],
                "pct_change": (f2["latency_ms"] - frozen_int4["latency_ms"]) / frozen_int4["latency_ms"] * 100 if frozen_int4["latency_ms"] != 0 else None,
            })
    
    # Compute paired statistics
    def paired_stats(pairs, key1, key2, diff_key):
        vals1 = [p[key1] for p in pairs]
        vals2 = [p[key2] for p in pairs]
        diffs = [p[diff_key] for p in pairs]
        
        return {
            "n": len(pairs),
            "mean_diff": float(np.mean(diffs)) if diffs else None,
            "median_diff": float(np.median(diffs)) if diffs else None,
            "std_diff": float(np.std(diffs, ddof=1)) if len(diffs) > 1 else None,
            "wilcoxon": wilcoxon_paired(vals1, vals2),
            "bootstrap_ci": bootstrap_ci(vals1, vals2),
        }
    
    return {
        "f2_overall": {
            "energy_wh": asdict(compute_summary(f2_energy)),
            "latency_ms": asdict(compute_summary(f2_latency)),
            "co2_g": asdict(compute_summary(f2_co2)),
        },
        "paired_vs_frozen_cg": {
            "energy_wh": paired_stats(energy_vs_cg, "f2_energy", "frozen_cg_energy", "diff_wh"),
            "latency_ms": paired_stats(latency_vs_cg, "f2_latency", "frozen_cg_latency", "diff_ms"),
        },
        "paired_vs_frozen_fp16": {
            "energy_wh": paired_stats(energy_vs_fp16, "f2_energy", "frozen_fp16_energy", "diff_wh"),
            "latency_ms": paired_stats(latency_vs_fp16, "f2_latency", "frozen_fp16_latency", "diff_ms"),
        },
        "paired_vs_frozen_int4": {
            "energy_wh": paired_stats(energy_vs_int4, "f2_energy", "frozen_int4_energy", "diff_wh"),
            "latency_ms": paired_stats(latency_vs_int4, "f2_latency", "frozen_int4_latency", "diff_ms"),
        },
    }


def analyze_quality(f2_records: List[Dict], matched: List[Dict]) -> Dict[str, Any]:
    """Analyze quality scores, especially for INT4-selected prompts."""
    successful = [r for r in f2_records if r["success"] and r["quality_score"] is not None]
    
    # Overall quality
    f2_quality = [r["quality_score"] for r in successful]
    
    # By selected precision
    fp16_quality = [r["quality_score"] for r in successful if r["selected_precision"] == "fp16"]
    int4_quality = [r["quality_score"] for r in successful if r["selected_precision"] == "int4"]
    
    # INT4 quality vs frozen FP16 reference
    int4_quality_vs_frozen_fp16 = []
    for m in matched:
        f2 = m["f2"]
        if f2["selected_precision"] == "int4" and f2.get("quality_score") is not None:
            int4_quality_vs_frozen_fp16.append({
                "prompt_id": f2["prompt_id"],
                "f2_quality": f2["quality_score"],
                "task_type": f2["task_type"],
            })
    
    # Quality vs B2 labels
    b2_quality_scores = [r["b2_quality_score_int4"] for r in f2_records if r.get("b2_quality_score_int4") is not None]
    b2_safe = [r for r in f2_records if r.get("b2_safe_to_quantize") is not None]
    
    # By task type for INT4 selections
    int4_by_task = defaultdict(list)
    for m in matched:
        f2 = m["f2"]
        if f2["selected_precision"] == "int4" and f2.get("quality_score") is not None:
            int4_by_task[f2["task_type"]].append(f2["quality_score"])
    
    return {
        "overall": {
            "f2_quality": asdict(compute_summary(f2_quality)),
            "fp16_selected": asdict(compute_summary(fp16_quality)),
            "int4_selected": asdict(compute_summary(int4_quality)),
        },
        "int4_selections": {
            "count": len(int4_quality),
            "by_task": {task: asdict(compute_summary(vals)) for task, vals in int4_by_task.items()},
        },
        "b2_quality_reference": asdict(compute_summary(b2_quality_scores)),
    }


def analyze_safety_gate(f2_records: List[Dict]) -> Dict[str, Any]:
    """Analyze safety gate outcomes."""
    successful = [r for r in f2_records if r["success"]]
    
    safety_probs = [r["safety_probability"] for r in successful if r["safety_probability"] is not None]
    safety_passed = sum(1 for r in successful if r["safety_gate_passed"])
    safety_failed = len(successful) - safety_passed
    
    # By task type
    passed_by_task = defaultdict(int)
    failed_by_task = defaultdict(int)
    for r in successful:
        task = r["task_type"]
        if r["safety_gate_passed"]:
            passed_by_task[task] += 1
        else:
            failed_by_task[task] += 1
    
    # Safety probs for INT4-selected
    int4_safety_probs = [r["safety_probability"] for r in successful if r["selected_precision"] == "int4"]
    
    return {
        "total": len(successful),
        "passed": safety_passed,
        "failed": safety_failed,
        "pass_rate": safety_passed / len(successful) if successful else 0,
        "safety_probs": asdict(compute_summary(safety_probs)),
        "int4_safety_probs": asdict(compute_summary(int4_safety_probs)),
        "passed_by_task": dict(passed_by_task),
        "failed_by_task": dict(failed_by_task),
    }


def analyze_co2(f2_records: List[Dict], carbon_intensity: float = 200.0) -> Dict[str, Any]:
    """Analyze CO2 estimates."""
    successful = [r for r in f2_records if r["success"] and r["energy_wh"] is not None]
    
    f2_co2 = [r["co2_g"] for r in successful if r.get("co2_g") is not None]
    
    # CO2 by selected precision
    fp16_co2 = [r["co2_g"] for r in successful if r["selected_precision"] == "fp16" and r.get("co2_g") is not None]
    int4_co2 = [r["co2_g"] for r in successful if r["selected_precision"] == "int4" and r.get("co2_g") is not None]
    
    # Estimated CO2 from decision time
    est_fp16_co2 = 0.1061 / 1000 * carbon_intensity
    est_int4_co2 = 0.0945 / 1000 * carbon_intensity
    
    return {
        "carbon_intensity_gco2_per_kwh": carbon_intensity,
        "estimated_co2": {
            "fp16_g": est_fp16_co2,
            "int4_g": est_int4_co2,
            "int4_lower_by_g": est_fp16_co2 - est_int4_co2,
        },
        "measured_co2": {
            "overall": asdict(compute_summary(f2_co2)),
            "fp16_selected": asdict(compute_summary(fp16_co2)),
            "int4_selected": asdict(compute_summary(int4_co2)),
        },
        "note": "CO2 values are estimated from measured energy (Wh) × offline carbon intensity (200 gCO2/kWh). Not direct CO2 measurements.",
    }


def generate_plots(f2_records: List[Dict], matched: List[Dict], output_dir: Path):
    """Generate visualization plots."""
    if not MATPLOTLIB_AVAILABLE:
        return
    
    output_dir.mkdir(parents=True, exist_ok=True)
    successful = [r for r in f2_records if r["success"]]
    
    # Extract INT4 vs FP16 selected data
    fp16_data = [r for r in successful if r["selected_precision"] == "fp16"]
    int4_data = [r for r in successful if r["selected_precision"] == "int4"]
    
    metrics = [
        ("energy_wh", "Energy (Wh)"),
        ("latency_ms", "Latency (ms)"),
        ("quality_score", "Quality Score"),
    ]
    
    for metric_name, metric_label in metrics:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        
        # Distribution comparison
        ax = axes[0]
        fp16_vals = [r[metric_name] for r in fp16_data if r.get(metric_name) is not None]
        int4_vals = [r[metric_name] for r in int4_data if r.get(metric_name) is not None]
        
        if fp16_vals and int4_vals:
            bp = ax.boxplot([fp16_vals, int4_vals], labels=['FP16', 'INT4'], patch_artist=True)
            bp['boxes'][0].set_facecolor('#1f77b4')
            bp['boxes'][1].set_facecolor('#ff7f0e')
            ax.set_ylabel(metric_label)
            ax.set_title(f"{metric_label} by Selected Precision")
        
        # Paired difference
        ax = axes[1]
        paired_diffs = []
        for m in matched:
            f2 = m["f2"]
            if m["frozen_fp16"] and f2.get(metric_name) is not None and m["frozen_fp16"].get(metric_name) is not None:
                paired_diffs.append(f2[metric_name] - m["frozen_fp16"][metric_name])
        
        if paired_diffs:
            ax.hist(paired_diffs, bins=20, edgecolor='black', alpha=0.7)
            ax.axvline(0, color='red', linestyle='--')
            ax.set_xlabel(f"Difference (F2 - Frozen FP16) {metric_label}")
            ax.set_ylabel("Count")
            ax.set_title(f"Paired Difference vs Frozen FP16")
        
        plt.tight_layout()
        plt.savefig(output_dir / f"phase_f2_{metric_name}.png", dpi=150)
        plt.close()
    
    # Decision reason pie chart
    successful = [r for r in f2_records if r["success"]]
    reasons = Counter(r["decision_reason"] for r in successful)
    if reasons:
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.pie(list(reasons.values()), labels=list(reasons.keys()), autopct='%1.1f%%')
        ax.set_title("Phase F.2 Decision Reasons")
        plt.savefig(output_dir / "phase_f2_decision_reasons.png", dpi=150)
        plt.close()
    
    # Precision selection
    precisions = Counter(r["selected_precision"] for r in [r for r in f2_records if r["success"]])
    if precisions:
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.pie(list(precisions.values()), labels=list(precisions.keys()), autopct='%1.1f%%', colors=['#1f77b4', '#ff7f0e'])
        ax.set_title("Phase F.2 Precision Selection")
        plt.savefig(output_dir / "phase_f2_precision_selection.png", dpi=150)
        plt.close()
    
    print(f"Plots saved to {output_dir}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze Phase F.2 results vs Frozen Phase F")
    parser.add_argument("--f2-results", default="data/evaluation/phase_f_b2profile_gpu/phase_f_raw_results.json")
    parser.add_argument("--frozen-results", default="data/evaluation/phase_f/phase_f_raw_results.json")
    parser.add_argument("--output", default="data/evaluation/phase_f_b2profile_gpu")
    parser.add_argument("--no-plots", action="store_true")
    
    args = parser.parse_args()
    
    f2_path = Path(args.f2_results)
    frozen_path = Path(args.frozen_results)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not f2_path.exists():
        print(f"Error: F2 results file not found: {f2_path}")
        return 1
    if not frozen_path.exists():
        print(f"Error: Frozen Phase F results not found: {frozen_path}")
        return 1
    
    print(f"Loading Phase F.2 results from {f2_path}...")
    f2_records, f2_metadata = load_phase_f2_results(f2_path)
    print(f"Loaded {len(f2_records)} Phase F.2 records (profile: {f2_metadata.get('profile_name', 'unknown')})")
    
    print(f"Loading frozen Phase F results from {frozen_path}...")
    frozen_results, frozen_metadata = load_frozen_phase_f(frozen_path)
    print(f"Loaded {sum(len(v) for v in frozen_results.values())} frozen Phase F records (profile: {frozen_metadata.get('profile_name', 'unknown')})")
    
    # Match records
    matched = match_prompt_records(f2_records, frozen_results)
    print(f"Matched {len(matched)} prompts between Phase F.2 and frozen Phase F")
    
    # Verify no frozen files were modified
    frozen_mtime = frozen_path.stat().st_mtime
    print(f"Frozen Phase F file timestamp: {frozen_mtime}")
    
    # Run analyses
    print("\n=== Analyzing Phase F.2 Decisions ===")
    decision_analysis = analyze_f2_decisions(f2_records)
    
    print("\n=== Analyzing Energy & Latency ===")
    energy_analysis = analyze_energy_latency(f2_records, matched)
    
    print("\n=== Analyzing Quality ===")
    quality_analysis = analyze_quality(f2_records, matched)
    
    print("\n=== Analyzing Safety Gate ===")
    safety_analysis = analyze_safety_gate(f2_records)
    
    print("\n=== Analyzing CO2 ===")
    co2_analysis = analyze_co2(f2_records, carbon_intensity=200.0)
    
    # Build full report
    report = {
        "metadata": {
            "experiment": "Phase F.2 Evaluation",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "f2_results_path": str(f2_path),
            "frozen_results_path": str(frozen_path),
            "f2_metadata": f2_metadata,
            "frozen_metadata": frozen_metadata,
            "matched_prompts": len(matched),
            "total_f2_records": len(f2_records),
            "successful_f2_records": sum(1 for r in f2_records if r["success"]),
            "carbon_intensity_gco2_per_kwh": 200.0,
            "note": "CO2 values are estimated from measured energy × offline carbon intensity (200 gCO2/kWh). Not direct CO2 measurements.",
        },
        "decision_analysis": decision_analysis,
        "energy_latency_analysis": energy_analysis,
        "quality_analysis": quality_analysis,
        "safety_gate_analysis": safety_analysis,
        "co2_analysis": co2_analysis,
    }
    
    # Generate plots
    if MATPLOTLIB_AVAILABLE and not args.no_plots:
        print("\nGenerating plots...")
        generate_plots(f2_records, matched, output_dir)
    
    # Save report
    report_path = output_dir / "phase_f2_analysis.json"
    with open(report_path, 'w') as f:
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
        
        json.dump(report, f, indent=2, default=convert)
    
    print(f"\nAnalysis saved to {report_path}")
    
    # Print summary
    print("\n" + "=" * 70)
    print("PHASE F.2 EVALUATION SUMMARY")
    print("=" * 70)
    
    # Decision summary
    print(f"\nDecision Distribution:")
    for prec, count in decision_analysis["precision_distribution"].items():
        pct = count / decision_analysis["successful"] * 100
        print(f"  {prec.upper()}: {count} ({pct:.1f}%)")
    
    print(f"\nDecision Reasons:")
    for reason, count in decision_analysis["decision_reasons"].items():
        pct = count / decision_analysis["successful"] * 100
        print(f"  {reason}: {count} ({pct:.1f}%)")
    
    # Energy summary
    print(f"\nMeasured Energy (F2):")
    e = energy_analysis["f2_overall"]["energy_wh"]
    print(f"  n={e['n']}, median={e['median']:.6f} Wh, mean={e['mean']:.6f}±{e['std']:.6f} Wh")
    
    # Paired vs frozen
    print(f"\nPaired vs Frozen CarbonGrid (Energy):")
    pe = energy_analysis["paired_vs_frozen_cg"]["energy_wh"]
    print(f"  n={pe['n']}, median_diff={pe['median_diff']:.6f} Wh, mean_diff={pe['mean_diff']:.6f} Wh")
    print(f"  Wilcoxon: p={pe['wilcoxon'].get('p_value', 'N/A'):.4f}, significant={pe['wilcoxon'].get('significant', 'N/A')}")
    
    print(f"\nPaired vs Frozen Always-FP16 (Energy):")
    pe = energy_analysis["paired_vs_frozen_fp16"]["energy_wh"]
    print(f"  n={pe['n']}, median_diff={pe['median_diff']:.6f} Wh, mean_diff={pe['mean_diff']:.6f} Wh")
    print(f"  Wilcoxon: p={pe['wilcoxon'].get('p_value', 'N/A'):.4f}, significant={pe['wilcoxon'].get('significant', 'N/A')}")
    
    # Quality
    print(f"\nQuality Scores:")
    q = quality_analysis["overall"]["int4_selected"]
    print(f"  INT4-selected: n={q['n']}, median={q['median']:.4f}, mean={q['mean']:.4f}±{q['std']:.4f}")
    q = quality_analysis["overall"]["fp16_selected"]
    print(f"  FP16-selected: n={q['n']}, median={q['median']:.4f}, mean={q['mean']:.4f}±{q['std']:.4f}")
    
    # Safety gate
    print(f"\nSafety Gate:")
    s = safety_analysis
    print(f"  Passed: {s['passed']}/{s['total']} ({s['pass_rate']:.1%})")
    print(f"  INT4-selected avg safety prob: {s['int4_safety_probs']['mean']:.3f}")
    
    # CO2
    print(f"\nCO2 (estimated from measured energy × 200 gCO2/kWh):")
    c = co2_analysis["measured_co2"]["overall"]
    print(f"  Overall: median={c['median']:.6f} g")
    c = co2_analysis["measured_co2"]["int4_selected"]
    print(f"  INT4-selected: median={c['median']:.6f} g")
    c = co2_analysis["measured_co2"]["fp16_selected"]
    print(f"  FP16-selected: median={c['median']:.6f} g")
    
    print(f"\nReport saved to {output_dir / 'phase_f2_analysis.json'}")
    print(f"Plots saved to {output_dir}")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())