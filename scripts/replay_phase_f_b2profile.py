#!/usr/bin/env python3
"""
Phase F.2 Preflight: CPU-only Decision Replay with B2 Workload Profile

Replays CarbonGrid decision policy on the same 182 Phase F prompts
using the B2 workload-level static profile (INT4 lower energy than FP16).
No GPU inference or benchmarks.
"""

import sys
import json
from pathlib import Path
from typing import Dict, List, Any, Tuple
from collections import Counter

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from carbongrid.classifier.service import ClassifierService
from carbongrid.decision import DecisionEngine, DecisionInput, DecisionPolicy, Precision, DataSourceType, DecisionReason


# ============================================================
# CONFIGURATION
# ============================================================

B2_DATASET_PATH = Path("C:/CARBON GRID AI/data/processed/b2_dataset.json")
OUTPUT_DIR = Path("C:/CARBON GRID AI/data/evaluation/phase_f_b2profile")
REPORT_PATH = OUTPUT_DIR / "replay_report.json"

# B2 workload profile values (from DEFAULT_PROFILES)
B2_PROFILE = {
    "fp16": {"energy_wh": 0.1061, "latency_ms": 11640},
    "int4": {"energy_wh": 0.0945, "latency_ms": 22077},
}

# Phase 0.5 profile for comparison
PHASE05_PROFILE = {
    "fp16": {"energy_wh": 0.0488, "latency_ms": 5867},
    "int4": {"energy_wh": 0.0493, "latency_ms": 10580},
}

CARBON_INTENSITY = 200.0  # gCO2/kWh (offline fallback)


# ============================================================
# DATA LOADING
# ============================================================

def load_b2_dataset() -> List[Dict[str, Any]]:
    """Load the 182 B2 prompts used in Phase F."""
    with open(B2_DATASET_PATH) as f:
        data = json.load(f)
    
    # Filter to complete measurements only (same as Phase F)
    records = []
    for r in data:
        if r.get("measurement_status") == "complete":
            records.append({
                "prompt_id": r["prompt_id"],
                "prompt": r["prompt"],
                "task_type": r.get("task_type", "unknown"),
                "complexity_level": r.get("complexity_level", 1),
                "b2_safe_to_quantize": r.get("safe_to_quantize", False),
                "b2_quality_score_int4": r.get("quality_score_int4", 0.0),
            })
    
    print(f"Loaded {len(records)} prompts from B2 dataset")
    return records


def load_phase_f_results() -> Dict[str, Any]:
    """Load Phase F raw results for comparison."""
    with open("C:/CARBON GRID AI/data/evaluation/phase_f/phase_f_raw_results.json") as f:
        return json.load(f)


# ============================================================
# DECISION REPLAY
# ============================================================

def run_decision_replay(records: List[Dict], classifier: ClassifierService) -> List[Dict]:
    """Run CarbonGrid decision for each prompt with B2 profile."""
    
    # Create policy with B2 workload profile
    policy = DecisionPolicy(
        profile_name="b2_workload",  # This will use the B2 profile from DEFAULT_PROFILES
        safety_threshold=classifier.get_threshold(),
    )
    
    engine = DecisionEngine(policy)
    
    results = []
    
    for i, record in enumerate(records):
        prompt = record["prompt"]
        prompt_id = record["prompt_id"]
        task_type = record["task_type"]
        
        # Get safety probability from classifier
        safety_result = classifier.predict_safety(prompt, task_type)
        safety_prob = safety_result["safety_probability"]
        safety_threshold = safety_result["threshold"]
        
        # Build decision input with B2 profile estimates
        decision_input = DecisionInput(
            safety_probability=safety_prob,
            safety_threshold=safety_threshold,
            carbon_intensity_gco2_per_kwh=CARBON_INTENSITY,
            carbon_data_source=DataSourceType.OFFLINE,
            carbon_zone="national",
            carbon_reading_type="actual",
            fp16_estimated_energy_wh=B2_PROFILE["fp16"]["energy_wh"],
            int4_estimated_energy_wh=B2_PROFILE["int4"]["energy_wh"],
            fp16_estimated_latency_ms=B2_PROFILE["fp16"]["latency_ms"],
            int4_estimated_latency_ms=B2_PROFILE["int4"]["latency_ms"],
            energy_data_source=DataSourceType.ESTIMATED,
            latency_data_source=DataSourceType.ESTIMATED,
            latency_requirement_ms=None,
        )
        
        # Run decision engine
        decision = engine.decide(decision_input)
        
        # Extract decision details
        result = {
            "prompt_id": prompt_id,
            "task_type": task_type,
            "b2_safe_to_quantize": record["b2_safe_to_quantize"],
            "safety_probability": safety_prob,
            "safety_threshold": safety_threshold,
            "safety_gate_passed": decision.safety_gate_passed,
            "selected_precision": decision.selected_precision.value,
            "decision_reason": decision.decision_reason.value,
            "fallback_used": decision.fallback_used,
            "fp16_energy_wh": decision.fp16_estimated_energy_wh,
            "int4_energy_wh": decision.int4_estimated_energy_wh,
            "fp16_co2_g": decision.fp16_estimated_co2_g,
            "int4_co2_g": decision.int4_estimated_co2_g,
            "fp16_latency_ms": decision.fp16_estimated_latency_ms,
            "int4_latency_ms": decision.int4_estimated_latency_ms,
            "co2_diff_mg": abs(decision.fp16_estimated_co2_g - decision.int4_estimated_co2_g) * 1000,
            "co2_threshold_mg": policy.co2_difference_threshold_mg,
        }
        
        results.append(result)
        
        if (i + 1) % 20 == 0:
            print(f"  Processed {i + 1}/{len(records)} prompts")
    
    return results


def analyze_results(results: List[Dict]) -> Dict[str, Any]:
    """Analyze decision distribution and compare with Phase F."""
    
    decisions = Counter(r["decision_reason"] for r in results)
    precisions = Counter(r["selected_precision"] for r in results)
    safety_gate = Counter(r["safety_gate_passed"] for r in results)
    
    # Per-task-type breakdown
    by_task = {}
    for r in results:
        task = r["task_type"]
        if task not in by_task:
            by_task[task] = {"fp16": 0, "int4": 0, "total": 0}
        by_task[task][r["selected_precision"]] += 1
        by_task[task]["total"] += 1
    
    # Safety gate failures by task type
    safety_fail_by_task = {}
    for r in results:
        if not r["safety_gate_passed"]:
            task = r["task_type"]
            safety_fail_by_task[task] = safety_fail_by_task.get(task, 0) + 1
    
    # INT4 selections detail
    int4_results = [r for r in results if r["selected_precision"] == "int4"]
    int4_by_task = Counter(r["task_type"] for r in int4_results)
    int4_safety = [r["safety_probability"] for r in int4_results]
    
    return {
        "total_prompts": len(results),
        "precision_distribution": dict(precisions),
        "decision_reasons": dict(decisions),
        "safety_gate_passed": dict(safety_gate),
        "by_task_type": by_task,
        "safety_gate_failures_by_task": safety_fail_by_task,
        "int4_selections": {
            "count": len(int4_results),
            "by_task": dict(int4_by_task),
            "safety_probabilities": int4_safety,
            "avg_safety_prob": sum(int4_safety) / len(int4_safety) if int4_safety else 0,
        },
        "energy_comparison": {
            "fp16_energy_wh": B2_PROFILE["fp16"]["energy_wh"],
            "int4_energy_wh": B2_PROFILE["int4"]["energy_wh"],
            "int4_lower_by_wh": B2_PROFILE["fp16"]["energy_wh"] - B2_PROFILE["int4"]["energy_wh"],
        },
    }


def compare_with_phase05(records: List[Dict], classifier: ClassifierService) -> Dict[str, Any]:
    """Also run with Phase 0.5 profile for direct comparison."""
    
    policy_05 = DecisionPolicy(
        profile_name="phase05",
        safety_threshold=classifier.get_threshold(),
    )
    engine_05 = DecisionEngine(policy_05)
    
    results_05 = []
    
    for record in records:
        prompt = record["prompt"]
        task_type = record["task_type"]
        prompt_id = record["prompt_id"]
        
        safety_result = classifier.predict_safety(prompt, task_type)
        safety_prob = safety_result["safety_probability"]
        safety_threshold = safety_result["threshold"]
        
        decision_input = DecisionInput(
            safety_probability=safety_prob,
            safety_threshold=safety_threshold,
            carbon_intensity_gco2_per_kwh=CARBON_INTENSITY,
            carbon_data_source=DataSourceType.OFFLINE,
            carbon_zone="national",
            carbon_reading_type="actual",
            fp16_estimated_energy_wh=PHASE05_PROFILE["fp16"]["energy_wh"],
            int4_estimated_energy_wh=PHASE05_PROFILE["int4"]["energy_wh"],
            fp16_estimated_latency_ms=PHASE05_PROFILE["fp16"]["latency_ms"],
            int4_estimated_latency_ms=PHASE05_PROFILE["int4"]["latency_ms"],
            energy_data_source=DataSourceType.ESTIMATED,
            latency_data_source=DataSourceType.ESTIMATED,
            latency_requirement_ms=None,
        )
        
        decision = engine_05.decide(decision_input)
        
        results_05.append({
            "prompt_id": prompt_id,
            "selected_precision": decision.selected_precision.value,
            "decision_reason": decision.decision_reason.value,
            "safety_gate_passed": decision.safety_gate_passed,
        })
    
    precisions_05 = Counter(r["selected_precision"] for r in results_05)
    decisions_05 = Counter(r["decision_reason"] for r in results_05)
    safety_05 = Counter(r["safety_gate_passed"] for r in results_05)
    
    return {
        "precision_distribution": dict(precisions_05),
        "decision_reasons": dict(decisions_05),
        "safety_gate_passed": dict(safety_05),
    }


def main():
    print("=" * 70)
    print("PHASE F.2 PREFLIGHT: CPU-ONLY DECISION REPLAY WITH B2 PROFILE")
    print("=" * 70)
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Load data
    records = load_b2_dataset()
    phase_f_results = load_phase_f_results()
    
    # Load classifier
    print("\n[1/3] Loading C.1.1 Classifier...")
    classifier = ClassifierService().load()
    
    # Run replay with B2 profile
    print("\n[2/3] Running decision replay with B2 workload profile...")
    print(f"  B2 Profile: FP16={B2_PROFILE['fp16']['energy_wh']:.4f} Wh, INT4={B2_PROFILE['int4']['energy_wh']:.4f} Wh")
    print(f"  INT4 lower by: {B2_PROFILE['fp16']['energy_wh'] - B2_PROFILE['int4']['energy_wh']:.4f} Wh")
    
    replay_results = run_decision_replay(records, classifier)
    
    # Run comparison with Phase 0.5 profile
    print("\n[3/3] Running comparison with Phase 0.5 profile...")
    phase05_results = compare_with_phase05(records, classifier)
    
    # Analyze
    analysis = analyze_results(replay_results)
    
    # Compile report
    report = {
        "metadata": {
            "experiment": "Phase F.2 Preflight Replay",
            "dataset": "B2 (182 prompts)",
            "profile_used": "b2_workload",
            "fp16_energy_wh": B2_PROFILE["fp16"]["energy_wh"],
            "int4_energy_wh": B2_PROFILE["int4"]["energy_wh"],
            "carbon_intensity_gco2_per_kwh": CARBON_INTENSITY,
            "safety_threshold": classifier.get_threshold(),
            "co2_difference_threshold_mg": 0.1,
        },
        "b2_profile_replay": analysis,
        "phase05_profile_replay": phase05_results,
    }
    
    # Save report
    with open(REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2, default=str)
    
    print(f"\nReport saved: {REPORT_PATH}")
    
    # Print summary
    print(f"\n{'=' * 70}")
    print("REPLAY RESULTS SUMMARY")
    print(f"{'=' * 70}")
    
    print(f"\nTotal prompts: {analysis['total_prompts']}")
    print(f"\nPrecision Distribution:")
    for prec, count in analysis["precision_distribution"].items():
        pct = count / analysis['total_prompts'] * 100
        print(f"  {prec.upper()}: {count} ({pct:.1f}%)")
    
    print(f"\nDecision Reasons:")
    for reason, count in analysis["decision_reasons"].items():
        pct = count / analysis['total_prompts'] * 100
        print(f"  {reason}: {count} ({pct:.1f}%)")
    
    print(f"\nSafety Gate:")
    for passed, count in analysis["safety_gate_passed"].items():
        pct = count / analysis['total_prompts'] * 100
        print(f"  Passed: {passed} -> {count} ({pct:.1f}%)")
    
    print(f"\nINT4 Selections: {analysis['int4_selections']['count']}")
    if analysis['int4_selections']['count'] > 0:
        print(f"  By task: {analysis['int4_selections']['by_task']}")
        print(f"  Avg safety prob: {analysis['int4_selections']['avg_safety_prob']:.3f}")
        print(f"  Safety probs: {[f'{p:.3f}' for p in analysis['int4_selections']['safety_probabilities']]}")
    
    print(f"\nEnergy Comparison (B2 Profile):")
    ec = analysis["energy_comparison"]
    print(f"  FP16: {ec['fp16_energy_wh']:.4f} Wh")
    print(f"  INT4: {ec['int4_energy_wh']:.4f} Wh")
    print(f"  INT4 lower by: {ec['int4_lower_by_wh']:.4f} Wh")
    
    print(f"\n--- Phase 0.5 Profile Comparison ---")
    print(f"Precision Distribution: {phase05_results['precision_distribution']}")
    print(f"Decision Reasons: {phase05_results['decision_reasons']}")
    
    # Verdict
    print(f"\n{'=' * 70}")
    print("PREFLIGHT VERDICT")
    print(f"{'=' * 70}")
    
    if analysis['int4_selections']['count'] > 0:
        print(f"✓ B2 profile selects INT4 for {analysis['int4_selections']['count']}/182 prompts")
        print(f"  → 182-run GPU experiment IS justified to measure actual outcomes")
    else:
        print(f"✗ B2 profile selects FP16 for all 182 prompts")
        print(f"  → GPU experiment would NOT answer the intended question")
        print(f"  → Next step: Analyze why (safety gate, CO2 threshold, latency, etc.)")
    
    return report


if __name__ == "__main__":
    main()