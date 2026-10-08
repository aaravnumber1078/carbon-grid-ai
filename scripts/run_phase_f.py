#!/usr/bin/env python3
"""
Phase F: Rigorous Experimental Evaluation of CarbonGrid-AI

Runs controlled evaluation of three policies:
1. Always FP16
2. Always INT4  
3. CarbonGrid (C.1.1 + Decision Engine)

Measures: latency, energy, CO2, quality, precision selection
"""

import sys
import json
import time
import torch
import gc
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict
from datetime import datetime, timezone
from enum import Enum

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from carbongrid.classifier.service import ClassifierService
from carbongrid.carbon.manager import create_carbon_manager, CarbonMode
from carbongrid.decision import (
    DecisionEngine,
    DecisionInput,
    DecisionPolicy,
    DecisionOutput,
    Precision as DecisionPrecision,
    DataSourceType as DecisionDataSourceType,
    DecisionReason,
    FallbackPolicy,
)
from carbongrid.inference.provider import (
    InferenceProvider,
    InferenceMetrics,
    Precision,
    create_inference_provider,
)
from carbongrid.evaluation.quality_v2 import (
    evaluate_pair_v2,
    QualityResult,
    get_evaluator_v2,
)


class Policy(Enum):
    ALWAYS_FP16 = "always_fp16"
    ALWAYS_INT4 = "always_int4"
    CARBONGRID = "carbongrid"


@dataclass
class ExperimentRecord:
    """Single experiment result record."""
    # Identifiers
    prompt_id: str
    task_type: str
    complexity_level: int
    b2_safe_to_quantize: bool
    b2_quality_score_int4: float
    b2_quality_difference: float
    
    # Experimental conditions
    policy: str
    carbon_intensity_gco2_per_kwh: float
    carbon_source: str
    carbon_zone: str = ""
    carbon_reading_type: str = ""
    carbon_data_source: str = ""
    latency_requirement_ms: Optional[float] = None
    
    # Decision info (for CarbonGrid)
    safety_probability: Optional[float] = None
    safety_threshold: Optional[float] = None
    safety_gate_passed: Optional[bool] = None
    decision_reason: Optional[str] = None
    selected_precision: Optional[str] = None
    fallback_used: Optional[bool] = None
    
    # Decision-time estimates (for CarbonGrid)
    fp16_estimated_energy_wh: Optional[float] = None
    int4_estimated_energy_wh: Optional[float] = None
    fp16_estimated_co2_g: Optional[float] = None
    int4_estimated_co2_g: Optional[float] = None
    fp16_estimated_latency_ms: Optional[float] = None
    int4_estimated_latency_ms: Optional[float] = None
    estimated_energy_wh: Optional[float] = None
    estimated_co2_g: Optional[float] = None
    estimated_co2_mg: Optional[float] = None
    estimated_latency_ms: Optional[float] = None
    
    # Energy/Carbon data source
    energy_data_source: str = ""
    carbon_data_source: str = ""
    carbon_reading_type: str = ""
    safety_gate_passed: Optional[bool] = None
    decision_policy_version: str = ""
    
    # Inference results
    latency_ms: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    tokens_per_second: float = 0.0
    peak_gpu_memory_mb: int = 0
    avg_power_w: Optional[float] = None
    energy_wh: Optional[float] = None
    co2_g: Optional[float] = None
    co2_mg: Optional[float] = None
    response: str = ""
    success: bool = False
    error: Optional[str] = None
    
    # Quality evaluation
    quality_score: Optional[float] = None
    quality_method: Optional[str] = None
    quality_passed: Optional[bool] = None
    quality_details: Optional[Dict] = None
    
    # Metadata
    measurement_source: str = "estimated"  # measured/estimated
    timestamp: str = ""
    gpu_name: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class PhaseFEvaluator:
    """Main evaluation orchestrator."""
    
    def __init__(
        self,
        dataset_path: str = "data/processed/b2_dataset.json",
        output_dir: str = "data/evaluation/phase_f",
        pilot_mode: bool = False,
        pilot_size: int = 20,
        latency_requirement_ms: Optional[float] = None,
        carbon_intensity_override: Optional[float] = None,
        carbon_mode: str = "offline",  # Use offline for reproducibility
    ):
        self.dataset_path = Path(dataset_path)
        self.output_dir = Path(output_dir)
        self.pilot_mode = pilot_mode
        self.pilot_size = pilot_size
        self.latency_requirement_ms = latency_requirement_ms
        self.carbon_intensity_override = carbon_intensity_override
        self.carbon_mode = carbon_mode
        self.profile_name = "phase05"  # Default profile
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Services (initialized lazily)
        self.classifier: Optional[ClassifierService] = None
        self.carbon_manager = None
        self.decision_engine: Optional[DecisionEngine] = None
        self.inference_provider: Optional[InferenceProvider] = None
        
        # Hardware info
        self.gpu_name = self._get_gpu_name()
        
        # Load dataset
        self.records = self._load_dataset()
        
    def _get_gpu_name(self) -> str:
        try:
            import subprocess
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5
            )
            return result.stdout.strip()
        except Exception:
            return "unknown"
    
    def _load_dataset(self) -> List[Dict]:
        with open(self.dataset_path, 'r') as f:
            records = json.load(f)
        
        if self.pilot_mode:
            # Stratified sampling across (task_type, complexity_level) using largest-remainder method
            # Deterministic: sort by (task_type, complexity_level, prompt_id) then allocate proportionally
            strata = defaultdict(list)
            for r in records:
                key = (r['task_type'], r['complexity_level'])
                strata[key].append(r)
            
            # Sort each stratum by prompt_id for deterministic selection
            for items in strata.values():
                items.sort(key=lambda x: x['prompt_id'])
            
            total_records = len(records)
            target_size = self.pilot_size
            
            # Proportional allocation with largest remainder
            allocations = {}
            remainders = {}
            
            for key, items in strata.items():
                exact = len(items) * target_size / total_records
                alloc = int(exact)
                allocations[key] = alloc
                remainders[key] = exact - alloc
            
            # Distribute remaining slots by largest remainder
            remaining = target_size - sum(allocations.values())
            if remaining > 0:
                sorted_by_remainder = sorted(remainders.items(), key=lambda x: x[1], reverse=True)
                for i in range(min(remaining, len(sorted_by_remainder))):
                    allocations[sorted_by_remainder[i][0]] += 1
            
            # Select from each stratum
            sampled = []
            for key, items in strata.items():
                alloc = allocations.get(key, 0)
                if alloc > 0:
                    sampled.extend(items[:alloc])
            
            # Sort final sample by (task_type, complexity_level, prompt_id) for consistent ordering
            sampled.sort(key=lambda x: (x['task_type'], x['complexity_level'], x['prompt_id']))
            records = sampled[:target_size]
            
            # Print selection summary
            print(f"Pilot selection ({len(records)} records):")
            dist = defaultdict(lambda: defaultdict(int))
            for r in records:
                dist[r['task_type']][r['complexity_level']] += 1
            for task in sorted(dist.keys()):
                for comp in sorted(dist[task].keys()):
                    print(f"  {task} complexity={comp}: {dist[task][comp]}")
        
        print(f"Loaded {len(records)} records for evaluation")
        return records
    
    def initialize_services(self):
        """Initialize all required services."""
        print("\n=== Initializing Services ===")
        
        # 1. Classifier
        print("[1/4] Loading C.1.1 Classifier...")
        self.classifier = ClassifierService().load()
        
        # 2. Carbon Manager (offline for reproducibility)
        print(f"[2/4] Initializing Carbon Manager ({self.carbon_mode})....")
        self.carbon_manager = create_carbon_manager(self.carbon_mode)
        
        # 3. Decision Engine
        print(f"[3/4] Initializing Decision Engine (profile: {self.profile_name})...")
        policy = DecisionPolicy(
            profile_name=self.profile_name,
            safety_threshold=self.classifier.get_threshold(),
        )
        self.decision_engine = DecisionEngine(policy)
        
        # 4. Inference Provider
        print("[4/4] Initializing Inference Provider...")
        self.inference_provider = create_inference_provider()
        
        print("\nAll services initialized successfully!")
    
    def _get_carbon_intensity(self) -> Tuple[float, str]:
        """Get current carbon intensity and source."""
        if self.carbon_intensity_override is not None:
            return self.carbon_intensity_override, DecisionDataSourceType.SIMULATED.value
        
        reading = self.carbon_manager.get_current_carbon("national")
        if reading:
            source_map = {
                "uk_carbon_intensity": DecisionDataSourceType.LIVE.value,
                "replay": DecisionDataSourceType.REPLAY.value,
                "offline": DecisionDataSourceType.OFFLINE.value,
            }
            source_type = source_map.get(reading.data_source.value, DecisionDataSourceType.UNKNOWN.value)
            return reading.carbon_intensity_gco2_per_kwh, source_type
        return 200.0, DecisionDataSourceType.OFFLINE.value
    
    def _run_always_fp16(self, prompt: str, max_new_tokens: int = 128) -> InferenceMetrics:
        """Run inference with FP16."""
        return self.inference_provider.run_inference(
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            precision=Precision.FP16,
        )
    
    def _run_always_int4(self, prompt: str, max_new_tokens: int = 128) -> InferenceMetrics:
        """Run inference with INT4."""
        return self.inference_provider.run_inference(
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            precision=Precision.INT4,
        )
    
    def _run_carbongrid(self, record: Dict, carbon_intensity: float, carbon_source: str) -> Tuple[InferenceMetrics, DecisionOutput]:
        """Run CarbonGrid decision + inference."""
        # Get safety probability
        safety_result = self.classifier.predict_safety(
            prompt=record["prompt"],
            task_type=record.get("task_type", "unknown"),
        )
        safety_prob = safety_result["safety_probability"]
        safety_threshold = safety_result["threshold"]
        
        # Build decision input
        profile = self.decision_engine.policy
        estimates = {
            "fp16_energy_wh": profile.get_fp16_energy_wh(),
            "int4_energy_wh": profile.get_int4_energy_wh(),
            "fp16_latency_ms": profile.get_fp16_latency_ms(),
            "int4_latency_ms": profile.get_int4_latency_ms(),
        }
        
        decision_input = DecisionInput(
            safety_probability=safety_prob,
            safety_threshold=safety_threshold,
            carbon_intensity_gco2_per_kwh=carbon_intensity,
            carbon_data_source=DecisionDataSourceType(carbon_source),
            carbon_zone="national",
            carbon_reading_type="actual",
            fp16_estimated_energy_wh=estimates["fp16_energy_wh"],
            int4_estimated_energy_wh=estimates["int4_energy_wh"],
            fp16_estimated_latency_ms=estimates["fp16_latency_ms"],
            int4_estimated_latency_ms=estimates["int4_latency_ms"],
            energy_data_source=DecisionDataSourceType.ESTIMATED,
            latency_data_source=DecisionDataSourceType.ESTIMATED,
            latency_requirement_ms=self.latency_requirement_ms,
        )
        
        decision = self.decision_engine.decide(decision_input)
        
        # Run inference at selected precision
        selected_precision = Precision(decision.selected_precision.value)
        inf_result = self.inference_provider.run_inference(
            prompt=record["prompt"],
            max_new_tokens=128,
            precision=selected_precision,
        )
        
        return inf_result, decision
    
    def _evaluate_quality(self, record: Dict, fp16_response: str, int4_response: str) -> QualityResult:
        """Evaluate quality using B2 evaluation methodology."""
        task_type = record.get("task_type", "unknown")
        reference = record.get("reference_answer", "")
        
        # Get FP16 response for this record (from B2 dataset)
        b2_fp16 = record.get("fp16_output", "")
        
        # Use B2's FP16 as reference, evaluate INT4 against it
        result = evaluate_pair_v2(
            fp16_output=b2_fp16,
            int4_output=int4_response,
            task_type=task_type,
            reference=reference,
        )
        return result
    
    def run_single(self, record: Dict, policy: Policy, carbon_intensity: float, carbon_source: str) -> ExperimentRecord:
        """Run a single experiment record."""
        base = ExperimentRecord(
            prompt_id=record["prompt_id"],
            task_type=record.get("task_type", "unknown"),
            complexity_level=record.get("complexity_level", 0),
            b2_safe_to_quantize=record.get("safe_to_quantize", False),
            b2_quality_score_int4=record.get("quality_score_int4", 0.0),
            b2_quality_difference=record.get("quality_difference", 0.0),
            policy=policy.value,
            carbon_intensity_gco2_per_kwh=carbon_intensity,
            carbon_source=carbon_source,
            latency_requirement_ms=self.latency_requirement_ms,
            timestamp=datetime.now(timezone.utc).isoformat(),
            gpu_name=self.gpu_name,
        )
        
        try:
            if policy == Policy.ALWAYS_FP16:
                inf_result = self._run_always_fp16(record["prompt"])
                base.selected_precision = "fp16"
                base.decision_reason = "fixed_policy"
                fp16_response = inf_result.response
                int4_response = None
                
            elif policy == Policy.ALWAYS_INT4:
                inf_result = self._run_always_int4(record["prompt"])
                base.selected_precision = "int4"
                base.decision_reason = "fixed_policy"
                fp16_response = None
                int4_response = inf_result.response
                
            elif policy == Policy.CARBONGRID:
                inf_result, decision = self._run_carbongrid(record, carbon_intensity, carbon_source)
                base.selected_precision = decision.selected_precision.value
                base.decision_reason = decision.decision_reason.value
                base.safety_probability = decision.safety_probability
                base.safety_threshold = decision.safety_threshold
                base.fallback_used = decision.fallback_used
                base.safety_gate_passed = decision.safety_gate_passed
                base.safety_probability = decision.safety_probability
                base.safety_threshold = decision.safety_threshold
                
                # Decision-time estimates
                base.fp16_estimated_energy_wh = decision.fp16_estimated_energy_wh
                base.int4_estimated_energy_wh = decision.int4_estimated_energy_wh
                base.fp16_estimated_co2_g = decision.fp16_estimated_co2_g
                base.int4_estimated_co2_g = decision.int4_estimated_co2_g
                base.fp16_estimated_latency_ms = decision.fp16_estimated_latency_ms
                base.int4_estimated_latency_ms = decision.int4_estimated_latency_ms
                base.estimated_energy_wh = decision.estimated_energy_wh
                base.estimated_co2_g = decision.estimated_co2_g
                base.estimated_co2_mg = decision.estimated_co2_mg
                base.estimated_latency_ms = decision.estimated_latency_ms
                
                # Carbon context
                base.carbon_zone = decision.carbon_zone
                base.carbon_reading_type = decision.carbon_reading_type
                base.carbon_data_source = decision.carbon_data_source.value
                base.energy_data_source = decision.energy_data_source.value
                base.carbon_reading_type = decision.carbon_reading_type
                base.safety_gate_passed = decision.safety_gate_passed
                base.decision_policy_version = decision.decision_policy_version
                
                # Get the response from the selected precision
                if decision.selected_precision == DecisionPrecision.FP16:
                    fp16_response = inf_result.response
                    int4_response = None
                else:
                    fp16_response = None
                    int4_response = inf_result.response
            
            # Fill inference metrics
            base.latency_ms = inf_result.latency_ms
            base.input_tokens = inf_result.input_tokens
            base.output_tokens = inf_result.output_tokens
            base.tokens_per_second = inf_result.tokens_per_second
            base.peak_gpu_memory_mb = inf_result.peak_gpu_memory_mb
            base.avg_power_w = inf_result.avg_power_w
            base.energy_wh = inf_result.energy_wh
            base.response = inf_result.response
            base.success = inf_result.success
            base.error = inf_result.error
            
            # CO2 calculation
            if inf_result.energy_wh is not None:
                base.energy_wh = inf_result.energy_wh
                base.co2_g = (inf_result.energy_wh / 1000.0) * carbon_intensity
                base.measurement_source = "measured" if inf_result.avg_power_w else "estimated"
            else:
                # Use Phase 0.5 baseline estimate
                profile = self.decision_engine.policy
                est_energy = profile.get_fp16_energy_wh() if base.selected_precision == "fp16" else profile.get_int4_energy_wh()
                base.energy_wh = est_energy
                base.co2_g = (est_energy / 1000.0) * carbon_intensity
                base.measurement_source = "phase05_profile_fallback"
            
            # Quality evaluation (only for INT4 selections, comparing to FP16 reference)
            if base.selected_precision == "int4" and int4_response:
                fp16_ref = record.get("fp16_output", "")
                quality_result = self._evaluate_quality(record, fp16_ref, int4_response)
                base.quality_score = quality_result.score
                base.quality_method = quality_result.method
                base.quality_passed = quality_result.passed
                base.quality_details = quality_result.details
            elif base.selected_precision == "fp16":
                # FP16 is reference, quality = 1.0 by definition
                base.quality_score = 1.0
                base.quality_method = "reference_convention"
                base.quality_passed = True
                base.quality_details = {
                    "note": "FP16 quality_score=1.0 is a reference convention, not an independently measured comparison against the B2 FP16 output."
                }
            
        except Exception as e:
            base.success = False
            base.error = str(e)
            import traceback
            traceback.print_exc()
        
        return base
    
    def run_policy(self, policy: Policy, records: List[Dict]) -> List[ExperimentRecord]:
        """Run all records under a single policy."""
        print(f"\n=== Running Policy: {policy.value} ===")
        results = []
        
        # Get carbon intensity once per policy run
        carbon_intensity, carbon_source = self._get_carbon_intensity()
        print(f"Carbon intensity: {carbon_intensity:.1f} gCO2/kWh ({carbon_source})")
        
        for i, record in enumerate(records):
            print(f"  [{i+1}/{len(records)}] {record['prompt_id']} ({record['task_type']})...", end=" ", flush=True)
            result = self.run_single(record, policy, carbon_intensity, carbon_source)
            results.append(result)
            status = "OK" if result.success else f"FAIL: {result.error}"
            energy_str = f"{result.energy_wh:.4f}Wh" if result.energy_wh is not None else "N/A"
            print(f"{status} | {result.selected_precision} | {result.latency_ms:.0f}ms | {energy_str}")
        
        return results
    
    def run_all_policies(self, records: List[Dict] = None) -> Dict[str, List[ExperimentRecord]]:
        """Run all three policies sequentially."""
        return self.run_selected_policies([Policy.ALWAYS_FP16, Policy.ALWAYS_INT4, Policy.CARBONGRID], records)
    
    def run_selected_policies(self, policies: List[Policy], records: List[Dict] = None) -> Dict[str, List[ExperimentRecord]]:
        """Run selected policies sequentially."""
        if records is None:
            records = self.records
        
        all_results = {}
        
        for policy in policies:
            # Unload previous model before next policy
            if self.inference_provider:
                self.inference_provider.unload()
            
            results = self.run_policy(policy, records)
            all_results[policy.value] = results
            
            # Save intermediate results
            self.save_results(all_results)
        
        return all_results
    
    def save_results(self, all_results: Dict[str, List[ExperimentRecord]]):
        """Save results to disk."""
        output = {
            "metadata": {
                "phase": "F",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "dataset": str(self.dataset_path),
                "pilot_mode": self.pilot_mode,
                "pilot_size": self.pilot_size if self.pilot_mode else None,
                "latency_requirement_ms": self.latency_requirement_ms,
                "carbon_intensity_override": self.carbon_intensity_override,
                "carbon_mode": self.carbon_mode,
                "gpu": self.gpu_name,
                "policies": list(all_results.keys()),
                "profile_name": self.profile_name,
                "sampling_method": "stratified_proportional_largest_remainder",
                "sampling_strata": ["task_type", "complexity_level"],
                "sampling_random_seed": 42,
                "energy_fallback_profile": "phase05",
                "quality_reference_convention": "FP16 quality_score=1.0 is a reference convention, not an independently measured comparison against the B2 FP16 output.",
            },
            "results": {
                policy: [r.to_dict() for r in records]
                for policy, records in all_results.items()
            }
        }
        
        output_path = self.output_dir / "phase_f_raw_results.json"
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2, default=str)
        print(f"\nSaved results to {output_path}")
    
    def run_sensitivity_analysis(self, records: List[Dict] = None) -> Dict[str, List[ExperimentRecord]]:
        """Run CarbonGrid under different carbon intensities."""
        if records is None:
            records = self.records
        
        print("\n=== Carbon Intensity Sensitivity Analysis ===")
        carbon_values = [50, 100, 150, 200, 300, 500]
        sensitivity_results = {}
        
        for intensity in carbon_values:
            print(f"\n--- Carbon intensity: {intensity} gCO2/kWh (simulated) ---")
            self.inference_provider.unload()
            
            results = self.run_policy(Policy.CARBONGRID, records)
            sensitivity_results[f"carbon_{intensity}"] = results
            self.save_results(sensitivity_results)
        
        return sensitivity_results
    
    def run_latency_constraint_analysis(self, records: List[Dict] = None) -> Dict[str, List[ExperimentRecord]]:
        """Run CarbonGrid under different latency constraints."""
        if records is None:
            records = self.records
        
        print("\n=== Latency Constraint Analysis ===")
        latency_values = [None, 6000, 8000, 12000, 25000]
        latency_results = {}
        
        original_latency = self.latency_requirement_ms
        
        for latency_ms in latency_values:
            print(f"\n--- Latency requirement: {latency_ms} ms ---")
            self.latency_requirement_ms = latency_ms
            self.inference_provider.unload()
            
            results = self.run_policy(Policy.CARBONGRID, records)
            key = f"latency_{latency_ms}" if latency_ms else "latency_none"
            latency_results[key] = results
            self.save_results(latency_results)
        
        self.latency_requirement_ms = original_latency
        return latency_results


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Phase F Experimental Evaluation")
    parser.add_argument("--pilot", action="store_true", help="Run pilot evaluation (20 prompts)")
    parser.add_argument("--pilot-size", type=int, default=20, help="Pilot size")
    parser.add_argument("--latency", type=float, help="Latency requirement in ms")
    parser.add_argument("--carbon", type=float, help="Override carbon intensity")
    parser.add_argument("--carbon-mode", default="offline", choices=["live", "replay", "offline"])
    parser.add_argument("--sensitivity", action="store_true", help="Run carbon sensitivity analysis")
    parser.add_argument("--latency-analysis", action="store_true", help="Run latency constraint analysis")
    parser.add_argument("--full", action="store_true", help="Run full evaluation (all prompts)")
    parser.add_argument("--profile", default="phase05", choices=["phase05", "b2_workload"], help="Energy/latency profile to use (default: phase05)")
    parser.add_argument("--policy", default="all", choices=["all", "always_fp16", "always_int4", "carbongrid"], help="Policy to run (default: all)")
    parser.add_argument("--output_dir", default="data/evaluation/phase_f", help="Output directory for results (default: data/evaluation/phase_f)")
    
    args = parser.parse_args()
    
    evaluator = PhaseFEvaluator(
        pilot_mode=args.pilot,
        pilot_size=args.pilot_size,
        latency_requirement_ms=args.latency,
        carbon_intensity_override=args.carbon,
        carbon_mode=args.carbon_mode,
        output_dir=args.output_dir,
    )
    
    # Override profile if specified
    if args.profile:
        evaluator.profile_name = args.profile
    
    evaluator.initialize_services()
    
    if args.sensitivity:
        evaluator.run_sensitivity_analysis()
    elif args.latency_analysis:
        evaluator.run_latency_constraint_analysis()
    elif args.full or args.pilot:
        # Determine which policies to run
        policies_to_run = []
        if args.policy == "all":
            policies_to_run = [Policy.ALWAYS_FP16, Policy.ALWAYS_INT4, Policy.CARBONGRID]
        elif args.policy == "always_fp16":
            policies_to_run = [Policy.ALWAYS_FP16]
        elif args.policy == "always_int4":
            policies_to_run = [Policy.ALWAYS_INT4]
        elif args.policy == "carbongrid":
            policies_to_run = [Policy.CARBONGRID]
        
        evaluator.run_selected_policies(policies_to_run)
    else:
        print("Use --pilot, --full, --sensitivity, or --latency-analysis")
        return 1
    
    return 0


if __name__ == "__main__":
    import subprocess
    sys.exit(main())