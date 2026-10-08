#!/usr/bin/env python3
"""
Phase D Demo - CarbonGrid Decision Engine Demonstration.

Shows the Decision Engine in action with various scenarios:
- Scenario A: Unsafe request -> FP16
- Scenario B: Safe + INT4 environmentally preferable + latency feasible -> INT4
- Scenario C: Safe + INT4 latency violation -> FP16
- Scenario D: Carbon provider fallback -> decision still works

Each scenario demonstrates the full decision flow with transparent reasoning.
"""

import sys
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from carbongrid.decision import (
    DecisionEngine,
    DecisionInput,
    DecisionPolicy,
    Precision,
    DataSourceType,
    DecisionReason,
    FallbackPolicy,
    estimate_co2_from_energy,
    DEFAULT_PROFILES,
)


def print_scenario_header(name: str):
    print(f"\n{'='*70}")
    print(f"SCENARIO {name}")
    print(f"{'='*70}")


def print_decision_details(decision, scenario_name: str):
    """Print detailed decision information."""
    print(f"\n--- Decision Result ---")
    print(f"Selected Precision:  {decision.selected_precision.value.upper()}")
    print(f"Decision Reason:     {decision.decision_reason.value}")
    print(f"Safety Gate Passed:  {decision.safety_gate_passed}")
    print(f"Fallback Used:       {decision.fallback_used}")
    print(f"")
    print(f"--- Safety Context ---")
    print(f"Safety Probability:  {decision.safety_probability:.3f}")
    print(f"Safety Threshold:    {decision.safety_threshold:.2f}")
    print(f"")
    print(f"--- Carbon Context ---")
    print(f"Carbon Intensity:    {decision.carbon_intensity_gco2_per_kwh:.1f} gCO2/kWh")
    print(f"Carbon Source:       {decision.carbon_data_source.value}")
    print(f"Carbon Zone:         {decision.carbon_zone}")
    print(f"")
    print(f"--- Energy/CO2 Estimates ---")
    print(f"FP16 Energy:         {decision.fp16_estimated_energy_wh:.4f} Wh")
    print(f"FP16 CO2:            {decision.fp16_estimated_co2_g:.6f} g ({decision.fp16_estimated_co2_g:.3f} mg)")
    print(f"INT4 Energy:         {decision.int4_estimated_energy_wh:.4f} Wh")
    print(f"INT4 CO2:            {decision.int4_estimated_co2_g:.6f} g ({decision.int4_estimated_co2_g:.3f} mg)")
    print(f"CO2 Difference:      {abs(decision.fp16_estimated_co2_g - decision.int4_estimated_co2_g):.3f} mg")
    print(f"")
    print(f"--- Selected Config Estimates ---")
    print(f"Energy:              {decision.estimated_energy_wh:.4f} Wh")
    print(f"CO2:                 {decision.estimated_co2_g:.6f} g ({decision.estimated_co2_mg:.3f} mg)")
    print(f"Latency:             {decision.estimated_latency_ms:.0f} ms")
    print(f"")
    print(f"--- Latency Constraints ---")
    print(f"Requirement:         {decision.latency_requirement_ms if decision.latency_requirement_ms else 'Not specified'} ms")
    print(f"Status:              {decision.latency_constraint_status}")
    print(f"")
    print(f"--- Metadata ---")
    print(f"Policy Version:      {decision.decision_policy_version}")
    print(f"Energy Source:       {decision.energy_data_source.value}")
    if decision.metadata:
        print(f"CO2 Diff Threshold:  {decision.metadata.get('co2_diff_threshold_mg', 'N/A')} mg")
        print(f"Profile Used:        {decision.metadata.get('profile_used', 'N/A')}")


def run_scenario_a():
    """Scenario A: Unsafe request -> FP16 (safety gate)."""
    print_scenario_header("A: Unsafe Request -> FP16 (Safety Gate)")
    
    print("""
Context:
- Classifier predicts low safety probability (0.30)
- Threshold is 0.50
- Carbon intensity is LOW (100 gCO2/kWh) - would favor INT4 if safe
- Both configs meet latency requirements
""")
    
    engine = DecisionEngine(DecisionPolicy(profile_name="phase05"))
    
    input_data = DecisionInput(
        safety_probability=0.30,
        safety_threshold=0.50,
        carbon_intensity_gco2_per_kwh=100.0,
        carbon_data_source=DataSourceType.LIVE,
        fp16_estimated_energy_wh=0.0488,
        int4_estimated_energy_wh=0.0493,
        fp16_estimated_latency_ms=5867,
        int4_estimated_latency_ms=10580,
        energy_data_source=DataSourceType.MEASURED,
    )
    
    decision = engine.decide(input_data)
    print_decision_details(decision, "A")
    
    assert decision.selected_precision == Precision.FP16
    assert decision.decision_reason == DecisionReason.SAFETY_GATE_FAILED
    assert decision.safety_gate_passed is False
    print("\n✓ PASS: Safety gate correctly rejects INT4 despite low carbon intensity")


def run_scenario_b():
    """Scenario B: Safe + INT4 environmentally preferable + latency feasible -> INT4."""
    print_scenario_header("B: Safe + INT4 Lower Environmental Cost -> INT4")
    
    print("""
Context:
- Classifier predicts high safety probability (0.85)
- Threshold is 0.50
- Using B2 workload profile where INT4 uses LESS energy (0.0945 vs 0.1061 Wh)
- Carbon intensity is moderate (200 gCO2/kWh)
- No latency requirement
""")
    
    engine = DecisionEngine(DecisionPolicy(profile_name="b2_workload"))
    
    input_data = DecisionInput(
        safety_probability=0.85,
        safety_threshold=0.50,
        carbon_intensity_gco2_per_kwh=200.0,
        carbon_data_source=DataSourceType.LIVE,
        fp16_estimated_energy_wh=0.1061,
        int4_estimated_energy_wh=0.0945,
        fp16_estimated_latency_ms=11640,
        int4_estimated_latency_ms=22077,
        energy_data_source=DataSourceType.MEASURED,
    )
    
    decision = engine.decide(input_data)
    print_decision_details(decision, "B")
    
    assert decision.selected_precision == Precision.INT4
    assert decision.decision_reason == DecisionReason.INT4_LOWER_ENVIRONMENTAL_COST
    assert decision.safety_gate_passed is True
    assert decision.estimated_co2_g < decision.fp16_estimated_co2_g
    print("\n✓ PASS: INT4 selected due to lower environmental cost (measured B2 workload)")


def run_scenario_c():
    """Scenario C: Safe + INT4 latency violation -> FP16."""
    print_scenario_header("C: Safe + INT4 Latency Violation -> FP16")
    
    print("""
Context:
- Classifier predicts high safety probability (0.80)
- Threshold is 0.50
- Request has strict latency requirement: 8000 ms
- FP16 latency: 5867 ms (OK)
- INT4 latency: 10580 ms (VIOLATES)
- Carbon intensity: 200 gCO2/kWh
""")
    
    engine = DecisionEngine(DecisionPolicy(profile_name="phase05"))
    
    input_data = DecisionInput(
        safety_probability=0.80,
        safety_threshold=0.50,
        carbon_intensity_gco2_per_kwh=200.0,
        carbon_data_source=DataSourceType.LIVE,
        fp16_estimated_energy_wh=0.0488,
        int4_estimated_energy_wh=0.0493,
        fp16_estimated_latency_ms=5867,
        int4_estimated_latency_ms=10580,
        latency_requirement_ms=8000,
        energy_data_source=DataSourceType.MEASURED,
    )
    
    decision = engine.decide(input_data)
    print_decision_details(decision, "C")
    
    assert decision.selected_precision == Precision.FP16
    assert decision.decision_reason == DecisionReason.INT4_LATENCY_VIOLATION
    assert decision.latency_constraint_status == "int4_violated"
    assert decision.latency_requirement_ms == 8000
    print("\n✓ PASS: FP16 selected because INT4 violates latency SLA")


def run_scenario_d():
    """Scenario D: Carbon provider fallback -> decision still works."""
    print_scenario_header("D: Carbon Provider Fallback (Offline) -> Decision Works")
    
    print("""
Context:
- Live carbon API unavailable
- Using offline fallback (200 gCO2/kWh default)
- Classifier predicts safe (0.75)
- Using Phase 0.5 profile (FP16 slightly lower energy)
- No latency requirement
""")
    
    engine = DecisionEngine(DecisionPolicy(profile_name="phase05"))
    
    input_data = DecisionInput(
        safety_probability=0.75,
        safety_threshold=0.50,
        carbon_intensity_gco2_per_kwh=200.0,  # Offline default
        carbon_data_source=DataSourceType.OFFLINE,
        fp16_estimated_energy_wh=0.0488,
        int4_estimated_energy_wh=0.0493,
        fp16_estimated_latency_ms=5867,
        int4_estimated_latency_ms=10580,
        energy_data_source=DataSourceType.MEASURED,
    )
    
    decision = engine.decide(input_data)
    print_decision_details(decision, "D")
    
    assert decision.carbon_data_source == DataSourceType.OFFLINE
    assert decision.carbon_intensity_gco2_per_kwh == 200.0
    assert decision.selected_precision == Precision.FP16  # Phase 0.5: FP16 lower energy
    assert decision.decision_reason == DecisionReason.FP16_LOWER_ENVIRONMENTAL_COST
    print("\n✓ PASS: Decision works with offline carbon fallback")


def run_scenario_e():
    """Scenario E: Equal environmental cost -> tie-break by latency."""
    print_scenario_header("E: Equal Environmental Cost -> Tie-break by Latency")
    
    print("""
Context:
- Safety probability: 0.80 (safe)
- Both configs have SAME energy estimate (0.05 Wh)
- FP16 latency: 5000 ms
- INT4 latency: 8000 ms
- CO2 difference threshold: 10 mg (large, so considered equal)
""")
    
    policy = DecisionPolicy(
        co2_difference_threshold_mg=10.0,
        fp16_energy_wh_override=0.05,
        int4_energy_wh_override=0.05,
        fp16_latency_ms_override=5000,
        int4_latency_ms_override=8000,
    )
    engine = DecisionEngine(policy)
    
    input_data = DecisionInput(
        safety_probability=0.80,
        safety_threshold=0.50,
        carbon_intensity_gco2_per_kwh=200.0,
        carbon_data_source=DataSourceType.LIVE,
        fp16_estimated_energy_wh=0.05,
        int4_estimated_energy_wh=0.05,
        fp16_estimated_latency_ms=5000,
        int4_estimated_latency_ms=8000,
    )
    
    decision = engine.decide(input_data)
    print_decision_details(decision, "E")
    
    assert decision.decision_reason == DecisionReason.ENVIRONMENTAL_COST_EQUAL
    assert decision.selected_precision == Precision.FP16  # Lower latency wins
    print("\n✓ PASS: Tie-break correctly prefers lower latency (FP16)")


def run_scenario_f():
    """Scenario F: Both configurations violate latency -> fallback policy."""
    print_scenario_header("F: Both Violate Latency -> Fallback Policy")
    
    print("""
Context:
- Safety probability: 0.85 (safe)
- Latency requirement: 3000 ms (very strict)
- FP16: 5867 ms (VIOLATES)
- INT4: 10580 ms (VIOLATES)
- Fallback policy: PREFER_FP16 (default)
""")
    
    engine = DecisionEngine(DecisionPolicy(profile_name="phase05"))
    
    input_data = DecisionInput(
        safety_probability=0.85,
        safety_threshold=0.50,
        carbon_intensity_gco2_per_kwh=200.0,
        carbon_data_source=DataSourceType.LIVE,
        fp16_estimated_energy_wh=0.0488,
        int4_estimated_energy_wh=0.0493,
        fp16_estimated_latency_ms=5867,
        int4_estimated_latency_ms=10580,
        latency_requirement_ms=3000,
    )
    
    decision = engine.decide(input_data)
    print_decision_details(decision, "F")
    
    assert decision.latency_constraint_status == "both_violated"
    assert decision.fallback_used is True
    assert decision.decision_reason == DecisionReason.FALLBACK_POLICY
    assert decision.selected_precision == Precision.FP16
    print("\n✓ PASS: Fallback policy correctly applied when both violate SLA")


def run_scenario_g():
    """Scenario G: Resource constraint (VRAM) blocks FP16."""
    print_scenario_header("G: Resource Constraint (VRAM) -> INT4")
    
    print("""
Context:
- Safety probability: 0.80 (safe)
- Available VRAM: 2000 MB
- FP16 requires: 3200 MB (BLOCKED)
- INT4 requires: 1400 MB (OK)
- No latency requirement
""")
    
    policy = DecisionPolicy(
        min_vram_mb_for_fp16=3200,
        min_vram_mb_for_int4=1400,
    )
    engine = DecisionEngine(policy)
    
    input_data = DecisionInput(
        safety_probability=0.80,
        safety_threshold=0.50,
        carbon_intensity_gco2_per_kwh=200.0,
        carbon_data_source=DataSourceType.LIVE,
        fp16_estimated_energy_wh=0.0488,
        int4_estimated_energy_wh=0.0493,
        fp16_estimated_latency_ms=5867,
        int4_estimated_latency_ms=10580,
        gpu_vram_free_mb=2000,
        resource_state="constrained",
    )
    
    decision = engine.decide(input_data)
    print_decision_details(decision, "G")
    
    assert decision.selected_precision == Precision.INT4
    assert decision.decision_reason == DecisionReason.RESOURCE_CONSTRAINT
    assert decision.gpu_vram_free_mb == 2000
    print("\n✓ PASS: Resource constraint correctly routes to INT4")


def run_scenario_h():
    """Scenario H: Replay carbon data source."""
    print_scenario_header("H: Replay Carbon Data -> Decision Works")
    
    print("""
Context:
- Using replay carbon data (historical)
- Carbon intensity from replay: 180 gCO2/kWh
- Safety probability: 0.90
- B2 workload profile (INT4 lower energy)
""")
    
    engine = DecisionEngine(DecisionPolicy(profile_name="b2_workload"))
    
    input_data = DecisionInput(
        safety_probability=0.90,
        safety_threshold=0.50,
        carbon_intensity_gco2_per_kwh=180.0,
        carbon_data_source=DataSourceType.REPLAY,
        fp16_estimated_energy_wh=0.1061,
        int4_estimated_energy_wh=0.0945,
        fp16_estimated_latency_ms=11640,
        int4_estimated_latency_ms=22077,
    )
    
    decision = engine.decide(input_data)
    print_decision_details(decision, "H")
    
    assert decision.carbon_data_source == DataSourceType.REPLAY
    assert decision.carbon_intensity_gco2_per_kwh == 180.0
    assert decision.selected_precision == Precision.INT4
    print("\n✓ PASS: Replay carbon data correctly used in decision")


def run_scenario_i():
    """Scenario I: Simulated carbon data (explicitly labeled)."""
    print_scenario_header("I: Simulated Carbon Data -> Explicitly Labeled")
    
    print("""
Context:
- Using simulated regional carbon data
- Carbon intensity: 50 gCO2/kWh (simulated low-carbon region)
- Explicitly labeled as SIMULATED
- Safety probability: 0.70
""")
    
    engine = DecisionEngine(DecisionPolicy(profile_name="phase05"))
    
    input_data = DecisionInput(
        safety_probability=0.70,
        safety_threshold=0.50,
        carbon_intensity_gco2_per_kwh=50.0,
        carbon_data_source=DataSourceType.SIMULATED,
        fp16_estimated_energy_wh=0.0488,
        int4_estimated_energy_wh=0.0493,
        fp16_estimated_latency_ms=5867,
        int4_estimated_latency_ms=10580,
    )
    
    decision = engine.decide(input_data)
    print_decision_details(decision, "I")
    
    assert decision.carbon_data_source == DataSourceType.SIMULATED
    assert decision.carbon_intensity_gco2_per_kwh == 50.0
    # At very low carbon, difference is tiny but FP16 still slightly lower energy
    assert decision.selected_precision == Precision.FP16
    print("\n✓ PASS: Simulated data explicitly labeled and used correctly")


def print_summary():
    """Print summary of all scenarios."""
    print(f"\n{'='*70}")
    print("PHASE D DEMO SUMMARY")
    print(f"{'='*70}")
    print("""
All scenarios demonstrate the Decision Engine correctly handling:

A. SAFETY GATE: Unsafe classifier output -> FP16 (hard constraint)
B. ENVIRONMENTAL COST: INT4 lower energy (B2 workload) -> INT4
C. LATENCY CONSTRAINT: INT4 violates SLA -> FP16
D. OFFLINE FALLBACK: Offline carbon data -> decision works
E. TIE-BREAK: Equal CO2 -> lower latency wins (FP16)
F. BOTH VIOLATE: Both violate latency -> fallback policy (FP16)
G. RESOURCE CONSTRAINT: Low VRAM blocks FP16 -> INT4
H. REPLAY DATA: Historical carbon data -> decision works
I. SIMULATED DATA: Explicitly labeled -> decision works

KEY PRINCIPLES DEMONSTRATED:
✓ Safety is a HARD GATE - cannot be bypassed by carbon intensity
✓ Latency constraints are RESPECTED when specified
✓ No artificial SLA created when not specified
✓ Environmental cost comparison is EXPLICIT (not weighted score)
✓ CO2 = energy × carbon_intensity (correct units)
✓ Data sources are TRANSPARENT (live/replay/offline/simulated/measured/estimated)
✓ Fallback policies are CONFIGURABLE
✓ All decisions are EXPLAINABLE with full reasoning
✓ Output is fully JSON-serializable for telemetry
""")


def main():
    print("=" * 70)
    print("CARBONGRID-AI PHASE D: DECISION ENGINE DEMONSTRATION")
    print("=" * 70)
    print("\nThis demo shows the Decision Engine with deterministic inputs.")
    print("All carbon/energy values are from MEASURED Phase 0.5 and B2 workloads.")
    print("Classifier probabilities are SIMULATED for demonstration.")
    
    # Run all scenarios
    run_scenario_a()
    run_scenario_b()
    run_scenario_c()
    run_scenario_d()
    run_scenario_e()
    run_scenario_f()
    run_scenario_g()
    run_scenario_h()
    run_scenario_i()
    
    # Print summary
    print_summary()
    
    print("\n" + "=" * 70)
    print("DEMO COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()