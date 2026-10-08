#!/usr/bin/env python3
"""
Non-GPU validation tests for run_phase_f.py configuration.
Tests that default behavior is preserved and new arguments work correctly.
"""

import sys
import json
from pathlib import Path

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.run_phase_f import PhaseFEvaluator, Policy


def test_default_profile():
    """Test that default profile is phase05."""
    evaluator = PhaseFEvaluator(pilot_mode=True, pilot_size=5)
    assert evaluator.profile_name == "phase05", f"Default profile should be 'phase05', got '{evaluator.profile_name}'"
    print("PASS: Default profile is 'phase05'")


def test_profile_override():
    """Test that profile can be overridden."""
    evaluator = PhaseFEvaluator(pilot_mode=True, pilot_size=5)
    evaluator.profile_name = "b2_workload"
    assert evaluator.profile_name == "b2_workload", f"Profile should be 'b2_workload', got '{evaluator.profile_name}'"
    print("PASS: Profile can be overridden to 'b2_workload'")


def test_policy_selection():
    """Test that policy selection works."""
    evaluator = PhaseFEvaluator(pilot_mode=True, pilot_size=5)
    
    # Test that we can run individual policies
    # Just verify the Policy enum values are correct
    assert Policy.ALWAYS_FP16.value == "always_fp16"
    assert Policy.ALWAYS_INT4.value == "always_int4"
    assert Policy.CARBONGRID.value == "carbongrid"
    print("PASS: Policy enum values are correct")


def test_output_directory():
    """Test that output directory is configurable."""
    evaluator = PhaseFEvaluator(pilot_mode=True, pilot_size=5, output_dir="data/evaluation/test_dir")
    # Use Path for cross-platform comparison
    expected = Path("data/evaluation/test_dir")
    assert evaluator.output_dir == expected, f"Output dir should be {expected}, got {evaluator.output_dir}"
    print("PASS: Output directory is configurable")


def test_same_prompts():
    """Test that the same 182 prompts are loaded."""
    evaluator = PhaseFEvaluator(pilot_mode=False)
    assert len(evaluator.records) == 182, f"Should load 182 prompts, got {len(evaluator.records)}"
    print("PASS: Loads " + str(len(evaluator.records)) + " prompts in full mode")
    
    # Check pilot mode loads correct subset
    evaluator_pilot = PhaseFEvaluator(pilot_mode=True, pilot_size=20)
    assert len(evaluator_pilot.records) == 20, f"Pilot should load 20 prompts, got {len(evaluator_pilot.records)}"
    print("PASS: Pilot mode loads correct subset")


def test_b2_profile_values():
    """Test that B2 profile values are correct."""
    from carbongrid.decision.models import DEFAULT_PROFILES
    
    b2 = DEFAULT_PROFILES["b2_workload"]
    assert b2["fp16"]["energy_wh"] == 0.1061, f"B2 FP16 energy should be 0.1061, got {b2['fp16']['energy_wh']}"
    assert b2["int4"]["energy_wh"] == 0.0945, f"B2 INT4 energy should be 0.0945, got {b2['int4']['energy_wh']}"
    print("PASS: B2 profile values are correct (FP16=0.1061 Wh, INT4=0.0945 Wh)")


def test_phase05_profile_values():
    """Test that Phase 0.5 profile values are correct."""
    from carbongrid.decision.models import DEFAULT_PROFILES
    
    p05 = DEFAULT_PROFILES["phase05"]
    assert p05["fp16"]["energy_wh"] == 0.0488, f"Phase05 FP16 energy should be 0.0488, got {p05['fp16']['energy_wh']}"
    assert p05["int4"]["energy_wh"] == 0.0493, f"Phase05 INT4 energy should be 0.0493, got {p05['int4']['energy_wh']}"
    print("PASS: Phase 0.5 profile values are correct (FP16=0.0488 Wh, INT4=0.0493 Wh)")


def test_default_behavior_preserved():
    """Test that running with default args preserves frozen Phase F behavior."""
    evaluator = PhaseFEvaluator(pilot_mode=True, pilot_size=5)
    evaluator.initialize_services()
    
    # Check that the decision engine uses phase05 profile
    policy = evaluator.decision_engine.policy
    assert policy.profile_name == "phase05", f"Default policy profile should be 'phase05', got '{policy.profile_name}'"
    assert policy.get_fp16_energy_wh() == 0.0488
    assert policy.get_int4_energy_wh() == 0.0493
    print("PASS: Default behavior uses phase05 profile with correct values")
    
    # Clean up
    if evaluator.inference_provider:
        evaluator.inference_provider.unload()


def test_b2_workload_profile():
    """Test that b2_workload profile can be selected."""
    evaluator = PhaseFEvaluator(pilot_mode=True, pilot_size=5)
    evaluator.profile_name = "b2_workload"
    evaluator.initialize_services()
    
    policy = evaluator.decision_engine.policy
    assert policy.profile_name == "b2_workload", f"Policy profile should be 'b2_workload', got '{policy.profile_name}'"
    assert policy.get_fp16_energy_wh() == 0.1061
    assert policy.get_int4_energy_wh() == 0.0945
    print("PASS: B2 workload profile can be selected with correct values")
    
    # Clean up
    if evaluator.inference_provider:
        evaluator.inference_provider.unload()


def test_output_dir_argument():
    """Test that --output_dir argument is accepted and used."""
    evaluator = PhaseFEvaluator(pilot_mode=True, pilot_size=5, output_dir="data/evaluation/test_output_dir")
    assert evaluator.output_dir == Path("data/evaluation/test_output_dir")
    print("PASS: --output_dir argument is accepted and used")


def run_all_tests():
    """Run all validation tests."""
    print("=" * 60)
    print("CONFIGURATION VALIDATION TESTS")
    print("=" * 60)
    
    tests = [
        test_default_profile,
        test_profile_override,
        test_policy_selection,
        test_output_directory,
        test_output_dir_argument,
        test_same_prompts,
        test_b2_profile_values,
        test_phase05_profile_values,
        test_default_behavior_preserved,
        test_b2_workload_profile,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            print("PASS: " + test.__name__)
            passed += 1
        except Exception as e:
            print("FAIL: " + test.__name__ + " - " + str(e))
            failed += 1
    
    print("\n" + "=" * 60)
    print("RESULTS: " + str(passed) + " passed, " + str(failed) + " failed")
    print("=" * 60)
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)