"""
Unit tests for CarbonGrid Decision Engine.

Tests cover all decision scenarios from the Phase D specification.
"""

import pytest
from datetime import datetime, timezone

from carbongrid.decision import (
    DecisionEngine,
    DecisionInput,
    DecisionOutput,
    DecisionPolicy,
    Precision,
    DataSourceType,
    DecisionReason,
    FallbackPolicy,
    estimate_co2_from_energy,
    DEFAULT_POLICY,
    DEFAULT_PROFILES,
    create_policy,
)


class TestCO2Calculation:
    """Test CO2 calculation utilities."""
    
    def test_estimate_co2_basic(self):
        """Test basic CO2 calculation."""
        energy_wh = 0.05  # 50 mWh
        carbon_intensity = 200  # gCO2/kWh
        co2_g, co2_mg = estimate_co2_from_energy(energy_wh, carbon_intensity)
        
        # 0.05 Wh = 0.00005 kWh
        # 0.00005 * 200 = 0.01 g = 10 mg
        assert co2_g == 0.01
        assert co2_mg == 10.0
    
    def test_estimate_co2_dimensional_sanity(self):
        """TEST: 1 Wh at 200 gCO2/kWh must produce 0.2 gCO2 (0.001 kWh * 200)."""
        energy_wh = 1.0
        carbon_intensity = 200
        co2_g, co2_mg = estimate_co2_from_energy(energy_wh, carbon_intensity)
        
        assert co2_g == 0.2
        assert co2_mg == 200.0
    
    def test_estimate_co2_zero_energy(self):
        """Test CO2 calculation with zero energy."""
        co2_g, co2_mg = estimate_co2_from_energy(0.0, 200)
        assert co2_g == 0.0
        assert co2_mg == 0.0
    
    def test_estimate_co2_zero_intensity(self):
        """Test CO2 calculation with zero carbon intensity."""
        co2_g, co2_mg = estimate_co2_from_energy(0.1, 0.0)
        assert co2_g == 0.0
        assert co2_mg == 0.0
    
    def test_estimate_co2_high_intensity(self):
        """Test CO2 calculation with high carbon intensity."""
        energy_wh = 0.1
        carbon_intensity = 500
        co2_g, co2_mg = estimate_co2_from_energy(energy_wh, carbon_intensity)
        
        # 0.1 Wh = 0.0001 kWh
        # 0.0001 * 500 = 0.05 g = 50 mg
        assert co2_g == 0.05
        assert co2_mg == 50.0
    
    def test_estimate_co2_very_small_energy(self):
        """Test CO2 calculation with very small positive energy."""
        energy_wh = 1e-6  # 1 µWh
        carbon_intensity = 200
        co2_g, co2_mg = estimate_co2_from_energy(energy_wh, carbon_intensity)
        
        # 1e-6 Wh = 1e-9 kWh
        # 1e-9 * 200 = 2e-7 g = 0.0002 mg
        assert abs(co2_g - 2e-7) < 1e-15
        assert abs(co2_mg - 0.0002) < 1e-12
    
    def test_estimate_co2_large_values(self):
        """Test CO2 calculation with large but finite values."""
        energy_wh = 1000.0  # 1 kWh
        carbon_intensity = 1000.0
        co2_g, co2_mg = estimate_co2_from_energy(energy_wh, carbon_intensity)
        
        # 1000 Wh = 1 kWh
        # 1 * 1000 = 1000 g = 1,000,000 mg
        assert co2_g == 1000.0
        assert co2_mg == 1_000_000.0
    
    def test_estimate_co2_negative_energy_raises(self):
        """Negative energy should raise ValueError."""
        with pytest.raises(ValueError, match="energy_wh cannot be negative"):
            estimate_co2_from_energy(-0.1, 200)
    
    def test_estimate_co2_negative_intensity_raises(self):
        """Negative carbon intensity should raise ValueError."""
        with pytest.raises(ValueError, match="carbon_intensity_gco2_per_kwh cannot be negative"):
            estimate_co2_from_energy(0.1, -100)
    
    def test_estimate_co2_nan_energy_raises(self):
        """NaN energy should raise ValueError."""
        with pytest.raises(ValueError, match="energy_wh cannot be NaN"):
            estimate_co2_from_energy(float('nan'), 200)
    
    def test_estimate_co2_nan_intensity_raises(self):
        """NaN carbon intensity should raise ValueError."""
        with pytest.raises(ValueError, match="carbon_intensity_gco2_per_kwh cannot be NaN"):
            estimate_co2_from_energy(0.1, float('nan'))
    
    def test_estimate_co2_inf_energy_raises(self):
        """Infinite energy should raise ValueError."""
        with pytest.raises(ValueError, match="energy_wh cannot be infinite"):
            estimate_co2_from_energy(float('inf'), 200)
        with pytest.raises(ValueError, match="energy_wh cannot be infinite"):
            estimate_co2_from_energy(float('-inf'), 200)
    
    def test_estimate_co2_inf_intensity_raises(self):
        """Infinite carbon intensity should raise ValueError."""
        with pytest.raises(ValueError, match="carbon_intensity_gco2_per_kwh cannot be infinite"):
            estimate_co2_from_energy(0.1, float('inf'))
        with pytest.raises(ValueError, match="carbon_intensity_gco2_per_kwh cannot be infinite"):
            estimate_co2_from_energy(0.1, float('-inf'))
    
    def test_estimate_co2_none_energy_raises(self):
        """None energy should raise ValueError."""
        with pytest.raises(ValueError, match="energy_wh cannot be None"):
            estimate_co2_from_energy(None, 200)
    
    def test_estimate_co2_none_intensity_raises(self):
        """None carbon intensity should raise ValueError."""
        with pytest.raises(ValueError, match="carbon_intensity_gco2_per_kwh cannot be None"):
            estimate_co2_from_energy(0.1, None)
    
    def test_estimate_co2_string_energy_raises(self):
        """String energy should raise ValueError."""
        with pytest.raises(ValueError, match="energy_wh must be numeric"):
            estimate_co2_from_energy("0.1", 200)
    
    def test_estimate_co2_string_intensity_raises(self):
        """String carbon intensity should raise ValueError."""
        with pytest.raises(ValueError, match="carbon_intensity_gco2_per_kwh must be numeric"):
            estimate_co2_from_energy(0.1, "200")


class TestDecisionEngineSafetyGate:
    """Test safety gate behavior (TEST 1, 13)."""
    
    def setup_method(self):
        self.engine = DecisionEngine(DEFAULT_POLICY)
    
    def test_unsafe_classifier_probability_returns_fp16(self):
        """TEST 1: Unsafe classifier probability -> FP16."""
        input_data = DecisionInput(
            safety_probability=0.30,  # Below threshold 0.50
            safety_threshold=0.50,
            carbon_intensity_gco2_per_kwh=100,  # Low carbon - should favor INT4 if safe
            fp16_estimated_energy_wh=0.0488,
            int4_estimated_energy_wh=0.0493,
            fp16_estimated_latency_ms=5867,
            int4_estimated_latency_ms=10580,
            carbon_data_source=DataSourceType.LIVE,
        )
        
        decision = self.engine.decide(input_data)
        
        assert decision.selected_precision == Precision.FP16
        assert decision.decision_reason == DecisionReason.SAFETY_GATE_FAILED
        assert decision.safety_gate_passed is False
        assert decision.fallback_used is False
    
    def test_safety_gate_cannot_be_bypassed_by_high_carbon(self):
        """TEST 13: Safety gate cannot be bypassed by high carbon intensity."""
        input_data = DecisionInput(
            safety_probability=0.30,  # Below threshold
            safety_threshold=0.50,
            carbon_intensity_gco2_per_kwh=500,  # Very high carbon
            fp16_estimated_energy_wh=0.0488,
            int4_estimated_energy_wh=0.0493,
            fp16_estimated_latency_ms=5867,
            int4_estimated_latency_ms=10580,
            carbon_data_source=DataSourceType.LIVE,
        )
        
        decision = self.engine.decide(input_data)
        
        assert decision.selected_precision == Precision.FP16
        assert decision.decision_reason == DecisionReason.SAFETY_GATE_FAILED
        assert decision.safety_gate_passed is False
    
    def test_safe_classifier_probability_passes_gate(self):
        """Safe probability passes safety gate."""
        input_data = DecisionInput(
            safety_probability=0.80,  # Above threshold
            safety_threshold=0.50,
            carbon_intensity_gco2_per_kwh=200,
            fp16_estimated_energy_wh=0.0488,
            int4_estimated_energy_wh=0.0493,
            fp16_estimated_latency_ms=5867,
            int4_estimated_latency_ms=10580,
            carbon_data_source=DataSourceType.LIVE,
        )
        
        decision = self.engine.decide(input_data)
        
        assert decision.safety_gate_passed is True


class TestDecisionEngineFailClosed:
    """Test fail-closed behavior for invalid/missing classifier output."""

    def setup_method(self):
        self.engine = DecisionEngine(DEFAULT_POLICY)

    def test_nan_safety_probability_fails_closed(self):
        """NaN safety probability -> FP16 with CLASSIFIER_ERROR."""
        input_data = DecisionInput(
            safety_probability=float('nan'),
            safety_threshold=0.50,
            carbon_intensity_gco2_per_kwh=200,
            fp16_estimated_energy_wh=0.0488,
            int4_estimated_energy_wh=0.0493,
            fp16_estimated_latency_ms=5867,
            int4_estimated_latency_ms=10580,
            carbon_data_source=DataSourceType.LIVE,
        )
        
        decision = self.engine.decide(input_data)
        
        assert decision.selected_precision == Precision.FP16
        assert decision.decision_reason == DecisionReason.CLASSIFIER_ERROR
        assert decision.safety_gate_passed is False
        assert decision.fallback_used is False

    def test_infinite_safety_probability_fails_closed(self):
        """Infinite safety probability -> FP16 with CLASSIFIER_ERROR."""
        input_data = DecisionInput(
            safety_probability=float('inf'),
            safety_threshold=0.50,
            carbon_intensity_gco2_per_kwh=200,
            fp16_estimated_energy_wh=0.0488,
            int4_estimated_energy_wh=0.0493,
            fp16_estimated_latency_ms=5867,
            int4_estimated_latency_ms=10580,
            carbon_data_source=DataSourceType.LIVE,
        )
        
        decision = self.engine.decide(input_data)
        
        assert decision.selected_precision == Precision.FP16
        assert decision.decision_reason == DecisionReason.CLASSIFIER_ERROR
        assert decision.safety_gate_passed is False

    def test_negative_safety_probability_fails_closed(self):
        """Negative safety probability -> FP16 with CLASSIFIER_ERROR."""
        input_data = DecisionInput(
            safety_probability=-0.1,
            safety_threshold=0.50,
            carbon_intensity_gco2_per_kwh=200,
            fp16_estimated_energy_wh=0.0488,
            int4_estimated_energy_wh=0.0493,
            fp16_estimated_latency_ms=5867,
            int4_estimated_latency_ms=10580,
            carbon_data_source=DataSourceType.LIVE,
        )
        
        decision = self.engine.decide(input_data)
        
        assert decision.selected_precision == Precision.FP16
        assert decision.decision_reason == DecisionReason.CLASSIFIER_ERROR
        assert decision.safety_gate_passed is False

    def test_out_of_range_safety_probability_fails_closed(self):
        """Safety probability > 1.0 -> FP16 with CLASSIFIER_ERROR."""
        input_data = DecisionInput(
            safety_probability=1.5,
            safety_threshold=0.50,
            carbon_intensity_gco2_per_kwh=200,
            fp16_estimated_energy_wh=0.0488,
            int4_estimated_energy_wh=0.0493,
            fp16_estimated_latency_ms=5867,
            int4_estimated_latency_ms=10580,
            carbon_data_source=DataSourceType.LIVE,
        )
        
        decision = self.engine.decide(input_data)
        
        assert decision.selected_precision == Precision.FP16
        assert decision.decision_reason == DecisionReason.CLASSIFIER_ERROR
        assert decision.safety_gate_passed is False

    def test_none_safety_probability_fails_closed(self):
        """None safety probability -> FP16 with CLASSIFIER_ERROR."""
        input_data = DecisionInput(
            safety_probability=None,
            safety_threshold=0.50,
            carbon_intensity_gco2_per_kwh=200,
            fp16_estimated_energy_wh=0.0488,
            int4_estimated_energy_wh=0.0493,
            fp16_estimated_latency_ms=5867,
            int4_estimated_latency_ms=10580,
            carbon_data_source=DataSourceType.LIVE,
        )
        
        decision = self.engine.decide(input_data)
        
        assert decision.selected_precision == Precision.FP16
        assert decision.decision_reason == DecisionReason.CLASSIFIER_ERROR
        assert decision.safety_gate_passed is False


class TestDecisionEngineLatency:
    """Test latency constraint handling (TEST 3, 7)."""
    
    def setup_method(self):
        self.engine = DecisionEngine(DEFAULT_POLICY)
    
    def test_int4_violates_latency_fp16_satisfies(self):
        """TEST 3: INT4 violates latency, FP16 satisfies -> FP16."""
        input_data = DecisionInput(
            safety_probability=0.80,
            safety_threshold=0.50,
            carbon_intensity_gco2_per_kwh=200,
            fp16_estimated_energy_wh=0.0488,
            int4_estimated_energy_wh=0.0493,
            fp16_estimated_latency_ms=5867,
            int4_estimated_latency_ms=10580,
            latency_requirement_ms=8000,  # FP16 OK, INT4 violates
            carbon_data_source=DataSourceType.LIVE,
            gpu_vram_free_mb=4000,  # Sufficient for both configs
        )
        
        decision = self.engine.decide(input_data)
        
        assert decision.selected_precision == Precision.FP16
        assert decision.decision_reason == DecisionReason.INT4_LATENCY_VIOLATION
        assert decision.latency_constraint_status == "int4_violated"
        assert decision.latency_requirement_ms == 8000
    
    def test_fp16_violates_latency_int4_satisfies(self):
        """FP16 violates latency, INT4 satisfies -> INT4 (if safe)."""
        # Use a policy with swapped latencies for this test
        policy = DecisionPolicy(
            fp16_latency_ms_override=15000,
            int4_latency_ms_override=5000,
        )
        engine = DecisionEngine(policy)
        
        input_data = DecisionInput(
            safety_probability=0.80,
            safety_threshold=0.50,
            carbon_intensity_gco2_per_kwh=200,
            fp16_estimated_energy_wh=0.0488,
            int4_estimated_energy_wh=0.0493,
            fp16_estimated_latency_ms=15000,
            int4_estimated_latency_ms=5000,
            latency_requirement_ms=8000,
            carbon_data_source=DataSourceType.LIVE,
            gpu_vram_free_mb=4000,  # Sufficient for both configs
        )
        
        decision = engine.decide(input_data)
        
        assert decision.selected_precision == Precision.INT4
        assert decision.decision_reason == DecisionReason.FP16_LATENCY_VIOLATION
        assert decision.latency_constraint_status == "fp16_violated"
    
    def test_both_violate_latency_fallback(self):
        """Both violate latency -> fallback policy."""
        input_data = DecisionInput(
            safety_probability=0.80,
            safety_threshold=0.50,
            carbon_intensity_gco2_per_kwh=200,
            fp16_estimated_energy_wh=0.0488,
            int4_estimated_energy_wh=0.0493,
            fp16_estimated_latency_ms=5867,
            int4_estimated_latency_ms=10580,
            latency_requirement_ms=3000,  # Both violate
            carbon_data_source=DataSourceType.LIVE,
            gpu_vram_free_mb=4000,  # Sufficient for both configs
        )
        
        decision = self.engine.decide(input_data)
        
        assert decision.latency_constraint_status == "both_violated"
        assert decision.fallback_used is True
        assert decision.decision_reason == DecisionReason.FALLBACK_POLICY
        # Default fallback prefers FP16
        assert decision.selected_precision == Precision.FP16
    
    def test_missing_latency_requirement_no_artificial_sla(self):
        """TEST 7: Missing latency requirement -> no artificial SLA."""
        input_data = DecisionInput(
            safety_probability=0.80,
            safety_threshold=0.50,
            carbon_intensity_gco2_per_kwh=200,
            fp16_estimated_energy_wh=0.0488,
            int4_estimated_energy_wh=0.0493,
            fp16_estimated_latency_ms=5867,
            int4_estimated_latency_ms=10580,
            latency_requirement_ms=None,  # Not specified
            carbon_data_source=DataSourceType.LIVE,
        )
        
        decision = self.engine.decide(input_data)
        
        assert decision.latency_constraint_status == "not_specified"
        assert decision.latency_requirement_ms is None
        # Should proceed to environmental cost comparison


class TestDecisionEngineEnvironmentalCost:
    """Test energy/CO2 comparison logic (TEST 2, 4)."""
    
    def setup_method(self):
        self.engine = DecisionEngine(DEFAULT_POLICY)
    
    def test_int4_lower_environmental_cost(self):
        """TEST 2: Safe + INT4 lower environmental cost -> INT4."""
        # Use B2 workload profile where INT4 uses less energy
        policy = DecisionPolicy(profile_name="b2_workload")
        engine = DecisionEngine(policy)
        
        input_data = DecisionInput(
            safety_probability=0.80,
            safety_threshold=0.50,
            carbon_intensity_gco2_per_kwh=200,
            fp16_estimated_energy_wh=0.1061,  # B2 FP16
            int4_estimated_energy_wh=0.0945,  # B2 INT4 (lower)
            fp16_estimated_latency_ms=11640,
            int4_estimated_latency_ms=22077,
            carbon_data_source=DataSourceType.LIVE,
            energy_data_source=DataSourceType.MEASURED,
            gpu_vram_free_mb=4000,  # Sufficient for both configs
        )
        
        decision = engine.decide(input_data)
        
        # INT4 has lower energy -> lower CO2 -> should select INT4
        assert decision.selected_precision == Precision.INT4
        assert decision.decision_reason == DecisionReason.INT4_LOWER_ENVIRONMENTAL_COST
        assert decision.estimated_co2_g < decision.fp16_estimated_co2_g
    
    def test_fp16_lower_environmental_cost(self):
        """TEST 4: Safe + FP16 lower environmental cost -> FP16."""
        # Phase 0.5 profile: FP16 uses slightly less energy
        # Use a smaller CO2 difference threshold to ensure the difference is significant
        policy = DecisionPolicy(profile_name="phase05", co2_difference_threshold_mg=0.05)
        engine = DecisionEngine(policy)
        
        input_data = DecisionInput(
            safety_probability=0.80,
            safety_threshold=0.50,
            carbon_intensity_gco2_per_kwh=200,
            fp16_estimated_energy_wh=0.0488,  # Phase 0.5 FP16
            int4_estimated_energy_wh=0.0493,  # Phase 0.5 INT4 (higher)
            fp16_estimated_latency_ms=5867,
            int4_estimated_latency_ms=10580,
            carbon_data_source=DataSourceType.LIVE,
            energy_data_source=DataSourceType.MEASURED,
            gpu_vram_free_mb=4000,  # Sufficient for both configs
        )
        
        decision = engine.decide(input_data)
        
        # FP16 has lower energy -> lower CO2 -> should select FP16
        assert decision.selected_precision == Precision.FP16
        assert decision.decision_reason == DecisionReason.FP16_LOWER_ENVIRONMENTAL_COST
        assert decision.estimated_co2_g < decision.int4_estimated_co2_g
    
    def test_equal_environmental_cost_tiebreak_latency(self):
        """TEST 9: Equal environmental cost -> tie-break by latency."""
        policy = DecisionPolicy(
            co2_difference_threshold_mg=10.0,  # Large threshold
            fp16_energy_wh_override=0.05,
            int4_energy_wh_override=0.05,
            fp16_latency_ms_override=5000,
            int4_latency_ms_override=8000,
        )
        engine = DecisionEngine(policy)
        
        input_data = DecisionInput(
            safety_probability=0.80,
            safety_threshold=0.50,
            carbon_intensity_gco2_per_kwh=200,
            fp16_estimated_energy_wh=0.05,
            int4_estimated_energy_wh=0.05,
            fp16_estimated_latency_ms=5000,
            int4_estimated_latency_ms=8000,
            carbon_data_source=DataSourceType.LIVE,
            gpu_vram_free_mb=4000,  # Sufficient for both configs
        )
        
        decision = engine.decide(input_data)
        
        # Equal CO2, FP16 has lower latency -> FP16
        assert decision.decision_reason == DecisionReason.ENVIRONMENTAL_COST_EQUAL
        assert decision.selected_precision == Precision.FP16


class TestDecisionEngineCarbonSource:
    """Test carbon data source handling (TEST 5, 6)."""
    
    def setup_method(self):
        self.engine = DecisionEngine(DEFAULT_POLICY)
    
    def test_live_carbon_source_recorded(self):
        """TEST 5: Live carbon source correctly recorded."""
        input_data = DecisionInput(
            safety_probability=0.80,
            safety_threshold=0.50,
            carbon_intensity_gco2_per_kwh=150,
            fp16_estimated_energy_wh=0.0488,
            int4_estimated_energy_wh=0.0493,
            fp16_estimated_latency_ms=5867,
            int4_estimated_latency_ms=10580,
            carbon_data_source=DataSourceType.LIVE,
        )
        
        decision = self.engine.decide(input_data)
        
        assert decision.carbon_data_source == DataSourceType.LIVE
        assert decision.carbon_intensity_gco2_per_kwh == 150
    
    def test_replay_carbon_source_recorded(self):
        """Replay carbon source correctly recorded."""
        input_data = DecisionInput(
            safety_probability=0.80,
            safety_threshold=0.50,
            carbon_intensity_gco2_per_kwh=180,
            fp16_estimated_energy_wh=0.0488,
            int4_estimated_energy_wh=0.0493,
            fp16_estimated_latency_ms=5867,
            int4_estimated_latency_ms=10580,
            carbon_data_source=DataSourceType.REPLAY,
        )
        
        decision = self.engine.decide(input_data)
        
        assert decision.carbon_data_source == DataSourceType.REPLAY
    
    def test_offline_fallback_works(self):
        """TEST 6: Offline fallback works when live unavailable."""
        input_data = DecisionInput(
            safety_probability=0.80,
            safety_threshold=0.50,
            carbon_intensity_gco2_per_kwh=200,  # Default offline value
            fp16_estimated_energy_wh=0.0488,
            int4_estimated_energy_wh=0.0493,
            fp16_estimated_latency_ms=5867,
            int4_estimated_latency_ms=10580,
            carbon_data_source=DataSourceType.OFFLINE,
        )
        
        decision = self.engine.decide(input_data)
        
        assert decision.carbon_data_source == DataSourceType.OFFLINE
        assert decision.carbon_intensity_gco2_per_kwh == 200


class TestDecisionEngineResource:
    """Test resource feasibility (TEST 8)."""
    
    def setup_method(self):
        self.engine = DecisionEngine(DEFAULT_POLICY)
    
    def test_missing_resource_telemetry_no_fabrication(self):
        """TEST 8: Missing resource telemetry -> no fabrication (legacy caller)."""
        input_data = DecisionInput(
            safety_probability=0.80,
            safety_threshold=0.50,
            carbon_intensity_gco2_per_kwh=200,
            fp16_estimated_energy_wh=0.0488,
            int4_estimated_energy_wh=0.0493,
            fp16_estimated_latency_ms=5867,
            int4_estimated_latency_ms=10580,
            carbon_data_source=DataSourceType.LIVE,
            resource_state=None,
            gpu_vram_free_mb=None,
            vram_telemetry_available=False,  # Legacy caller did not attempt telemetry
        )
        
        decision = self.engine.decide(input_data)
        
        assert decision.resource_state is None
        assert decision.gpu_vram_free_mb is None
        # Should still make a decision based on other factors (legacy behavior)
        assert decision.selected_precision in [Precision.FP16, Precision.INT4]
    
    def test_resource_constraint_fp16_blocked(self):
        """Low VRAM blocks FP16 but allows INT4."""
        policy = DecisionPolicy(min_vram_mb_for_fp16=3200, min_vram_mb_for_int4=1400)
        engine = DecisionEngine(policy)
        
        input_data = DecisionInput(
            safety_probability=0.80,
            safety_threshold=0.50,
            carbon_intensity_gco2_per_kwh=200,
            fp16_estimated_energy_wh=0.0488,
            int4_estimated_energy_wh=0.0493,
            fp16_estimated_latency_ms=5867,
            int4_estimated_latency_ms=10580,
            carbon_data_source=DataSourceType.LIVE,
            gpu_vram_free_mb=2000,  # Enough for INT4, not FP16
            vram_telemetry_available=True,  # Valid telemetry provided
        )
        
        decision = engine.decide(input_data)
        
        assert decision.selected_precision == Precision.INT4
        assert decision.decision_reason == DecisionReason.RESOURCE_CONSTRAINT


class TestVRAMTelemetrySemantics:
    """Test VRAM telemetry semantics (Fix 1)."""
    
    def setup_method(self):
        self.engine = DecisionEngine(DEFAULT_POLICY)
    
    def test_valid_measured_vram_allows_both(self):
        """Valid measured VRAM -> normal threshold checks (both feasible)."""
        input_data = DecisionInput(
            safety_probability=0.80,
            safety_threshold=0.50,
            carbon_intensity_gco2_per_kwh=200,
            fp16_estimated_energy_wh=0.0488,
            int4_estimated_energy_wh=0.0493,
            fp16_estimated_latency_ms=5867,
            int4_estimated_latency_ms=10580,
            carbon_data_source=DataSourceType.LIVE,
            gpu_vram_free_mb=4000,  # Sufficient for both
            vram_telemetry_available=True,
        )
        
        decision = self.engine.decide(input_data)
        
        # Both configs feasible - proceeds to environmental cost comparison
        assert decision.selected_precision in [Precision.FP16, Precision.INT4]
        assert decision.decision_reason != DecisionReason.RESOURCE_CONSTRAINT
    
    def test_valid_measured_vram_blocks_fp16(self):
        """Valid measured VRAM below FP16 threshold -> INT4 selected."""
        input_data = DecisionInput(
            safety_probability=0.80,
            safety_threshold=0.50,
            carbon_intensity_gco2_per_kwh=200,
            fp16_estimated_energy_wh=0.0488,
            int4_estimated_energy_wh=0.0493,
            fp16_estimated_latency_ms=5867,
            int4_estimated_latency_ms=10580,
            carbon_data_source=DataSourceType.LIVE,
            gpu_vram_free_mb=2000,  # Enough for INT4 (1400), not FP16 (3200)
            vram_telemetry_available=True,
        )
        
        decision = self.engine.decide(input_data)
        
        assert decision.selected_precision == Precision.INT4
        assert decision.decision_reason == DecisionReason.RESOURCE_CONSTRAINT
    
    def test_vram_telemetry_unavailable_fails_closed(self):
        """VRAM telemetry attempted but failed -> fail closed (neither feasible)."""
        input_data = DecisionInput(
            safety_probability=0.80,
            safety_threshold=0.50,
            carbon_intensity_gco2_per_kwh=200,
            fp16_estimated_energy_wh=0.0488,
            int4_estimated_energy_wh=0.0493,
            fp16_estimated_latency_ms=5867,
            int4_estimated_latency_ms=10580,
            carbon_data_source=DataSourceType.LIVE,
            gpu_vram_free_mb=None,  # Measurement failed
            vram_telemetry_available=True,  # Telemetry was attempted
        )
        
        decision = self.engine.decide(input_data)
        
        # Both configs infeasible -> fallback used
        assert decision.fallback_used is True
        assert decision.decision_reason == DecisionReason.RESOURCE_CONSTRAINT
    
    def test_invalid_vram_nan_fails_closed(self):
        """NaN VRAM -> fail closed."""
        input_data = DecisionInput(
            safety_probability=0.80,
            safety_threshold=0.50,
            carbon_intensity_gco2_per_kwh=200,
            fp16_estimated_energy_wh=0.0488,
            int4_estimated_energy_wh=0.0493,
            fp16_estimated_latency_ms=5867,
            int4_estimated_latency_ms=10580,
            carbon_data_source=DataSourceType.LIVE,
            gpu_vram_free_mb=float('nan'),
            vram_telemetry_available=True,
        )
        
        decision = self.engine.decide(input_data)
        
        assert decision.fallback_used is True
        assert decision.decision_reason == DecisionReason.RESOURCE_CONSTRAINT
    
    def test_invalid_vram_negative_fails_closed(self):
        """Negative VRAM -> fail closed."""
        input_data = DecisionInput(
            safety_probability=0.80,
            safety_threshold=0.50,
            carbon_intensity_gco2_per_kwh=200,
            fp16_estimated_energy_wh=0.0488,
            int4_estimated_energy_wh=0.0493,
            fp16_estimated_latency_ms=5867,
            int4_estimated_latency_ms=10580,
            carbon_data_source=DataSourceType.LIVE,
            gpu_vram_free_mb=-100,
            vram_telemetry_available=True,
        )
        
        decision = self.engine.decide(input_data)
        
        assert decision.fallback_used is True
        assert decision.decision_reason == DecisionReason.RESOURCE_CONSTRAINT
    
    def test_invalid_vram_inf_fails_closed(self):
        """Infinite VRAM -> fail closed."""
        input_data = DecisionInput(
            safety_probability=0.80,
            safety_threshold=0.50,
            carbon_intensity_gco2_per_kwh=200,
            fp16_estimated_energy_wh=0.0488,
            int4_estimated_energy_wh=0.0493,
            fp16_estimated_latency_ms=5867,
            int4_estimated_latency_ms=10580,
            carbon_data_source=DataSourceType.LIVE,
            gpu_vram_free_mb=float('inf'),
            vram_telemetry_available=True,
        )
        
        decision = self.engine.decide(input_data)
        
        assert decision.fallback_used is True
        assert decision.decision_reason == DecisionReason.RESOURCE_CONSTRAINT
    
    def test_legacy_caller_without_telemetry_skips_check(self):
        """Legacy/direct caller without vram_telemetry_available -> skips VRAM check."""
        input_data = DecisionInput(
            safety_probability=0.80,
            safety_threshold=0.50,
            carbon_intensity_gco2_per_kwh=200,
            fp16_estimated_energy_wh=0.0488,
            int4_estimated_energy_wh=0.0493,
            fp16_estimated_latency_ms=5867,
            int4_estimated_latency_ms=10580,
            carbon_data_source=DataSourceType.LIVE,
            gpu_vram_free_mb=None,
            # vram_telemetry_available defaults to False (legacy behavior)
        )
        
        decision = self.engine.decide(input_data)
        
        # Should proceed to environmental cost comparison (legacy behavior)
        assert decision.selected_precision in [Precision.FP16, Precision.INT4]
        assert decision.decision_reason != DecisionReason.RESOURCE_CONSTRAINT
    
    def test_both_configs_infeasible_vram_unavailable(self):
        """Both configurations infeasible when VRAM unavailable -> fallback used."""
        input_data = DecisionInput(
            safety_probability=0.80,
            safety_threshold=0.50,
            carbon_intensity_gco2_per_kwh=200,
            fp16_estimated_energy_wh=0.0488,
            int4_estimated_energy_wh=0.0493,
            fp16_estimated_latency_ms=5867,
            int4_estimated_latency_ms=10580,
            carbon_data_source=DataSourceType.LIVE,
            gpu_vram_free_mb=None,
            vram_telemetry_available=True,
        )
        
        decision = self.engine.decide(input_data)
        
        assert decision.fallback_used is True
        assert decision.decision_reason == DecisionReason.RESOURCE_CONSTRAINT
        # Default fallback policy prefers FP16
        assert decision.selected_precision == Precision.FP16
    
    def test_normal_feasible_configuration(self):
        """Normal feasible configuration with valid VRAM works correctly."""
        input_data = DecisionInput(
            safety_probability=0.80,
            safety_threshold=0.50,
            carbon_intensity_gco2_per_kwh=200,
            fp16_estimated_energy_wh=0.0488,
            int4_estimated_energy_wh=0.0493,
            fp16_estimated_latency_ms=5867,
            int4_estimated_latency_ms=10580,
            carbon_data_source=DataSourceType.LIVE,
            gpu_vram_free_mb=4000,
            vram_telemetry_available=True,
        )
        
        decision = self.engine.decide(input_data)
        
        # Both feasible, FP16 has slightly lower energy in phase05 profile
        assert decision.selected_precision == Precision.FP16
        assert decision.decision_reason == DecisionReason.FP16_LOWER_ENVIRONMENTAL_COST
        assert decision.fallback_used is False


class TestDecisionEngineOutput:
    """Test decision output serialization and structure (TEST 10, 11, 12)."""
    
    def setup_method(self):
        self.engine = DecisionEngine(DEFAULT_POLICY)
    
    def test_output_serializable_to_json(self):
        """TEST 10: Decision output fully serializable to JSON."""
        input_data = DecisionInput(
            safety_probability=0.80,
            safety_threshold=0.50,
            carbon_intensity_gco2_per_kwh=200,
            fp16_estimated_energy_wh=0.0488,
            int4_estimated_energy_wh=0.0493,
            fp16_estimated_latency_ms=5867,
            int4_estimated_latency_ms=10580,
            carbon_data_source=DataSourceType.LIVE,
        )
        
        decision = self.engine.decide(input_data)
        
        # Should serialize without error
        json_str = decision.to_json()
        assert isinstance(json_str, str)
        
        # Should deserialize correctly
        parsed = DecisionOutput.from_dict(decision.to_dict())
        assert parsed.selected_precision == decision.selected_precision
        assert parsed.decision_reason == decision.decision_reason
    
    def test_decision_reason_explains_fp16(self):
        """TEST 11: Decision reason explains why FP16 was selected."""
        input_data = DecisionInput(
            safety_probability=0.30,  # Unsafe
            safety_threshold=0.50,
            carbon_intensity_gco2_per_kwh=100,
            fp16_estimated_energy_wh=0.0488,
            int4_estimated_energy_wh=0.0493,
            fp16_estimated_latency_ms=5867,
            int4_estimated_latency_ms=10580,
            carbon_data_source=DataSourceType.LIVE,
        )
        
        decision = self.engine.decide(input_data)
        
        assert "safety_gate" in decision.decision_reason.value
        assert decision.selected_precision == Precision.FP16
    
    def test_decision_reason_explains_int4(self):
        """TEST 12: Decision reason explains why INT4 was selected."""
        policy = DecisionPolicy(profile_name="b2_workload")
        engine = DecisionEngine(policy)
        
        input_data = DecisionInput(
            safety_probability=0.80,
            safety_threshold=0.50,
            carbon_intensity_gco2_per_kwh=200,
            fp16_estimated_energy_wh=0.1061,
            int4_estimated_energy_wh=0.0945,
            fp16_estimated_latency_ms=11640,
            int4_estimated_latency_ms=22077,
            carbon_data_source=DataSourceType.LIVE,
            gpu_vram_free_mb=4000,  # Sufficient for both configs
        )
        
        decision = engine.decide(input_data)
        
        assert decision.selected_precision == Precision.INT4
        assert "environmental_cost" in decision.decision_reason.value


class TestCO2UnitConversion:
    """Test CO2 unit conversion correctness (TEST 14)."""
    
    def test_co2_conversion_wh_to_g(self):
        """TEST 14: CO2 unit conversion Wh -> g CO2."""
        # 1 Wh = 0.001 kWh
        # At 200 gCO2/kWh: 0.001 * 200 = 0.2 g = 200 mg
        energy_wh = 1.0
        intensity = 200
        co2_g, co2_mg = estimate_co2_from_energy(energy_wh, intensity)
        
        assert co2_g == 0.2
        assert co2_mg == 200.0
    
    def test_co2_conversion_mwh(self):
        """Test CO2 from milliwatt-hours."""
        energy_wh = 0.05  # 50 mWh
        intensity = 400
        co2_g, co2_mg = estimate_co2_from_energy(energy_wh, intensity)
        
        # 0.05 Wh = 0.00005 kWh
        # 0.00005 * 400 = 0.02 g = 20 mg
        assert co2_g == 0.02
        assert co2_mg == 20.0


class TestArtifactsUntouched:
    """Test that C.1/C.1.1 artifacts remain untouched (TEST 15)."""
    
    def test_c1_artifacts_exist(self):
        """Verify C.1 artifacts exist."""
        import os
        assert os.path.exists("models/quantization_safety/classifier.pkl")
        assert os.path.exists("models/quantization_safety/metadata_c1.json")
    
    def test_c1_1_artifacts_exist(self):
        """Verify C.1.1 artifacts exist."""
        import os
        assert os.path.exists("models/quantization_safety/classifier_c1_1.pkl")
        assert os.path.exists("models/quantization_safety/feature_extractor_c1_1.pkl")
        assert os.path.exists("models/quantization_safety/metadata_c1_1.json")
    
    def test_b2_dataset_untouched(self):
        """Verify B2 dataset exists and is valid."""
        import os
        import json
        assert os.path.exists("data/processed/b2_dataset.json")
        
        with open("data/processed/b2_dataset.json") as f:
            data = json.load(f)
        
        assert len(data) == 182
        # Verify structure
        assert "prompt" in data[0]
        assert "safe_to_quantize" in data[0]
        assert "complexity_level" in data[0]


class TestPolicyConfiguration:
    """Test policy configuration and defaults."""
    
    def test_default_policy_values(self):
        """Test default policy has expected values."""
        policy = DEFAULT_POLICY
        
        assert policy.safety_threshold == 0.50
        assert policy.latency_tolerance_pct == 0.10
        assert policy.fallback_policy == FallbackPolicy.PREFER_FP16
        assert policy.default_precision == Precision.FP16
        assert policy.profile_name == "phase05"
    
    def test_policy_override(self):
        """Test policy can be overridden."""
        policy = create_policy(
            safety_threshold=0.60,
            fallback_policy=FallbackPolicy.PREFER_LOWER_LATENCY,
            profile_name="b2_workload",
        )
        
        assert policy.safety_threshold == 0.60
        assert policy.fallback_policy == FallbackPolicy.PREFER_LOWER_LATENCY
        assert policy.profile_name == "b2_workload"
    
    def test_profile_selection(self):
        """Test profile selection affects energy/latency values."""
        policy_phase05 = DecisionPolicy(profile_name="phase05")
        policy_b2 = DecisionPolicy(profile_name="b2_workload")
        
        # Phase 0.5: FP16 slightly lower energy
        assert policy_phase05.get_fp16_energy_wh() == 0.0488
        assert policy_phase05.get_int4_energy_wh() == 0.0493
        
        # B2 workload: INT4 lower energy
        assert policy_b2.get_fp16_energy_wh() == 0.1061
        assert policy_b2.get_int4_energy_wh() == 0.0945


class TestDecisionInputOutputModels:
    """Test DecisionInput/Output model serialization."""
    
    def test_decision_input_roundtrip(self):
        """Test DecisionInput to_dict/from_dict roundtrip."""
        input_data = DecisionInput(
            safety_probability=0.80,
            safety_threshold=0.50,
            carbon_intensity_gco2_per_kwh=200,
            carbon_data_source=DataSourceType.LIVE,
            fp16_estimated_energy_wh=0.0488,
            int4_estimated_energy_wh=0.0493,
            fp16_estimated_latency_ms=5867,
            int4_estimated_latency_ms=10580,
        )
        
        d = input_data.to_dict()
        restored = DecisionInput.from_dict(d)
        
        assert restored.safety_probability == input_data.safety_probability
        assert restored.carbon_data_source == input_data.carbon_data_source
        assert restored.fallback_policy == input_data.fallback_policy
    
    def test_decision_output_roundtrip(self):
        """Test DecisionOutput to_dict/from_dict roundtrip."""
        output = DecisionOutput(
            selected_precision=Precision.INT4,
            decision_reason=DecisionReason.INT4_LOWER_ENVIRONMENTAL_COST,
            safety_probability=0.80,
            safety_threshold=0.50,
            safety_gate_passed=True,
            carbon_intensity_gco2_per_kwh=200,
            carbon_data_source=DataSourceType.LIVE,
            carbon_zone="national",
            carbon_reading_type="actual",
            estimated_energy_wh=0.0945,
            estimated_co2_g=0.0189,
            estimated_co2_mg=18.9,
            estimated_latency_ms=22077,
            latency_constraint_status="satisfied",
            fp16_estimated_energy_wh=0.1061,
            fp16_estimated_co2_g=0.0212,
            fp16_estimated_latency_ms=11640,
            int4_estimated_energy_wh=0.0945,
            int4_estimated_co2_g=0.0189,
            int4_estimated_latency_ms=22077,
            energy_data_source=DataSourceType.MEASURED,
        )
        
        d = output.to_dict()
        restored = DecisionOutput.from_dict(d)
        
        assert restored.selected_precision == output.selected_precision
        assert restored.decision_reason == output.decision_reason
        assert restored.carbon_data_source == output.carbon_data_source
        assert restored.energy_data_source == output.energy_data_source


class TestEnergyPredictorValidation:
    """Test energy predictor validation guards (Fix 5)."""

    def test_validate_energy_prediction_valid_positive(self):
        """Valid positive prediction passes through unchanged."""
        from carbongrid.evaluation.energy_predictor_utils import validate_energy_prediction
        
        result = validate_energy_prediction(0.001)
        assert result == 0.001
        
        result = validate_energy_prediction(0.5)
        assert result == 0.5
        
        result = validate_energy_prediction(5.0)
        assert result == 5.0

    def test_validate_energy_prediction_zero(self):
        """Zero prediction is valid (no energy difference)."""
        from carbongrid.evaluation.energy_predictor_utils import validate_energy_prediction
        
        result = validate_energy_prediction(0.0)
        assert result == 0.0

    def test_validate_energy_prediction_negative_clamped(self):
        """Negative prediction is clamped to minimum (0.0)."""
        from carbongrid.evaluation.energy_predictor_utils import validate_energy_prediction
        
        result = validate_energy_prediction(-0.001)
        assert result == 0.0  # Clamped to MIN_VALID_ENERGY_WH

    def test_validate_energy_prediction_nan_raises(self):
        """NaN prediction raises EnergyPredictorInvalidPredictionError."""
        from carbongrid.evaluation.energy_predictor_utils import (
            validate_energy_prediction, EnergyPredictorInvalidPredictionError
        )
        
        with pytest.raises(EnergyPredictorInvalidPredictionError, match="NaN"):
            validate_energy_prediction(float('nan'))

    def test_validate_energy_prediction_inf_raises(self):
        """Infinite prediction raises EnergyPredictorInvalidPredictionError."""
        from carbongrid.evaluation.energy_predictor_utils import (
            validate_energy_prediction, EnergyPredictorInvalidPredictionError
        )
        
        with pytest.raises(EnergyPredictorInvalidPredictionError, match="infinite"):
            validate_energy_prediction(float('inf'))
        
        with pytest.raises(EnergyPredictorInvalidPredictionError, match="infinite"):
            validate_energy_prediction(float('-inf'))

    def test_validate_energy_prediction_none_raises(self):
        """None prediction raises EnergyPredictorInvalidPredictionError."""
        from carbongrid.evaluation.energy_predictor_utils import (
            validate_energy_prediction, EnergyPredictorInvalidPredictionError
        )
        
        with pytest.raises(EnergyPredictorInvalidPredictionError, match="None"):
            validate_energy_prediction(None)

    def test_validate_energy_prediction_string_raises(self):
        """String prediction raises EnergyPredictorInvalidPredictionError."""
        from carbongrid.evaluation.energy_predictor_utils import (
            validate_energy_prediction, EnergyPredictorInvalidPredictionError
        )
        
        with pytest.raises(EnergyPredictorInvalidPredictionError, match="not numeric"):
            validate_energy_prediction("0.001")

    def test_validate_energy_prediction_too_large_clamped(self):
        """Implausibly high prediction is clamped to maximum."""
        from carbongrid.evaluation.energy_predictor_utils import validate_energy_prediction
        
        result = validate_energy_prediction(100.0)
        assert result == 10.0  # Clamped to MAX_VALID_ENERGY_WH

    def test_energy_predictor_status_constants(self):
        """Verify predictor status constants are correct."""
        from carbongrid.evaluation.energy_predictor_utils import PREDICTOR_STATUS
        
        assert PREDICTOR_STATUS["used_in_production_routing"] is False
        assert PREDICTOR_STATUS["used_in_telemetry"] is False
        assert "EXPLORATORY" in PREDICTOR_STATUS["audit_status"]
        assert "Negative predictions observed" in str(PREDICTOR_STATUS["known_issues"])
        assert PREDICTOR_STATUS["recommendation"] == "Do not use for routing decisions. Use profile-based estimates (Phase 0.5, B2) instead."

    def test_predict_energy_difference_safe_returns_fallback(self):
        """predict_energy_difference_safe returns fallback with metadata."""
        from carbongrid.evaluation.energy_predictor_utils import predict_energy_difference_safe
        
        prediction, metadata = predict_energy_difference_safe({}, fallback=0.0)
        
        assert prediction == 0.0
        assert metadata["fallback_used"] is True
        assert metadata["success"] is False
        assert "EXPLORATORY ONLY" in metadata.get("note", "")

    def test_predictor_not_used_in_production_routing(self):
        """Confirm predictor is not used in production routing path."""
        from carbongrid.evaluation.energy_predictor_utils import PREDICTOR_STATUS
        
        assert PREDICTOR_STATUS["used_in_production_routing"] is False
        assert PREDICTOR_STATUS["used_in_telemetry"] is False
        assert "do not use" in PREDICTOR_STATUS["recommendation"].lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])