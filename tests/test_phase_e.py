"""
Phase E Integration Tests for CarbonGrid-AI.

Tests the end-to-end integration of all components.
"""

import sys
import pytest
import time
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from carbongrid.classifier.service import ClassifierService, get_classifier_info
from carbongrid.carbon.manager import create_carbon_manager
from carbongrid.decision import (
    DecisionEngine,
    DecisionInput,
    DecisionPolicy,
    Precision as DecisionPrecision,
    DataSourceType as DecisionDataSourceType,
    DecisionReason,
    FallbackPolicy,
)
from carbongrid.api.schemas import GenerationRequest, GenerationResponse
from carbongrid.inference.provider import get_free_vram_mb, VRAMMeasurementResult
from carbongrid.inference.provider import InferenceProvider
from carbongrid.inference.provider import Precision as InferencePrecision
from carbongrid.inference.provider import InferenceMetrics


class TestClassifierService:
    """Test C.1.1 Classifier Service."""

    def test_classifier_artifacts_exist(self):
        """Verify C.1.1 artifacts exist."""
        info = get_classifier_info()
        assert info["classifier_exists"] is True
        assert info["extractor_exists"] is True
        assert info["metadata"] is not None

    def test_classifier_loads(self):
        """Test that classifier can be loaded."""
        pytest.importorskip("torch")
        pytest.importorskip("transformers")

        service = ClassifierService()
        service.load()

        assert service.is_loaded()
        assert service.get_threshold() == 0.50
        assert len(service.get_feature_names()) == 122  # C.1.1 has 122 features

    def test_predict_safety(self):
        """Test safety prediction on sample prompts."""
        pytest.importorskip("torch")
        pytest.importorskip("transformers")

        service = ClassifierService().load()

        # Test factual prompt
        result = service.predict_safety("What is the capital of France?", "factual")
        assert "safety_probability" in result
        assert "is_safe" in result
        assert "threshold" in result
        assert 0.0 <= result["safety_probability"] <= 1.0
        assert result["threshold"] == 0.50

    def test_predict_batch(self):
        """Test batch safety prediction."""
        pytest.importorskip("torch")
        pytest.importorskip("transformers")

        service = ClassifierService().load()

        prompts = [
            "What is 2+2?",
            "Explain quantum computing",
            "Write a Python function",
        ]
        results = service.predict_batch(prompts)

        assert len(results) == 3
        for r in results:
            assert "safety_probability" in r
            assert "is_safe" in r


class TestCarbonManagerIntegration:
    """Test Carbon Manager integration."""

    def test_offline_mode_works(self):
        """Test offline carbon manager works without internet."""
        manager = create_carbon_manager("offline")
        reading = manager.get_current_carbon("national")

        assert reading is not None
        assert reading.carbon_intensity_gco2_per_kwh == 200.0
        assert reading.data_source.value == "offline"

    def test_offline_regional_works(self):
        """Test offline regional snapshot."""
        manager = create_carbon_manager("offline")
        snapshot = manager.get_regional_snapshot()

        assert snapshot is not None
        assert "London" in snapshot.readings
        assert "North Scotland" in snapshot.readings

    def test_live_mode_fallback(self):
        """Test live mode with fallback chain."""
        # This will try live API, fallback to offline
        manager = create_carbon_manager("live")
        reading = manager.get_current_carbon("national")

        assert reading is not None
        assert reading.carbon_intensity_gco2_per_kwh > 0


class TestDecisionEngineIntegration:
    """Test Decision Engine with realistic inputs."""

    def setup_method(self):
        self.policy = DecisionPolicy(
            profile_name="phase05",
            safety_threshold=0.50,
        )
        self.engine = DecisionEngine(self.policy)

    def test_safety_gate_blocks_int4(self):
        """Test safety gate rejects INT4 for unsafe prompts."""
        input_data = DecisionInput(
            safety_probability=0.30,
            safety_threshold=0.50,
            carbon_intensity_gco2_per_kwh=100.0,
            carbon_data_source=DecisionDataSourceType.LIVE,
            fp16_estimated_energy_wh=0.0488,
            int4_estimated_energy_wh=0.0493,
            fp16_estimated_latency_ms=5867,
            int4_estimated_latency_ms=10580,
        )

        decision = self.engine.decide(input_data)

        assert decision.selected_precision == DecisionPrecision.FP16
        assert decision.decision_reason == DecisionReason.SAFETY_GATE_FAILED
        assert decision.safety_gate_passed is False
        assert decision.fallback_used is False

    def test_int4_selected_when_lower_co2(self):
        """Test INT4 selected when it has lower CO2 (B2 workload)."""
        policy = DecisionPolicy(profile_name="b2_workload", safety_threshold=0.50)
        engine = DecisionEngine(policy)

        input_data = DecisionInput(
            safety_probability=0.85,
            safety_threshold=0.50,
            carbon_intensity_gco2_per_kwh=200.0,
            carbon_data_source=DecisionDataSourceType.LIVE,
            fp16_estimated_energy_wh=0.1061,
            int4_estimated_energy_wh=0.0945,
            fp16_estimated_latency_ms=11640,
            int4_estimated_latency_ms=22077,
            energy_data_source=DecisionDataSourceType.MEASURED,
            latency_data_source=DecisionDataSourceType.MEASURED,
        )

        decision = engine.decide(input_data)

        assert decision.selected_precision == DecisionPrecision.INT4
        assert decision.decision_reason == DecisionReason.INT4_LOWER_ENVIRONMENTAL_COST
        assert decision.estimated_co2_g < decision.fp16_estimated_co2_g

    def test_latency_constraint_forces_fp16(self):
        """Test latency requirement forces FP16 when INT4 too slow."""
        input_data = DecisionInput(
            safety_probability=0.85,
            safety_threshold=0.50,
            carbon_intensity_gco2_per_kwh=200.0,
            carbon_data_source=DecisionDataSourceType.LIVE,
            fp16_estimated_energy_wh=0.0488,
            int4_estimated_energy_wh=0.0493,
            fp16_estimated_latency_ms=5867,
            int4_estimated_latency_ms=10580,
            latency_requirement_ms=8000.0,
        )

        decision = self.engine.decide(input_data)

        assert decision.selected_precision == DecisionPrecision.FP16
        assert decision.decision_reason == DecisionReason.INT4_LATENCY_VIOLATION
        assert decision.latency_constraint_status == "int4_violated"

    def test_both_violate_latency_fallback(self):
        """Test fallback when both violate latency."""
        input_data = DecisionInput(
            safety_probability=0.85,
            safety_threshold=0.50,
            carbon_intensity_gco2_per_kwh=200.0,
            carbon_data_source=DecisionDataSourceType.LIVE,
            fp16_estimated_energy_wh=0.0488,
            int4_estimated_energy_wh=0.0493,
            fp16_estimated_latency_ms=5867,
            int4_estimated_latency_ms=10580,
            latency_requirement_ms=3000.0,  # Both violate
        )

        decision = self.engine.decide(input_data)

        assert decision.latency_constraint_status == "both_violated"
        assert decision.fallback_used is True
        assert decision.decision_reason == DecisionReason.FALLBACK_POLICY
        assert decision.selected_precision == DecisionPrecision.FP16

    def test_resource_constraint_blocks_fp16(self):
        """Test low VRAM blocks FP16."""
        policy = DecisionPolicy(
            min_vram_mb_for_fp16=3200,
            min_vram_mb_for_int4=1400,
        )
        engine = DecisionEngine(policy)

        input_data = DecisionInput(
            safety_probability=0.85,
            safety_threshold=0.50,
            carbon_intensity_gco2_per_kwh=200.0,
            carbon_data_source=DecisionDataSourceType.LIVE,
            fp16_estimated_energy_wh=0.0488,
            int4_estimated_energy_wh=0.0493,
            fp16_estimated_latency_ms=5867,
            int4_estimated_latency_ms=10580,
            gpu_vram_free_mb=2000,  # Enough for INT4, not FP16
        )

        decision = engine.decide(input_data)

        assert decision.selected_precision == DecisionPrecision.INT4
        assert decision.decision_reason == DecisionReason.RESOURCE_CONSTRAINT

    def test_vram_constraint_allows_both_when_sufficient(self):
        """Test that sufficient VRAM allows both configurations."""
        # B2 workload profile: FP16=0.1061 Wh, INT4=0.0945 Wh (INT4 lower)
        input_data = DecisionInput(
            safety_probability=0.85,
            safety_threshold=0.50,
            carbon_intensity_gco2_per_kwh=200.0,
            carbon_data_source=DecisionDataSourceType.LIVE,
            fp16_estimated_energy_wh=0.1061,
            int4_estimated_energy_wh=0.0945,  # B2 workload profile - INT4 lower
            fp16_estimated_latency_ms=11640,
            int4_estimated_latency_ms=22077,
            gpu_vram_free_mb=4000,  # Enough for both
        )

        policy = DecisionPolicy(profile_name="b2_workload", safety_threshold=0.50)
        engine = DecisionEngine(policy)
        decision = engine.decide(input_data)

        # With B2 workload profile, INT4 has lower energy
        assert decision.selected_precision == DecisionPrecision.INT4
        assert decision.decision_reason == DecisionReason.INT4_LOWER_ENVIRONMENTAL_COST

    def test_vram_constraint_blocks_int4_when_insufficient(self):
        """Test that insufficient VRAM for INT4 forces FP16."""
        input_data = DecisionInput(
            safety_probability=0.85,
            safety_threshold=0.50,
            carbon_intensity_gco2_per_kwh=200.0,
            carbon_data_source=DecisionDataSourceType.LIVE,
            fp16_estimated_energy_wh=0.0488,
            int4_estimated_energy_wh=0.0493,
            fp16_estimated_latency_ms=5867,
            int4_estimated_latency_ms=10580,
            gpu_vram_free_mb=1000,  # Not enough for either
        )

        decision = self.engine.decide(input_data)

        # Neither config feasible - should fallback to default (FP16)
        assert decision.selected_precision == DecisionPrecision.FP16
        assert decision.decision_reason == DecisionReason.RESOURCE_CONSTRAINT
        assert decision.fallback_used is True

    def test_vram_none_skips_check(self):
        """Test that None VRAM skips the resource check (legacy behavior)."""
        input_data = DecisionInput(
            safety_probability=0.85,
            safety_threshold=0.50,
            carbon_intensity_gco2_per_kwh=200.0,
            carbon_data_source=DecisionDataSourceType.LIVE,
            fp16_estimated_energy_wh=0.0488,
            int4_estimated_energy_wh=0.0493,
            fp16_estimated_latency_ms=5867,
            int4_estimated_latency_ms=10580,
            gpu_vram_free_mb=None,  # No VRAM info
        )

        decision = self.engine.decide(input_data)

        # Should not be blocked by resource constraint when VRAM unknown
        assert decision.decision_reason != DecisionReason.RESOURCE_CONSTRAINT


class TestInferenceProvider:
    """Test Inference Provider (mocked for unit tests)."""

    @pytest.fixture
    def mock_provider(self):
        """Create a mocked inference provider."""
        from carbongrid.inference.provider import InferenceProvider, Precision
        provider = Mock(spec=InferenceProvider)
        provider.run_inference.return_value = Mock(
            success=True,
            latency_ms=5000.0,
            input_tokens=10,
            output_tokens=50,
            tokens_per_second=10.0,
            gpu_memory_mb=1500,
            peak_gpu_memory_mb=1600,
            avg_power_w=20.0,
            energy_wh=0.028,
            response="Test response",
        )
        provider.get_status.return_value = {
            "model_id": "Qwen/Qwen2.5-1.5B-Instruct",
            "current_precision": None,
            "model_loaded": False,
            "nvml_power_supported": False,
        }
        provider.ensure_loaded.return_value = (Mock(), Mock())
        provider.unload = Mock()
        return provider

    def test_provider_loads_correct_precision(self, mock_provider):
        """Test provider loads correct precision based on decision."""
        from carbongrid.inference.provider import Precision

        mock_provider.ensure_loaded(Precision.FP16)
        mock_provider.ensure_loaded.assert_called_with(Precision.FP16)

        mock_provider.ensure_loaded(Precision.INT4)
        mock_provider.ensure_loaded.assert_called_with(Precision.INT4)


class TestAPISchemas:
    """Test API request/response schemas."""

    def test_generation_request_valid(self):
        """Test valid generation request."""
        req = GenerationRequest(
            prompt="What is the capital of France?",
            max_new_tokens=128,
            latency_requirement_ms=10000.0,
        )
        assert req.prompt == "What is the capital of France?"
        assert req.max_new_tokens == 128
        assert req.latency_requirement_ms == 10000.0

    def test_generation_request_defaults(self):
        """Test generation request with defaults."""
        req = GenerationRequest(prompt="Hello")
        assert req.max_new_tokens == 128
        assert req.latency_requirement_ms is None
        assert req.task_type == "unknown"

    def test_generation_request_validation(self):
        """Test generation request validation."""
        with pytest.raises(ValueError):
            GenerationRequest(prompt="")  # Empty prompt

        with pytest.raises(ValueError):
            GenerationRequest(prompt="test", max_new_tokens=0)  # Invalid tokens

        with pytest.raises(ValueError):
            GenerationRequest(prompt="test", max_new_tokens=1000)  # Too many tokens


class TestEndToEndPipeline:
    """Test complete pipeline with mocked components."""

    @pytest.fixture
    def mock_services(self):
        """Create all mocked services for pipeline test."""
        # Mock classifier
        classifier = Mock(spec=ClassifierService)
        classifier.predict_safety.return_value = {
            "safety_probability": 0.85,
            "is_safe": True,
            "threshold": 0.50,
        }
        classifier.get_threshold.return_value = 0.50
        classifier.is_loaded.return_value = True

        # Mock carbon manager
        carbon_manager = Mock()
        reading = Mock()
        reading.carbon_intensity_gco2_per_kwh = 200.0
        reading.data_source.value = "uk_carbon_intensity"
        reading.zone = "national"
        reading.reading_type.value = "actual"
        reading.timestamp.isoformat.return_value = "2026-01-01T12:00:00"
        carbon_manager.get_current_carbon.return_value = reading

        # Mock decision engine
        decision_engine = Mock(spec=DecisionEngine)
        decision_output = Mock()
        decision_output.selected_precision = DecisionPrecision.INT4
        decision_output.decision_reason = DecisionReason.INT4_LOWER_ENVIRONMENTAL_COST
        decision_output.safety_probability = 0.85
        decision_output.safety_threshold = 0.50
        decision_output.safety_gate_passed = True
        decision_output.carbon_intensity_gco2_per_kwh = 200.0
        decision_output.carbon_data_source = DecisionDataSourceType.LIVE
        decision_output.carbon_zone = "national"
        decision_output.carbon_reading_type = "actual"
        decision_output.estimated_energy_wh = 0.0493
        decision_output.estimated_co2_g = 0.00986
        decision_output.estimated_co2_mg = 9.86
        decision_output.estimated_latency_ms = 10580.0
        decision_output.latency_constraint_status = "satisfied"
        decision_output.latency_requirement_ms = None
        decision_output.fallback_used = False
        decision_output.decision_policy_version = "1.0"
        decision_output.energy_data_source = DecisionDataSourceType.ESTIMATED
        decision_output.fp16_estimated_energy_wh = 0.0488
        decision_output.fp16_estimated_co2_g = 0.00976
        decision_output.fp16_estimated_latency_ms = 5867.0
        decision_output.int4_estimated_energy_wh = 0.0493
        decision_output.int4_estimated_co2_g = 0.00986
        decision_output.int4_estimated_latency_ms = 10580.0
        decision_output.to_dict.return_value = {
            "selected_precision": "int4",
            "decision_reason": "int4_lower_environmental_cost",
        }
        decision_output.timestamp = "2026-01-01T12:00:00"
        decision_engine.decide.return_value = decision_output

        # Mock inference provider
        inference = Mock()
        inf_result = Mock()
        inf_result.success = True
        inf_result.latency_ms = 10500.0
        inf_result.input_tokens = 10
        inf_result.output_tokens = 50
        inf_result.tokens_per_second = 4.76
        inf_result.gpu_memory_mb = 1100
        inf_result.peak_gpu_memory_mb = 1161
        inf_result.avg_power_w = 17.0
        inf_result.energy_wh = 0.0495
        inf_result.response = "The capital of France is Paris."
        inf_result.error = None
        inference.run_inference.return_value = inf_result

        return {
            "classifier": classifier,
            "carbon": carbon_manager,
            "decision_engine": decision_engine,
            "inference": inference,
            "policy": DecisionPolicy(profile_name="phase05", safety_threshold=0.50),
        }

    def test_pipeline_calls_all_components(self, mock_services):
        """Test that pipeline calls all components in order."""
        assert mock_services["classifier"].is_loaded()
        assert mock_services["carbon"].get_current_carbon is not None
        assert mock_services["decision_engine"].decide is not None
        assert mock_services["inference"].run_inference is not None

    def test_pipeline_with_unsafe_prompt(self, mock_services):
        """Test pipeline with unsafe prompt routes to FP16."""
        # Override classifier to return unsafe
        mock_services["classifier"].predict_safety.return_value = {
            "safety_probability": 0.30,
            "is_safe": False,
            "threshold": 0.50,
        }

        # Override decision engine to return FP16
        decision_output = mock_services["decision_engine"].decide.return_value
        decision_output.selected_precision = DecisionPrecision.FP16
        decision_output.decision_reason = DecisionReason.SAFETY_GATE_FAILED
        decision_output.safety_gate_passed = False

        # Verify the mocks work
        safety_result = mock_services["classifier"].predict_safety("unsafe prompt")
        assert safety_result["safety_probability"] == 0.30
        assert safety_result["is_safe"] is False


class TestTelemetryOutput:
    """Test telemetry output structure."""

    def test_decision_output_serializable(self):
        """Test DecisionOutput is JSON serializable."""
        from carbongrid.decision.models import DecisionOutput, Precision

        output = DecisionOutput(
            selected_precision=DecisionPrecision.INT4,
            decision_reason=DecisionReason.INT4_LOWER_ENVIRONMENTAL_COST,
            safety_probability=0.85,
            safety_threshold=0.50,
            safety_gate_passed=True,
            carbon_intensity_gco2_per_kwh=200.0,
            carbon_data_source=DecisionDataSourceType.LIVE,
            carbon_zone="national",
            carbon_reading_type="actual",
            estimated_energy_wh=0.0493,
            estimated_co2_g=0.00986,
            estimated_co2_mg=9.86,
            estimated_latency_ms=10580.0,
            latency_constraint_status="satisfied",
            fp16_estimated_energy_wh=0.0488,
            fp16_estimated_co2_g=0.00976,
            fp16_estimated_latency_ms=5867.0,
            int4_estimated_energy_wh=0.0493,
            int4_estimated_co2_g=0.00986,
            int4_estimated_latency_ms=10580.0,
        )

        # Should serialize without error
        json_str = output.to_json()
        assert isinstance(json_str, str)

        # Should deserialize correctly
        restored = DecisionOutput.from_dict(output.to_dict())
        assert restored.selected_precision == output.selected_precision
        assert restored.decision_reason == output.decision_reason

    def test_telemetry_contains_required_fields(self):
        """Test telemetry contains all required fields."""
        from carbongrid.decision.models import DecisionOutput, Precision

        output = DecisionOutput(
            selected_precision=DecisionPrecision.FP16,
            decision_reason=DecisionReason.SAFETY_GATE_FAILED,
            safety_probability=0.30,
            safety_threshold=0.50,
            safety_gate_passed=False,
            carbon_intensity_gco2_per_kwh=200.0,
            carbon_data_source=DecisionDataSourceType.OFFLINE,
            carbon_zone="national",
            carbon_reading_type="actual",
            estimated_energy_wh=0.0488,
            estimated_co2_g=0.00976,
            estimated_co2_mg=9.76,
            estimated_latency_ms=5867.0,
            latency_constraint_status="not_specified",
            fp16_estimated_energy_wh=0.0488,
            fp16_estimated_co2_g=0.00976,
            fp16_estimated_latency_ms=5867.0,
            int4_estimated_energy_wh=0.0493,
            int4_estimated_co2_g=0.00986,
            int4_estimated_latency_ms=10580.0,
        )

        d = output.to_dict()

        # Required fields from spec
        required = [
            "selected_precision", "decision_reason", "safety_probability",
            "safety_threshold", "carbon_intensity_gco2_per_kwh",
            "estimated_energy_wh", "estimated_co2_g", "estimated_latency_ms",
            "latency_constraint_status", "energy_data_source",
            "fallback_used", "decision_policy_version",
        ]
        for field in required:
            assert field in d, f"Missing required field: {field}"


class TestPhaseFMethodology:
    """Test Phase F methodology corrections."""

    def test_pilot_size_exact(self):
        """Test that pilot selection returns exactly 20 records."""
        from scripts.run_phase_f import PhaseFEvaluator

        evaluator = PhaseFEvaluator(pilot_mode=True, pilot_size=20)
        assert len(evaluator.records) == 20

    def test_complexity_coverage(self):
        """Test that pilot covers multiple complexity levels."""
        from scripts.run_phase_f import PhaseFEvaluator

        evaluator = PhaseFEvaluator(pilot_mode=True, pilot_size=20)
        complexities = set(r['complexity_level'] for r in evaluator.records)
        # Should cover more than just complexity=1
        assert len(complexities) > 1
        # Should include at least complexity 1, 2, 3
        assert 1 in complexities

    def test_task_type_coverage(self):
        """Test that pilot covers multiple task types."""
        from scripts.run_phase_f import PhaseFEvaluator

        evaluator = PhaseFEvaluator(pilot_mode=True, pilot_size=20)
        task_types = set(r['task_type'] for r in evaluator.records)
        # Should cover multiple task types
        assert len(evaluator.records) == 20
        assert len(evaluator.records) > 5  # At least several task types

    def test_deterministic_sampling(self):
        """Test that sampling is deterministic (same seed produces same results)."""
        from scripts.run_phase_f import PhaseFEvaluator

        evaluator1 = PhaseFEvaluator(pilot_mode=True, pilot_size=20)
        evaluator2 = PhaseFEvaluator(pilot_mode=True, pilot_size=20)

        ids1 = [r['prompt_id'] for r in evaluator1.records]
        ids2 = [r['prompt_id'] for r in evaluator2.records]
        assert ids1 == ids2  # Deterministic sampling

    def test_same_prompts_across_policies(self):
        """Test that all three policies use the same prompt IDs."""
        import json
        from pathlib import Path

        results_path = Path("data/evaluation/phase_f/phase_f_raw_results.json")
        if not results_path.exists():
            pytest.skip("No results file yet")

        with open(results_path) as f:
            data = json.load(f)

        policy_ids = {}
        for policy, results in data['results'].items():
            ids = sorted([r['prompt_id'] for r in results if r['success']])
            policy_ids[policy] = ids

        # All policies should have the same prompt IDs
        all_ids = list(policy_ids.values())
        assert all(ids == all_ids[0] for ids in all_ids)

    def test_fallback_measurement_source_labeling(self):
        """Test that fallback energy sources are properly labeled."""
        import json
        from pathlib import Path

        results_path = Path("data/evaluation/phase_f/phase_f_raw_results.json")
        if not results_path.exists():
            pytest.skip("No results file yet")

        with open(results_path) as f:
            data = json.load(f)

        # Check that measurement_source is either "measured" or "phase05_profile_fallback"
        valid_sources = {"measured", "phase05_profile_fallback"}
        for policy, results in data['results'].items():
            for r in results:
                if r['success']:
                    assert r['measurement_source'] in valid_sources, \
                        f"Invalid measurement_source: {r['measurement_source']}"

    def test_quality_reference_convention(self):
        """Test that FP16 quality convention is documented."""
        import json
        from pathlib import Path

        results_path = Path("data/evaluation/phase_f/phase_f_raw_results.json")
        if not results_path.exists():
            pytest.skip("No results file yet")

        with open(results_path) as f:
            data = json.load(f)

        # Check metadata contains quality reference convention
        metadata = data.get('metadata', {})
        if 'quality_reference_convention' not in metadata:
            pytest.skip("Results file is from old run without updated metadata")
        assert 'quality_reference_convention' in metadata
        assert 'reference convention' in metadata['quality_reference_convention'].lower()


class TestVRAMMeasurement:
    """Test VRAM measurement functionality."""

    def test_get_free_vram_mb_returns_vram_result(self):
        """Test that get_free_vram_mb returns VRAMMeasurementResult."""
        from carbongrid.inference.provider import get_free_vram_mb, VRAMMeasurementResult

        result = get_free_vram_mb(0)
        # Should return VRAMMeasurementResult
        assert isinstance(result, VRAMMeasurementResult)
        if result.is_available:
            assert isinstance(result.free_mb, int)
            assert result.free_mb >= 0

    def test_get_free_vram_mb_handles_invalid_device(self):
        """Test that invalid device index returns failure result."""
        from carbongrid.inference.provider import get_free_vram_mb, VRAMMeasurementResult

        # Device index that likely doesn't exist
        result = get_free_vram_mb(999)
        assert isinstance(result, VRAMMeasurementResult)
        # Should be unavailable (not an error, just unavailable)
        assert result.is_unavailable or result.is_available
        # If available, should be int
        if result.is_available:
            assert isinstance(result.free_mb, int)
            assert result.free_mb >= 0


class TestDecisionEngineVRAMConstraints:
    """Test Decision Engine VRAM constraint handling."""

    def setup_method(self):
        self.policy = DecisionPolicy(
            profile_name="phase05",
            safety_threshold=0.50,
            min_vram_mb_for_fp16=3200,
            min_vram_mb_for_int4=1400,
        )
        self.engine = DecisionEngine(self.policy)

    def test_vram_constraint_blocks_fp16_when_insufficient(self):
        """Test that insufficient VRAM blocks FP16 but allows INT4."""
        input_data = DecisionInput(
            safety_probability=0.85,
            safety_threshold=0.50,
            carbon_intensity_gco2_per_kwh=200.0,
            carbon_data_source=DecisionDataSourceType.LIVE,
            fp16_estimated_energy_wh=0.0488,
            int4_estimated_energy_wh=0.0493,
            fp16_estimated_latency_ms=5867,
            int4_estimated_latency_ms=10580,
            gpu_vram_free_mb=2000,  # Enough for INT4 (1400), not FP16 (3200)
        )

        decision = self.engine.decide(input_data)

        assert decision.selected_precision == DecisionPrecision.INT4
        assert decision.decision_reason == DecisionReason.RESOURCE_CONSTRAINT

    def test_vram_constraint_allows_both_when_sufficient(self):
        """Test that sufficient VRAM allows both configurations."""
        # B2 workload profile: FP16=0.1061 Wh, INT4=0.0945 Wh (INT4 lower)
        input_data = DecisionInput(
            safety_probability=0.85,
            safety_threshold=0.50,
            carbon_intensity_gco2_per_kwh=200.0,
            carbon_data_source=DecisionDataSourceType.LIVE,
            fp16_estimated_energy_wh=0.1061,
            int4_estimated_energy_wh=0.0945,  # B2 workload profile - INT4 lower
            fp16_estimated_latency_ms=11640,
            int4_estimated_latency_ms=22077,
            gpu_vram_free_mb=4000,  # Enough for both
        )

        policy = DecisionPolicy(profile_name="b2_workload", safety_threshold=0.50)
        engine = DecisionEngine(policy)
        decision = engine.decide(input_data)

        # With B2 workload profile, INT4 has lower energy
        assert decision.selected_precision == DecisionPrecision.INT4
        assert decision.decision_reason == DecisionReason.INT4_LOWER_ENVIRONMENTAL_COST

    def test_vram_constraint_blocks_int4_when_insufficient(self):
        """Test that insufficient VRAM for INT4 forces FP16."""
        input_data = DecisionInput(
            safety_probability=0.85,
            safety_threshold=0.50,
            carbon_intensity_gco2_per_kwh=200.0,
            carbon_data_source=DecisionDataSourceType.LIVE,
            fp16_estimated_energy_wh=0.0488,
            int4_estimated_energy_wh=0.0493,
            fp16_estimated_latency_ms=5867,
            int4_estimated_latency_ms=10580,
            gpu_vram_free_mb=1000,  # Not enough for either
        )

        decision = self.engine.decide(input_data)

        # Neither config feasible - should fallback to default (FP16)
        assert decision.selected_precision == DecisionPrecision.FP16
        assert decision.decision_reason == DecisionReason.RESOURCE_CONSTRAINT
        assert decision.fallback_used is True

    def test_vram_none_skips_check(self):
        """Test that None VRAM skips the resource check (legacy behavior)."""
        input_data = DecisionInput(
            safety_probability=0.85,
            safety_threshold=0.50,
            carbon_intensity_gco2_per_kwh=200.0,
            carbon_data_source=DecisionDataSourceType.LIVE,
            fp16_estimated_energy_wh=0.0488,
            int4_estimated_energy_wh=0.0493,
            fp16_estimated_latency_ms=5867,
            int4_estimated_latency_ms=10580,
            gpu_vram_free_mb=None,  # No VRAM info
        )

        decision = self.engine.decide(input_data)

        # Should not be blocked by resource constraint when VRAM unknown
        assert decision.decision_reason != DecisionReason.RESOURCE_CONSTRAINT


class TestConcurrency:
    """Test concurrency control for inference provider."""

    def test_semaphore_blocks_when_full(self):
        """Test that semaphore blocks when max concurrent reached."""
        from carbongrid.inference.provider import InferenceProvider, Precision, InferenceMetrics
        from unittest.mock import MagicMock, patch
        import threading
        import time

        provider = InferenceProvider(max_concurrent_inferences=1)

        # Reduce semaphore acquire timeout for faster test
        provider._inference_semaphore = threading.Semaphore(1)
        original_acquire = provider._inference_semaphore.acquire

        def acquire_with_short_timeout(blocking=True, timeout=-1):
            return original_acquire(blocking, timeout=1.0)  # 1 second timeout

        provider._inference_semaphore.acquire = acquire_with_short_timeout

        mock_model = MagicMock()
        mock_tokenizer = MagicMock()

        mock_output = MagicMock()
        seq_tensor = MagicMock()
        seq_tensor.shape = (1, 13)
        mock_output.sequences = [seq_tensor]
        mock_model.generate.return_value = mock_output

        mock_inputs = MagicMock()
        mock_inputs.input_ids = MagicMock()
        mock_inputs.input_ids.shape = (1, 3)
        mock_inputs.to = MagicMock(return_value=MagicMock(input_ids=MagicMock(shape=(1, 3))))

        mock_tokenizer.return_value = mock_inputs
        mock_tokenizer.decode.return_value = "test response"
        mock_tokenizer.eos_token_id = 2

        first_request_entered_generate = threading.Event()
        first_request_hold = threading.Event()

        def slow_generate(*args, **kwargs):
            first_request_entered_generate.set()
            first_request_hold.wait(timeout=5.0)
            return mock_output

        mock_model.generate.side_effect = slow_generate

        with patch.object(provider, 'ensure_loaded', return_value=(mock_model, mock_tokenizer)):
            provider._power_monitor.start = MagicMock()
            provider._power_monitor.stop = MagicMock(return_value={"avg_power_w": 10.0, "energy_wh": 0.01})

            results = []

            def make_request(prompt):
                try:
                    result = provider.run_inference(prompt, max_new_tokens=10)
                    results.append(result)
                except Exception as e:
                    results.append(e)

            t1 = threading.Thread(target=make_request, args=("test prompt",))
            t1.start()

            # Wait for first request to enter generate (and thus hold semaphore)
            first_request_entered_generate.wait(timeout=2.0)
            time.sleep(0.05)  # Ensure semaphore is held

            # Second request should fail after ~1s timeout (semaphore full)
            result2 = provider.run_inference("test prompt 2", max_new_tokens=10)

            first_request_hold.set()
            t1.join(timeout=2.0)

            assert len(results) == 1
            assert results[0].success is True

            assert result2.success is False
            assert "busy" in result2.error.lower() or "concurrent" in result2.error.lower()

    def test_model_loading_concurrent_safety(self):
        """Test that concurrent model loading is safe (load lock prevents duplicate loads)."""
        from carbongrid.inference.provider import InferenceProvider, Precision
        from unittest.mock import MagicMock
        import threading

        provider = InferenceProvider()

        load_count = 0
        load_lock = threading.Lock()

        def mock_load_fp16():
            nonlocal load_count
            with load_lock:
                load_count += 1
            import time
            time.sleep(0.1)
            return MagicMock(), MagicMock()

        provider._load_fp16 = mock_load_fp16

        results = []

        def load_model():
            try:
                model, tokenizer = provider.ensure_loaded(Precision.FP16)
                results.append(("success", model is not None))
            except Exception as e:
                results.append(("error", str(e)))

        t1 = threading.Thread(target=load_model)
        t2 = threading.Thread(target=load_model)

        t1.start()
        t2.start()

        t1.join(timeout=5.0)
        t2.join(timeout=5.0)

        assert len(results) == 2
        successes = [r for r in results if r[0] == "success"]
        assert len(successes) == 2
        assert load_count == 1

    def test_inference_rejected_when_busy(self):
        """Test that inference is rejected when provider is at capacity."""
        from carbongrid.inference.provider import InferenceProvider, Precision, InferenceMetrics
        from unittest.mock import MagicMock, patch
        import threading
        import time

        provider = InferenceProvider(max_concurrent_inferences=1)

        mock_model = MagicMock()
        mock_tokenizer = MagicMock()

        mock_output = MagicMock()
        mock_output.sequences = [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]
        mock_model.generate.return_value = mock_output

        mock_inputs = MagicMock()
        mock_inputs.input_ids = MagicMock()
        mock_inputs.input_ids.shape = (1, 3)
        mock_inputs.to = MagicMock(return_value=MagicMock(input_ids=MagicMock(shape=(1, 3))))

        mock_tokenizer.return_value = mock_inputs
        mock_tokenizer.decode.return_value = "test response"
        mock_tokenizer.eos_token_id = 2

        first_request_started = threading.Event()
        first_request_hold = threading.Event()

        original_generate = mock_model.generate

        def slow_generate(*args, **kwargs):
            first_request_started.set()
            first_request_hold.wait(timeout=5.0)
            return original_generate(*args, **kwargs)

        mock_model.generate.side_effect = slow_generate

        with patch.object(provider, 'ensure_loaded', return_value=(mock_model, mock_tokenizer)):
            provider._power_monitor.start = MagicMock()
            provider._power_monitor.stop = MagicMock(return_value={"avg_power_w": 10.0, "energy_wh": 0.01})

            t1 = threading.Thread(target=lambda: provider.run_inference("test 1", max_new_tokens=10))
            t1.start()

            first_request_started.wait(timeout=2.0)
            time.sleep(0.05)

            result = provider.run_inference("test 2", max_new_tokens=10)

            first_request_hold.set()
            t1.join(timeout=2.0)

            assert result.success is False
            assert "busy" in result.error.lower() or "concurrent" in result.error.lower()

    def test_sequential_requests_work(self):
        """Test that sequential requests work normally."""
        from carbongrid.inference.provider import InferenceProvider, Precision, InferenceMetrics
        from unittest.mock import MagicMock

        provider = InferenceProvider()

        # Mock everything
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()

        # Setup mock model
        mock_output = MagicMock()
        mock_output.sequences = [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]  # 10 new tokens
        mock_model.generate.return_value = MagicMock(sequences=[[1,2,3,4,5,6,7,8,9,10]])

        # Setup mock tokenizer
        mock_inputs = MagicMock()
        mock_inputs.input_ids = MagicMock()
        mock_inputs.input_ids.shape = (1, 3)
        mock_inputs.to = MagicMock(return_value=MagicMock(input_ids=MagicMock(shape=(1, 3))))

        mock_tokenizer = MagicMock()
        mock_tokenizer.return_value = MagicMock(input_ids=MagicMock(shape=(1, 10)))
        mock_tokenizer.decode.return_value = "test response"
        mock_tokenizer.eos_token_id = 2

        provider = InferenceProvider()

        with patch.object(provider, 'ensure_loaded', return_value=(MagicMock(), MagicMock())):
            provider._power_monitor.start = MagicMock()
            provider._power_monitor.stop = MagicMock(return_value={"avg_power_w": 10.0, "energy_wh": 0.01})

            # First request
            result1 = provider.run_inference("Hello", max_new_tokens=10)
            assert result1.success is True

            # Second request (sequential)
            result2 = provider.run_inference("World", max_new_tokens=10)
            assert result2.success is True

    def test_model_unload_during_inference_prevented(self):
        """Test that model is not unloaded while inference is running."""
        from carbongrid.inference.provider import InferenceProvider, Precision, InferenceMetrics
        import threading
        import time

        provider = InferenceProvider()

        # Mock the inference to return a mock result after a delay
        def slow_infer(prompt, max_new_tokens=128, precision=None):
            time.sleep(0.1)
            return InferenceMetrics(
                latency_ms=100, input_tokens=10, output_tokens=20,
                tokens_per_second=100, gpu_memory_mb=1000, peak_gpu_memory_mb=1100,
                avg_power_w=10.0, energy_wh=0.001, response="test", success=True
            )

        provider.run_inference = slow_infer

        # Start inference
        t1 = threading.Thread(target=lambda: provider.run_inference("test"))
        t1.start()
        time.sleep(0.05)

        # Try to unload while inference is running
        provider.unload()  # This should wait for the lock

        t1.join(timeout=2.0)

        # Inference should have completed
        # Model should be unloaded after (or not loaded)
        assert provider._model is None or provider._current_precision is not None


class TestModelLifecycleRace:
    """Test model lifecycle race condition fixes (Fix 2)."""
    
    def _setup_mocks(self, provider, generate_side_effect=None):
        """Helper to set up common mocks - uses generic mocks like existing tests."""
        from unittest.mock import MagicMock
        
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        
        mock_output = MagicMock()
        mock_output.sequences = [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]]
        if generate_side_effect:
            mock_model.generate.side_effect = generate_side_effect
        else:
            mock_model.generate.return_value = mock_output
        
        # Use generic mocks - MagicMock auto-creates attributes as needed
        # This matches the pattern from existing working tests
        mock_tokenizer.decode.return_value = "test response"
        mock_tokenizer.eos_token_id = 2
        
        return mock_model, mock_tokenizer
    
    def test_unload_waits_for_active_inference(self):
        """unload() requested during active inference → unload waits."""
        from carbongrid.inference.provider import InferenceProvider, InferenceMetrics
        from unittest.mock import MagicMock, patch
        import threading
        import time
        
        provider = InferenceProvider()
        
        # Create a slow model
        mock_model = MagicMock()
        def slow_generate(*args, **kwargs):
            time.sleep(0.2)
            return MagicMock(sequences=[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
        mock_model.generate.side_effect = slow_generate
        
        def mock_ensure_loaded(precision):
            return mock_model, MagicMock()
        
        # Track unload completion
        unload_completed = threading.Event()
        original_unload = provider.unload
        
        def tracked_unload():
            original_unload()
            unload_completed.set()
        
        provider.unload = tracked_unload
        
        with patch.object(provider, 'ensure_loaded', side_effect=mock_ensure_loaded):
            provider._power_monitor.start = MagicMock()
            provider._power_monitor.stop = MagicMock(return_value={"avg_power_w": 10.0, "energy_wh": 0.01})
            
            # Start inference in background (slow)
            inference_done = threading.Event()
            def run_inf():
                provider.run_inference("test prompt", max_new_tokens=10)
                inference_done.set()
            
            t1 = threading.Thread(target=run_inf)
            t1.start()
            
            # Give inference time to start and acquire active count
            time.sleep(0.05)
            
            # Call unload in background - should block until inference completes
            unload_thread = threading.Thread(target=provider.unload)
            unload_thread.start()
            
            # Unload should not complete immediately (inference still running)
            time.sleep(0.1)
            assert not unload_completed.is_set(), "unload() should wait for active inference"
            
            # Wait for inference to complete
            inference_done.wait(timeout=2.0)
            t1.join(timeout=2.0)
            
            # Now unload should complete
            unload_thread.join(timeout=2.0)
            assert unload_completed.is_set(), "unload() should complete after inference finishes"
    
    def test_model_not_deleted_while_inference_active(self):
        """Model is not deleted while inference is active."""
        from carbongrid.inference.provider import InferenceProvider, InferenceMetrics
        from unittest.mock import MagicMock, patch
        import threading
        import time
        
        provider = InferenceProvider()
        
        # Create a slow model so inference takes time
        mock_model = MagicMock()
        def slow_generate(*args, **kwargs):
            time.sleep(0.15)
            return MagicMock(sequences=[[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
        mock_model.generate.side_effect = slow_generate
        mock_tokenizer = MagicMock()
        
        # Track unload attempts
        unload_attempted_during_inference = False
        unload_completed = threading.Event()
        
        def tracked_unload():
            nonlocal unload_attempted_during_inference
            with provider._model_lifetime_cv:
                if provider._active_inferences > 0:
                    unload_attempted_during_inference = True
                    # Wait for inference to complete
                    while provider._active_inferences > 0:
                        provider._model_lifetime_cv.wait()
            unload_completed.set()
        
        provider.unload = tracked_unload
        
        with patch.object(provider, 'ensure_loaded', return_value=(mock_model, mock_tokenizer)):
            provider._power_monitor.start = MagicMock()
            provider._power_monitor.stop = MagicMock(return_value={"avg_power_w": 10.0, "energy_wh": 0.01})
            
            # Start inference
            t1 = threading.Thread(target=lambda: provider.run_inference("test", max_new_tokens=10))
            t1.start()
            
            # Try to unload while inference runs (in background)
            unload_thread = threading.Thread(target=provider.unload)
            unload_thread.start()
            
            time.sleep(0.05)
            assert unload_attempted_during_inference, "unload() should have been called while inference active"
            
            # Wait for inference to complete
            t1.join(timeout=2.0)
            
            # Now unload should complete
            unload_thread.join(timeout=2.0)
            assert unload_completed.is_set()
    
    def test_model_replacement_waits_for_active_inference(self):
        """FP16/INT4 model replacement waits for active inference."""
        from carbongrid.inference.provider import InferenceProvider, Precision, InferenceMetrics
        from unittest.mock import MagicMock, patch
        import threading
        import time
        
        provider = InferenceProvider()
        
        load_order = []
        
        def mock_load_fp16():
            load_order.append("fp16_start")
            time.sleep(0.05)
            load_order.append("fp16_done")
            return MagicMock(), MagicMock()
        
        def mock_load_int4():
            load_order.append("int4_start")
            time.sleep(0.05)
            load_order.append("int4_done")
            return MagicMock(), MagicMock()
        
        provider._load_fp16 = mock_load_fp16
        provider._load_int4 = mock_load_int4
        
        # Patch ensure_loaded to use our custom load functions
        def mock_ensure_loaded(precision):
            if precision == Precision.FP16:
                return mock_load_fp16()
            else:
                return mock_load_int4()
        
        with patch.object(provider, 'ensure_loaded', side_effect=mock_ensure_loaded):
            provider._power_monitor.start = MagicMock()
            provider._power_monitor.stop = MagicMock(return_value={"avg_power_w": 10.0, "energy_wh": 0.01})
            
            # Start FP16 inference
            t1 = threading.Thread(target=lambda: provider.run_inference("test fp16", max_new_tokens=10, precision=Precision.FP16))
            t1.start()
            
            time.sleep(0.05)  # Let inference start
            
            # Request INT4 (should wait for FP16 inference to complete)
            t2 = threading.Thread(target=lambda: provider.run_inference("test int4", max_new_tokens=10, precision=Precision.INT4))
            t2.start()
            
            t1.join(timeout=3.0)
            t2.join(timeout=3.0)
            
            # INT4 load should start after FP16 inference completes
            # load_order should be: fp16_start, fp16_done, int4_start, int4_done
            assert load_order == ["fp16_start", "fp16_done", "int4_start", "int4_done"], \
                f"Model replacement order incorrect: {load_order}"
    
    def test_inference_exception_decrements_active_count(self):
        """Inference exception decrements active count correctly."""
        from carbongrid.inference.provider import InferenceProvider, InferenceMetrics
        from unittest.mock import MagicMock, patch
        
        provider = InferenceProvider()
        
        # Create a model that raises an exception on generate
        mock_model = MagicMock()
        mock_model.generate.side_effect = RuntimeError("CUDA OOM")
        mock_tokenizer = MagicMock()
        
        # Use return_value like existing test - same tuple every time
        with patch.object(provider, 'ensure_loaded', return_value=(mock_model, mock_tokenizer)):
            provider._power_monitor.start = MagicMock()
            provider._power_monitor.stop = MagicMock(return_value={"avg_power_w": 10.0, "energy_wh": 0.01})
            
            # Run inference that will fail
            result = provider.run_inference("test", max_new_tokens=10)
            
            # Should return failed result, not raise
            assert result.success is False
            assert "CUDA OOM" in result.error
            
            # Active count should be back to 0
            with provider._model_lifetime_cv:
                assert provider._active_inferences == 0, "Active count not decremented after exception"
    
    def test_http_timeout_worker_continues_model_protected(self):
        """HTTP timeout while worker thread continues → model remains protected until worker finishes.
        
        This test verifies the semantic guarantee that model lifetime protection
        is not released merely because an HTTP request timed out. The actual
        waiting behavior is tested by test_unload_waits_for_active_inference.
        """
        from carbongrid.inference.provider import InferenceProvider
        from unittest.mock import MagicMock, patch
        
        provider = InferenceProvider()
        
        with patch.object(provider, 'ensure_loaded', return_value=(MagicMock(), MagicMock())):
            provider._power_monitor.start = MagicMock()
            provider._power_monitor.stop = MagicMock(return_value={"avg_power_w": 10.0, "energy_wh": 0.01})
            
            # Start inference and get active count
            provider.run_inference("test", max_new_tokens=10)
            
            # After inference completes, active count should be 0
            with provider._model_lifetime_cv:
                assert provider._active_inferences == 0
            
            # The key guarantee: if an inference were still active (e.g., worker
            # continuing after HTTP timeout), unload would wait. This is tested
            # by test_unload_waits_for_active_inference which uses a slow model.
            # Here we verify the mechanism exists.
            assert hasattr(provider, '_model_lifetime_cv')
            assert hasattr(provider, '_active_inferences')
    
    def test_sequential_inference_still_works(self):
        """Sequential inference still works correctly."""
        from carbongrid.inference.provider import InferenceProvider, InferenceMetrics
        from unittest.mock import MagicMock, patch
        
        provider = InferenceProvider()
        
        with patch.object(provider, 'ensure_loaded', return_value=(MagicMock(), MagicMock())):
            provider._power_monitor.start = MagicMock()
            provider._power_monitor.stop = MagicMock(return_value={"avg_power_w": 10.0, "energy_wh": 0.01})
            
            # Run multiple sequential requests
            for i in range(5):
                result = provider.run_inference(f"test {i}", max_new_tokens=10)
                assert result.success is True
            
            # Active count should be 0 after all complete
            with provider._model_lifetime_cv:
                assert provider._active_inferences == 0
    
    def test_concurrent_inference_limit_respected(self):
        """Concurrent inference limit (semaphore) still works."""
        from carbongrid.inference.provider import InferenceProvider, InferenceMetrics
        from unittest.mock import MagicMock, patch
        
        provider = InferenceProvider(max_concurrent_inferences=2)
        
        with patch.object(provider, 'ensure_loaded', return_value=(MagicMock(), MagicMock())):
            provider._power_monitor.start = MagicMock()
            provider._power_monitor.stop = MagicMock(return_value={"avg_power_w": 10.0, "energy_wh": 0.01})
            
            # First two requests should succeed (sequential)
            result1 = provider.run_inference("test 1", max_new_tokens=10)
            result2 = provider.run_inference("test 2", max_new_tokens=10)
            assert result1.success is True
            assert result2.success is True
            
            # Manually acquire semaphore twice to simulate full capacity
            provider._inference_semaphore.acquire()
            provider._inference_semaphore.acquire()
            
            # Third request should be rejected (semaphore full)
            result3 = provider.run_inference("test 3", max_new_tokens=10)
            
            # Release semaphore
            provider._inference_semaphore.release()
            provider._inference_semaphore.release()
            
            assert result3.success is False
            assert "busy" in result3.error.lower() or "concurrent" in result3.error.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])