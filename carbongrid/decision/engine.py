"""
Decision Engine for CarbonGrid-AI.

Core decision logic that combines:
1. Quantization safety (from classifier)
2. Latency constraints
3. Energy/CO2 estimates
4. Carbon intensity context
5. Resource availability

The Decision Engine is independent of FastAPI and model execution.
"""

from dataclasses import dataclass
from typing import Optional, Tuple
from datetime import datetime, timezone

from carbongrid.decision.models import (
    DecisionInput,
    DecisionOutput,
    Precision,
    DataSourceType,
    DecisionReason,
    FallbackPolicy,
    estimate_co2_from_energy,
)
from carbongrid.decision.policy import DecisionPolicy, DEFAULT_POLICY


class DecisionEngine:
    """
    Core decision engine for CarbonGrid-AI.
    
    Makes deterministic, explainable decisions about inference configuration.
    
    Decision flow:
    1. Safety gate: if safety_probability < threshold -> FP16
    2. Resource feasibility: check VRAM constraints
    3. Latency constraints: if specified, filter infeasible configs
    4. Environmental cost: compare CO2 of feasible configs
    5. Tie-break: latency, then default policy
    """
    
    def __init__(self, policy: Optional[DecisionPolicy] = None):
        self.policy = policy or DEFAULT_POLICY
    
    def decide(self, input_data: DecisionInput) -> DecisionOutput:
        """
        Make a configuration decision.
        
        Args:
            input_data: Structured decision input
            
        Returns:
            DecisionOutput with selected configuration and full reasoning
        """
        # 0. INPUT VALIDATION - Fail-closed for invalid safety probability
        safety_prob = input_data.safety_probability
        if (safety_prob is None or not isinstance(safety_prob, (int, float)) 
                or safety_prob != safety_prob  # NaN check
                or safety_prob == float('inf') 
                or safety_prob == float('-inf')
                or safety_prob < 0.0 
                or safety_prob > 1.0):
            return self._create_decision(
                input_data=input_data,
                selected_precision=Precision.FP16,
                decision_reason=DecisionReason.CLASSIFIER_ERROR,
                safety_gate_passed=False,
                fallback_used=False,
            )
        
        # 1. SAFETY GATE - Hard constraint
        safety_gate_passed = safety_prob >= input_data.safety_threshold
        
        if not safety_gate_passed:
            return self._create_decision(
                input_data=input_data,
                selected_precision=Precision.FP16,
                decision_reason=DecisionReason.SAFETY_GATE_FAILED,
                safety_gate_passed=False,
                fallback_used=False,
            )
        
        # 2. RESOURCE FEASIBILITY
        resource_feasible = self._check_resource_feasibility(input_data)
        fp16_feasible = resource_feasible["fp16"]
        int4_feasible = resource_feasible["int4"]
        fp16_blocked_by_resource = not fp16_feasible
        int4_blocked_by_resource = not int4_feasible
        
        if not fp16_feasible and not int4_feasible:
            # Neither configuration is feasible - use fallback
            return self._create_decision(
                input_data=input_data,
                selected_precision=self.policy.default_precision,
                decision_reason=DecisionReason.RESOURCE_CONSTRAINT,
                safety_gate_passed=True,
                fallback_used=True,
            )
        
        # 3. LATENCY CONSTRAINTS
        latency_status = self._check_latency_constraints(input_data)
        fp16_latency_ok = latency_status["fp16_ok"]
        int4_latency_ok = latency_status["int4_ok"]
        latency_constraint_status = latency_status["status"]
        
        # Track why each config is infeasible
        fp16_blocked_by_latency = not fp16_latency_ok
        int4_blocked_by_latency = not int4_latency_ok
        
        # Filter by latency feasibility
        if not fp16_latency_ok:
            fp16_feasible = False
        if not int4_latency_ok:
            int4_feasible = False
        
        # 4. DECISION BASED ON FEASIBLE CONFIGURATIONS
        if fp16_feasible and int4_feasible:
            # Both feasible - compare environmental cost
            return self._compare_environmental_cost(input_data, safety_gate_passed, latency_constraint_status)
        elif fp16_feasible:
            # Only FP16 feasible
            if int4_blocked_by_resource:
                reason = DecisionReason.RESOURCE_CONSTRAINT
            elif int4_blocked_by_latency:
                reason = DecisionReason.INT4_LATENCY_VIOLATION
            else:
                reason = DecisionReason.FP16_LOWER_ENVIRONMENTAL_COST
            return self._create_decision(
                input_data=input_data,
                selected_precision=Precision.FP16,
                decision_reason=reason,
                safety_gate_passed=True,
                fallback_used=not int4_feasible,
                latency_constraint_status=latency_constraint_status,
            )
        elif int4_feasible:
            # Only INT4 feasible
            if fp16_blocked_by_resource:
                reason = DecisionReason.RESOURCE_CONSTRAINT
            elif fp16_blocked_by_latency:
                reason = DecisionReason.FP16_LATENCY_VIOLATION
            else:
                reason = DecisionReason.INT4_LOWER_ENVIRONMENTAL_COST
            return self._create_decision(
                input_data=input_data,
                selected_precision=Precision.INT4,
                decision_reason=reason,
                safety_gate_passed=True,
                fallback_used=not fp16_feasible,
                latency_constraint_status=latency_constraint_status,
            )
        else:
            # Neither feasible after latency filtering
            return self._handle_both_infeasible(input_data, safety_gate_passed, latency_constraint_status)
    
    def _check_resource_feasibility(self, input_data: DecisionInput) -> dict:
        """Check if configurations are feasible given resource constraints."""
        fp16_feasible = True
        int4_feasible = True
        
        vram = input_data.gpu_vram_free_mb
        vram_available = input_data.vram_telemetry_available
        
        # If VRAM telemetry was not attempted (legacy caller), skip VRAM check
        if vram is None and not vram_available:
            # Legacy behavior: skip VRAM check when not provided
            return {"fp16": True, "int4": True}
        
        # VRAM telemetry was attempted (vram_available=True) but measurement failed
        # or VRAM value is invalid (NaN, infinite, negative) -> fail closed
        vram_unavailable = (
            vram is None
            or not isinstance(vram, (int, float))
            or vram != vram  # NaN check
            or vram == float('inf')
            or vram == float('-inf')
            or vram < 0
        )
        
        if not vram_unavailable:
            fp16_feasible = vram >= self.policy.min_vram_mb_for_fp16
            int4_feasible = vram >= self.policy.min_vram_mb_for_int4
        else:
            # Unavailable/invalid VRAM -> neither configuration is feasible (fail closed)
            fp16_feasible = False
            int4_feasible = False
        
        return {"fp16": fp16_feasible, "int4": int4_feasible}
    
    def _check_latency_constraints(self, input_data: DecisionInput) -> dict:
        """Check latency constraints for both configurations."""
        if input_data.latency_requirement_ms is None:
            return {
                "fp16_ok": True,
                "int4_ok": True,
                "status": "not_specified",
                "requirement_ms": None,
            }
        
        requirement = input_data.latency_requirement_ms
        tolerance = requirement * self.policy.latency_tolerance_pct
        effective_requirement = requirement + tolerance
        
        # Use input latency estimates if provided, otherwise policy defaults
        fp16_latency = input_data.fp16_estimated_latency_ms if input_data.fp16_estimated_latency_ms > 0 else self.policy.get_fp16_latency_ms()
        int4_latency = input_data.int4_estimated_latency_ms if input_data.int4_estimated_latency_ms > 0 else self.policy.get_int4_latency_ms()
        
        fp16_ok = fp16_latency <= effective_requirement
        int4_ok = int4_latency <= effective_requirement
        
        if fp16_ok and int4_ok:
            status = "satisfied"
        elif not fp16_ok and not int4_ok:
            status = "both_violated"
        elif fp16_ok:
            status = "int4_violated"
        else:
            status = "fp16_violated"
        
        return {
            "fp16_ok": fp16_ok,
            "int4_ok": int4_ok,
            "status": status,
            "requirement_ms": requirement,
            "effective_requirement_ms": effective_requirement,
        }
    
    def _compare_environmental_cost(
        self,
        input_data: DecisionInput,
        safety_gate_passed: bool,
        latency_constraint_status: str,
    ) -> DecisionOutput:
        """Compare CO2 emissions of both feasible configurations."""
        # Get energy estimates from policy (which uses profile)
        fp16_energy_wh = self.policy.get_fp16_energy_wh()
        int4_energy_wh = self.policy.get_int4_energy_wh()
        
        # Use input estimates if provided (override profile)
        if input_data.fp16_estimated_energy_wh > 0:
            fp16_energy_wh = input_data.fp16_estimated_energy_wh
        if input_data.int4_estimated_energy_wh > 0:
            int4_energy_wh = input_data.int4_estimated_energy_wh
        
        # Calculate CO2
        fp16_co2_g, fp16_co2_mg = estimate_co2_from_energy(
            fp16_energy_wh, input_data.carbon_intensity_gco2_per_kwh
        )
        int4_co2_g, int4_co2_mg = estimate_co2_from_energy(
            int4_energy_wh, input_data.carbon_intensity_gco2_per_kwh
        )
        
        # Compare with threshold
        co2_diff_mg = abs(fp16_co2_mg - int4_co2_mg)
        co2_diff_mg = round(co2_diff_mg, 6)
        
        if co2_diff_mg < self.policy.co2_difference_threshold_mg:
            # Effectively equal - tie-break by latency
            fp16_latency = self.policy.get_fp16_latency_ms()
            int4_latency = self.policy.get_int4_latency_ms()
            
            if fp16_latency <= int4_latency:
                return self._create_decision(
                    input_data=input_data,
                    selected_precision=Precision.FP16,
                    decision_reason=DecisionReason.ENVIRONMENTAL_COST_EQUAL,
                    safety_gate_passed=safety_gate_passed,
                    fallback_used=False,
                    fp16_energy_wh=fp16_energy_wh,
                    fp16_co2_g=fp16_co2_g,
                    fp16_co2_mg=fp16_co2_mg,
                    int4_energy_wh=int4_energy_wh,
                    int4_co2_g=int4_co2_g,
                    int4_co2_mg=int4_co2_mg,
                    latency_constraint_status=latency_constraint_status,
                )
            else:
                return self._create_decision(
                    input_data=input_data,
                    selected_precision=Precision.INT4,
                    decision_reason=DecisionReason.ENVIRONMENTAL_COST_EQUAL,
                    safety_gate_passed=safety_gate_passed,
                    fallback_used=False,
                    fp16_energy_wh=fp16_energy_wh,
                    fp16_co2_g=fp16_co2_g,
                    fp16_co2_mg=fp16_co2_mg,
                    int4_energy_wh=int4_energy_wh,
                    int4_co2_g=int4_co2_g,
                    int4_co2_mg=int4_co2_mg,
                    latency_constraint_status=latency_constraint_status,
                )
        
        # Significant difference - choose lower CO2
        if int4_co2_mg < fp16_co2_mg:
            return self._create_decision(
                input_data=input_data,
                selected_precision=Precision.INT4,
                decision_reason=DecisionReason.INT4_LOWER_ENVIRONMENTAL_COST,
                safety_gate_passed=safety_gate_passed,
                fallback_used=False,
                fp16_energy_wh=fp16_energy_wh,
                fp16_co2_g=fp16_co2_g,
                fp16_co2_mg=fp16_co2_mg,
                int4_energy_wh=int4_energy_wh,
                int4_co2_g=int4_co2_g,
                int4_co2_mg=int4_co2_mg,
                latency_constraint_status=latency_constraint_status,
            )
        else:
            return self._create_decision(
                input_data=input_data,
                selected_precision=Precision.FP16,
                decision_reason=DecisionReason.FP16_LOWER_ENVIRONMENTAL_COST,
                safety_gate_passed=safety_gate_passed,
                fallback_used=False,
                fp16_energy_wh=fp16_energy_wh,
                fp16_co2_g=fp16_co2_g,
                fp16_co2_mg=fp16_co2_mg,
                int4_energy_wh=int4_energy_wh,
                int4_co2_g=int4_co2_g,
                int4_co2_mg=int4_co2_mg,
                latency_constraint_status=latency_constraint_status,
            )
    
    def _handle_both_infeasible(
        self,
        input_data: DecisionInput,
        safety_gate_passed: bool,
        latency_constraint_status: str,
    ) -> DecisionOutput:
        """Handle case where both configurations violate latency."""
        fallback = self.policy.fallback_policy
        
        if fallback == FallbackPolicy.PREFER_FP16:
            selected = Precision.FP16
            reason = DecisionReason.FALLBACK_POLICY
        elif fallback == FallbackPolicy.PREFER_LOWER_LATENCY:
            fp16_latency = self.policy.get_fp16_latency_ms()
            int4_latency = self.policy.get_int4_latency_ms()
            selected = Precision.FP16 if fp16_latency <= int4_latency else Precision.INT4
            reason = DecisionReason.FALLBACK_POLICY
        elif fallback == FallbackPolicy.PREFER_LOWER_ENERGY:
            fp16_energy = self.policy.get_fp16_energy_wh()
            int4_energy = self.policy.get_int4_energy_wh()
            selected = Precision.FP16 if fp16_energy <= int4_energy else Precision.INT4
            reason = DecisionReason.FALLBACK_POLICY
        else:  # REJECT_REQUEST
            selected = Precision.FP16  # Still must return something
            reason = DecisionReason.FALLBACK_POLICY
        
        return self._create_decision(
            input_data=input_data,
            selected_precision=selected,
            decision_reason=reason,
            safety_gate_passed=safety_gate_passed,
            fallback_used=True,
            latency_constraint_status=latency_constraint_status,
        )
    
    def _create_decision(
        self,
        input_data: DecisionInput,
        selected_precision: Precision,
        decision_reason: DecisionReason,
        safety_gate_passed: bool,
        fallback_used: bool,
        fp16_energy_wh: Optional[float] = None,
        fp16_co2_g: Optional[float] = None,
        fp16_co2_mg: Optional[float] = None,
        int4_energy_wh: Optional[float] = None,
        int4_co2_g: Optional[float] = None,
        int4_co2_mg: Optional[float] = None,
        latency_constraint_status: str = "not_specified",
    ) -> DecisionOutput:
        """Create the final DecisionOutput object."""
        
        # Get energy/latency from policy if not provided
        if fp16_energy_wh is None:
            fp16_energy_wh = self.policy.get_fp16_energy_wh()
        if int4_energy_wh is None:
            int4_energy_wh = self.policy.get_int4_energy_wh()
        
        # Override with input if provided
        if input_data.fp16_estimated_energy_wh > 0:
            fp16_energy_wh = input_data.fp16_estimated_energy_wh
        if input_data.int4_estimated_energy_wh > 0:
            int4_energy_wh = input_data.int4_estimated_energy_wh
        
        # Calculate CO2 for both
        fp16_co2_g_calc, fp16_co2_mg_calc = estimate_co2_from_energy(
            fp16_energy_wh, input_data.carbon_intensity_gco2_per_kwh
        )
        int4_co2_g_calc, int4_co2_mg_calc = estimate_co2_from_energy(
            int4_energy_wh, input_data.carbon_intensity_gco2_per_kwh
        )
        
        if fp16_co2_g is None:
            fp16_co2_g = fp16_co2_g_calc
        if fp16_co2_mg is None:
            fp16_co2_mg = fp16_co2_mg_calc
        if int4_co2_g is None:
            int4_co2_g = int4_co2_g_calc
        if int4_co2_mg is None:
            int4_co2_mg = int4_co2_mg_calc
        
        # Get latencies
        fp16_latency_ms = self.policy.get_fp16_latency_ms()
        int4_latency_ms = self.policy.get_int4_latency_ms()
        
        if input_data.fp16_estimated_latency_ms > 0:
            fp16_latency_ms = input_data.fp16_estimated_latency_ms
        if input_data.int4_estimated_latency_ms > 0:
            int4_latency_ms = input_data.int4_estimated_latency_ms
        
        # Select values for chosen configuration
        if selected_precision == Precision.FP16:
            estimated_energy_wh = fp16_energy_wh
            estimated_co2_g = fp16_co2_g
            estimated_co2_mg = fp16_co2_mg
            estimated_latency_ms = fp16_latency_ms
        else:
            estimated_energy_wh = int4_energy_wh
            estimated_co2_g = int4_co2_g
            estimated_co2_mg = int4_co2_mg
            estimated_latency_ms = int4_latency_ms
        
        return DecisionOutput(
            selected_precision=selected_precision,
            decision_reason=decision_reason,
            safety_probability=input_data.safety_probability,
            safety_threshold=input_data.safety_threshold,
            safety_gate_passed=safety_gate_passed,
            carbon_intensity_gco2_per_kwh=input_data.carbon_intensity_gco2_per_kwh,
            carbon_data_source=input_data.carbon_data_source,
            carbon_zone=input_data.carbon_zone,
            carbon_reading_type=input_data.carbon_reading_type,
            estimated_energy_wh=estimated_energy_wh,
            estimated_co2_g=estimated_co2_g,
            estimated_co2_mg=estimated_co2_mg,
            estimated_latency_ms=estimated_latency_ms,
            latency_constraint_status=latency_constraint_status,
            latency_requirement_ms=input_data.latency_requirement_ms,
            fp16_estimated_energy_wh=fp16_energy_wh,
            fp16_estimated_co2_g=fp16_co2_g,
            fp16_estimated_latency_ms=fp16_latency_ms,
            int4_estimated_energy_wh=int4_energy_wh,
            int4_estimated_co2_g=int4_co2_g,
            int4_estimated_latency_ms=int4_latency_ms,
            energy_data_source=input_data.energy_data_source,
            resource_state=input_data.resource_state,
            gpu_vram_free_mb=input_data.gpu_vram_free_mb,
            fallback_used=fallback_used,
            decision_policy_version=input_data.decision_policy_version or self.policy.version,
            metadata={
                "co2_difference_mg": abs(fp16_co2_mg - int4_co2_mg),
                "co2_diff_threshold_mg": self.policy.co2_difference_threshold_mg,
                "latency_tolerance_pct": self.policy.latency_tolerance_pct,
                "profile_used": self.policy.profile_name,
            },
        )


def create_decision_engine(policy: Optional[DecisionPolicy] = None) -> DecisionEngine:
    """Factory function to create DecisionEngine."""
    return DecisionEngine(policy)


def decide_from_raw(
    safety_probability: float,
    carbon_intensity_gco2_per_kwh: float,
    safety_threshold: float = 0.50,
    latency_requirement_ms: Optional[float] = None,
    fp16_energy_wh: Optional[float] = None,
    int4_energy_wh: Optional[float] = None,
    fp16_latency_ms: Optional[float] = None,
    int4_latency_ms: Optional[float] = None,
    carbon_data_source: DataSourceType = DataSourceType.ESTIMATED,
    energy_data_source: DataSourceType = DataSourceType.ESTIMATED,
    latency_data_source: DataSourceType = DataSourceType.ESTIMATED,
    policy: Optional[DecisionPolicy] = None,
) -> DecisionOutput:
    """
    Convenience function for simple decision making.
    
    Creates DecisionInput from raw values and runs decision engine.
    """
    engine = create_decision_engine(policy)
    
    input_data = DecisionInput(
        safety_probability=safety_probability,
        safety_threshold=safety_threshold,
        carbon_intensity_gco2_per_kwh=carbon_intensity_gco2_per_kwh,
        carbon_data_source=carbon_data_source,
        fp16_estimated_energy_wh=fp16_energy_wh or 0.0,
        int4_estimated_energy_wh=int4_energy_wh or 0.0,
        fp16_estimated_latency_ms=fp16_latency_ms or 0.0,
        int4_estimated_latency_ms=int4_latency_ms or 0.0,
        energy_data_source=energy_data_source,
        latency_data_source=latency_data_source,
        latency_requirement_ms=latency_requirement_ms,
    )
    
    return engine.decide(input_data)