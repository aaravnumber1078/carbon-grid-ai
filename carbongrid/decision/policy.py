"""
Decision Engine policy configuration for CarbonGrid-AI.

Centralizes all decision thresholds, defaults, and policy parameters.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional
from enum import Enum

from carbongrid.decision.models import (
    Precision,
    DataSourceType,
    FallbackPolicy,
    DEFAULT_PROFILES,
    get_default_profile,
)


class DecisionPolicyVersion(Enum):
    """Policy version identifiers."""
    V1_0 = "1.0"
    V1_1 = "1.1"  # Future: with more sophisticated routing


@dataclass
class DecisionPolicy:
    """
    Configuration for the Decision Engine.
    
    All thresholds and defaults are defined here for transparency and testability.
    """
    # Policy version
    version: str = DecisionPolicyVersion.V1_0.value
    
    # Safety gate
    safety_threshold: float = 0.50  # From C.1.1 validation
    
    # Latency constraint handling
    latency_tolerance_pct: float = 0.10  # 10% tolerance on latency estimates
    
    # Environmental cost comparison
    co2_difference_threshold_mg: float = 0.1  # Minimum CO2 difference to consider "different" (mg)
    
    # Fallback behavior
    fallback_policy: FallbackPolicy = FallbackPolicy.PREFER_FP16
    
    # Default precision when no other criteria distinguish
    default_precision: Precision = Precision.FP16
    
    # Energy/latency profiles to use
    profile_name: str = "phase05"  # "phase05" or "b2_workload"
    
    # Override specific profile values (None = use profile defaults)
    fp16_energy_wh_override: Optional[float] = None
    int4_energy_wh_override: Optional[float] = None
    fp16_latency_ms_override: Optional[float] = None
    int4_latency_ms_override: Optional[float] = None
    
    # Resource constraints
    min_vram_mb_for_fp16: int = 3200  # Minimum free VRAM for FP16
    min_vram_mb_for_int4: int = 1400  # Minimum free VRAM for INT4
    
    # Carbon intensity fallback (if provider unavailable)
    default_carbon_intensity: float = 200.0  # gCO2/kWh, UK average-ish
    
    # Logging/monitoring
    log_decisions: bool = True
    include_alternative_estimates: bool = True
    
    def get_fp16_energy_wh(self) -> float:
        """Get FP16 energy estimate (Wh)."""
        if self.fp16_energy_wh_override is not None:
            return self.fp16_energy_wh_override
        profile = get_default_profile(self.profile_name)
        return profile["fp16"]["energy_wh"]
    
    def get_int4_energy_wh(self) -> float:
        """Get INT4 energy estimate (Wh)."""
        if self.int4_energy_wh_override is not None:
            return self.int4_energy_wh_override
        profile = get_default_profile(self.profile_name)
        return profile["int4"]["energy_wh"]
    
    def get_fp16_latency_ms(self) -> float:
        """Get FP16 latency estimate (ms)."""
        if self.fp16_latency_ms_override is not None:
            return self.fp16_latency_ms_override
        profile = get_default_profile(self.profile_name)
        return profile["fp16"]["latency_ms"]
    
    def get_int4_latency_ms(self) -> float:
        """Get INT4 latency estimate (ms)."""
        if self.int4_latency_ms_override is not None:
            return self.int4_latency_ms_override
        profile = get_default_profile(self.profile_name)
        return profile["int4"]["latency_ms"]
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "version": self.version,
            "safety_threshold": self.safety_threshold,
            "latency_tolerance_pct": self.latency_tolerance_pct,
            "co2_difference_threshold_mg": self.co2_difference_threshold_mg,
            "fallback_policy": self.fallback_policy.value,
            "default_precision": self.default_precision.value,
            "profile_name": self.profile_name,
            "fp16_energy_wh_override": self.fp16_energy_wh_override,
            "int4_energy_wh_override": self.int4_energy_wh_override,
            "fp16_latency_ms_override": self.fp16_latency_ms_override,
            "int4_latency_ms_override": self.int4_latency_ms_override,
            "min_vram_mb_for_fp16": self.min_vram_mb_for_fp16,
            "min_vram_mb_for_int4": self.min_vram_mb_for_int4,
            "default_carbon_intensity": self.default_carbon_intensity,
            "log_decisions": self.log_decisions,
            "include_alternative_estimates": self.include_alternative_estimates,
            "effective_fp16_energy_wh": self.get_fp16_energy_wh(),
            "effective_int4_energy_wh": self.get_int4_energy_wh(),
            "effective_fp16_latency_ms": self.get_fp16_latency_ms(),
            "effective_int4_latency_ms": self.get_int4_latency_ms(),
        }


# Default policy instance
DEFAULT_POLICY = DecisionPolicy()


def create_policy(**overrides) -> DecisionPolicy:
    """
    Create a DecisionPolicy with optional overrides.
    
    Args:
        **overrides: Any field of DecisionPolicy to override
        
    Returns:
        Configured DecisionPolicy instance
    """
    return DecisionPolicy(**overrides)


def create_policy_from_dict(data: Dict[str, Any]) -> DecisionPolicy:
    """Create policy from dictionary (e.g., config file)."""
    # Handle enum fields
    if "fallback_policy" in data and isinstance(data["fallback_policy"], str):
        data["fallback_policy"] = FallbackPolicy(data["fallback_policy"])
    if "default_precision" in data and isinstance(data["default_precision"], str):
        data["default_precision"] = Precision(data["default_precision"])
    return DecisionPolicy(**data)