"""
Decision Engine data models for CarbonGrid-AI.

Defines the input/output contracts for the decision engine.
All models are serializable to JSON.
"""

from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from typing import Optional, Dict, Any, List
import json


class Precision(Enum):
    """Model precision configuration."""
    FP16 = "fp16"
    INT4 = "int4"


class DataSourceType(Enum):
    """Source/type of data used in decision."""
    MEASURED = "measured"       # Directly measured (e.g., NVML energy)
    ESTIMATED = "estimated"     # Calculated from benchmarks
    REPLAY = "replay"           # From recorded/replay data
    SIMULATED = "simulated"     # Simulated/hypothetical
    OFFLINE = "offline"         # Hardcoded fallback
    LIVE = "live"               # Live API
    UNKNOWN = "unknown"


class DecisionReason(Enum):
    """Reason for configuration selection."""
    SAFETY_GATE_FAILED = "safety_gate_failed"
    INT4_LATENCY_VIOLATION = "int4_latency_violation"
    FP16_LATENCY_VIOLATION = "fp16_latency_violation"
    BOTH_LATENCY_VIOLATION = "both_latency_violation"
    INT4_LOWER_ENVIRONMENTAL_COST = "int4_lower_environmental_cost"
    FP16_LOWER_ENVIRONMENTAL_COST = "fp16_lower_environmental_cost"
    ENVIRONMENTAL_COST_EQUAL = "environmental_cost_equal"
    FALLBACK_POLICY = "fallback_policy"
    RESOURCE_CONSTRAINT = "resource_constraint"
    CLASSIFIER_ERROR = "classifier_error"


class FallbackPolicy(Enum):
    """Fallback behavior when constraints cannot be satisfied."""
    PREFER_FP16 = "prefer_fp16"          # Default to FP16
    PREFER_LOWER_LATENCY = "prefer_lower_latency"
    PREFER_LOWER_ENERGY = "prefer_lower_energy"
    REJECT_REQUEST = "reject_request"


@dataclass
class DecisionInput:
    """
    Input to the Decision Engine.
    
    All fields should be provided by the caller (FastAPI gateway).
    """
    # Safety classifier output (required)
    safety_probability: float
    safety_threshold: float
    
    # Carbon context (required)
    carbon_intensity_gco2_per_kwh: float
    carbon_data_source: DataSourceType
    
    # Energy estimates (required)
    fp16_estimated_energy_wh: float
    int4_estimated_energy_wh: float
    
    # Latency estimates (required)
    fp16_estimated_latency_ms: float
    int4_estimated_latency_ms: float
    
    # Optional carbon context
    carbon_zone: str = "national"
    carbon_reading_type: str = "actual"
    carbon_timestamp: Optional[str] = None
    
    # Optional data sources
    energy_data_source: DataSourceType = DataSourceType.ESTIMATED
    latency_data_source: DataSourceType = DataSourceType.ESTIMATED
    
    # Optional constraints
    latency_requirement_ms: Optional[float] = None
    
    # Optional resource state
    resource_state: Optional[str] = None  # "available", "constrained", "unknown"
    gpu_vram_free_mb: Optional[int] = None
    vram_telemetry_available: bool = False
    
    # Policy configuration (can be overridden per-request)
    fallback_policy: FallbackPolicy = FallbackPolicy.PREFER_FP16
    default_precision: Precision = Precision.FP16
    
    # Metadata
    request_id: Optional[str] = None
    decision_policy_version: str = "1.0"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        d = asdict(self)
        # Convert enums to values
        d["carbon_data_source"] = self.carbon_data_source.value
        d["energy_data_source"] = self.energy_data_source.value
        d["latency_data_source"] = self.latency_data_source.value
        d["fallback_policy"] = self.fallback_policy.value
        d["default_precision"] = self.default_precision.value
        return d
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DecisionInput":
        """Create from dictionary."""
        data = data.copy()
        data["carbon_data_source"] = DataSourceType(data.get("carbon_data_source", "estimated"))
        data["energy_data_source"] = DataSourceType(data.get("energy_data_source", "estimated"))
        data["latency_data_source"] = DataSourceType(data.get("latency_data_source", "estimated"))
        data["fallback_policy"] = FallbackPolicy(data.get("fallback_policy", "prefer_fp16"))
        data["default_precision"] = Precision(data.get("default_precision", "fp16"))
        return cls(**data)


@dataclass
class DecisionOutput:
    """
    Output from the Decision Engine.
    
    Contains the selected configuration and full reasoning.
    """
    # Core decision
    selected_precision: Precision
    decision_reason: DecisionReason
    
    # Safety context
    safety_probability: float
    safety_threshold: float
    safety_gate_passed: bool
    
    # Carbon context
    carbon_intensity_gco2_per_kwh: float
    carbon_data_source: DataSourceType
    carbon_zone: str
    carbon_reading_type: str
    
    # Energy estimates for selected config
    estimated_energy_wh: float
    estimated_co2_g: float
    estimated_co2_mg: float
    
    # Latency estimates for selected config
    estimated_latency_ms: float
    
    # Constraint status
    latency_constraint_status: str  # "satisfied", "violated", "not_specified"
    latency_requirement_ms: Optional[float] = None
    
    # Alternative configuration estimates (for transparency)
    fp16_estimated_energy_wh: float = 0.0
    fp16_estimated_co2_g: float = 0.0
    fp16_estimated_latency_ms: float = 0.0
    int4_estimated_energy_wh: float = 0.0
    int4_estimated_co2_g: float = 0.0
    int4_estimated_latency_ms: float = 0.0
    
    # Energy data source
    energy_data_source: DataSourceType = DataSourceType.ESTIMATED
    
    # Resource context
    resource_state: Optional[str] = None
    gpu_vram_free_mb: Optional[int] = None
    
    # Fallback info
    fallback_used: bool = False
    
    # Policy
    decision_policy_version: str = "1.0"
    
    # Timestamp
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    
    # Additional context
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        d = asdict(self)
        d["selected_precision"] = self.selected_precision.value
        d["decision_reason"] = self.decision_reason.value
        d["carbon_data_source"] = self.carbon_data_source.value
        d["energy_data_source"] = self.energy_data_source.value
        return d
    
    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), default=str)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DecisionOutput":
        """Create from dictionary."""
        data = data.copy()
        data["selected_precision"] = Precision(data["selected_precision"])
        data["decision_reason"] = DecisionReason(data["decision_reason"])
        data["carbon_data_source"] = DataSourceType(data["carbon_data_source"])
        data["energy_data_source"] = DataSourceType(data["energy_data_source"])
        return cls(**data)


@dataclass
class TelemetryRecord:
    """
    Full telemetry record for a request.
    
    Combines decision output with actual execution results.
    """
    request_id: str
    prompt: str
    prompt_hash: str  # For deduplication/correlation
    
    # Decision info
    decision: DecisionOutput
    
    # Actual execution results (filled after inference)
    actual_precision: Optional[Precision] = None
    actual_latency_ms: Optional[float] = None
    actual_energy_wh: Optional[float] = None
    actual_co2_g: Optional[float] = None
    actual_tokens_per_sec: Optional[float] = None
    actual_gpu_memory_mb: Optional[int] = None
    actual_peak_gpu_memory_mb: Optional[int] = None
    actual_avg_power_w: Optional[float] = None
    
    # Response
    response: Optional[str] = None
    response_length_chars: Optional[int] = None
    
    # Status
    inference_status: str = "pending"  # "pending", "success", "failed", "timeout"
    error_message: Optional[str] = None
    
    # Timestamps
    request_received_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    decision_made_at: Optional[str] = None
    inference_started_at: Optional[str] = None
    inference_completed_at: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        d = asdict(self)
        if self.decision:
            d["decision"] = self.decision.to_dict()
        if self.actual_precision:
            d["actual_precision"] = self.actual_precision.value
        return d
    
    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), default=str)


# Default energy/latency profiles (from Phase 0.5 and B2 measurements)
DEFAULT_PROFILES = {
    "phase05": {
        "fp16": {
            "energy_wh": 0.0488,
            "latency_ms": 5867,
            "throughput_tok_s": 21.9,
            "power_w": 30.45,
            "peak_vram_mb": 2960,
            "source": "measured",
        },
        "int4": {
            "energy_wh": 0.0493,
            "latency_ms": 10580,
            "throughput_tok_s": 12.1,
            "power_w": 17.12,
            "peak_vram_mb": 1161,
            "source": "measured",
        }
    },
    "b2_workload": {
        "fp16": {
            "energy_wh": 0.1061,
            "latency_ms": 11640,
            "source": "measured",
        },
        "int4": {
            "energy_wh": 0.0945,
            "latency_ms": 22077,
            "source": "measured",
        }
    }
}


def get_default_profile(profile_name: str = "phase05") -> Dict[str, Any]:
    """Get default energy/latency profile."""
    return DEFAULT_PROFILES.get(profile_name, DEFAULT_PROFILES["phase05"])


def estimate_co2_from_energy(energy_wh: float, carbon_intensity_gco2_per_kwh: float) -> tuple:
    """
    Calculate CO2 from energy and carbon intensity.
    
    Args:
        energy_wh: Energy in watt-hours (must be >= 0)
        carbon_intensity_gco2_per_kwh: Carbon intensity in gCO2/kWh (must be >= 0)
        
    Returns:
        (co2_grams, co2_milligrams)
        
    Raises:
        ValueError: If inputs are negative, NaN, infinite, or None
    """
    # Validate energy_wh
    if energy_wh is None:
        raise ValueError("energy_wh cannot be None")
    if not isinstance(energy_wh, (int, float)):
        raise ValueError(f"energy_wh must be numeric, got {type(energy_wh).__name__}")
    if energy_wh != energy_wh:  # NaN check
        raise ValueError("energy_wh cannot be NaN")
    if energy_wh == float('inf') or energy_wh == float('-inf'):
        raise ValueError("energy_wh cannot be infinite")
    if energy_wh < 0:
        raise ValueError("energy_wh cannot be negative")
    
    # Validate carbon_intensity_gco2_per_kwh
    if carbon_intensity_gco2_per_kwh is None:
        raise ValueError("carbon_intensity_gco2_per_kwh cannot be None")
    if not isinstance(carbon_intensity_gco2_per_kwh, (int, float)):
        raise ValueError(f"carbon_intensity_gco2_per_kwh must be numeric, got {type(carbon_intensity_gco2_per_kwh).__name__}")
    if carbon_intensity_gco2_per_kwh != carbon_intensity_gco2_per_kwh:  # NaN check
        raise ValueError("carbon_intensity_gco2_per_kwh cannot be NaN")
    if carbon_intensity_gco2_per_kwh == float('inf') or carbon_intensity_gco2_per_kwh == float('-inf'):
        raise ValueError("carbon_intensity_gco2_per_kwh cannot be infinite")
    if carbon_intensity_gco2_per_kwh < 0:
        raise ValueError("carbon_intensity_gco2_per_kwh cannot be negative")
    
    energy_kwh = energy_wh / 1000.0
    co2_g = energy_kwh * carbon_intensity_gco2_per_kwh
    co2_mg = co2_g * 1000.0
    return co2_g, co2_mg