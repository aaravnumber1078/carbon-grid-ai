"""
API Schemas for CarbonGrid-AI FastAPI Gateway.

Defines request/response models using Pydantic for validation.
"""

from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List
from enum import Enum


class Precision(str, Enum):
    FP16 = "fp16"
    INT4 = "int4"


class DataSourceType(str, Enum):
    MEASURED = "measured"
    ESTIMATED = "estimated"
    REPLAY = "replay"
    SIMULATED = "simulated"
    OFFLINE = "offline"
    LIVE = "live"
    UNKNOWN = "unknown"


class DecisionReason(str, Enum):
    SAFETY_GATE_FAILED = "safety_gate_failed"
    INT4_LATENCY_VIOLATION = "int4_latency_violation"
    FP16_LATENCY_VIOLATION = "fp16_latency_violation"
    BOTH_LATENCY_VIOLATION = "both_latency_violation"
    INT4_LOWER_ENVIRONMENTAL_COST = "int4_lower_environmental_cost"
    FP16_LOWER_ENVIRONMENTAL_COST = "fp16_lower_environmental_cost"
    ENVIRONMENTAL_COST_EQUAL = "environmental_cost_equal"
    FALLBACK_POLICY = "fallback_policy"
    RESOURCE_CONSTRAINT = "resource_constraint"


class FallbackPolicy(str, Enum):
    PREFER_FP16 = "prefer_fp16"
    PREFER_LOWER_LATENCY = "prefer_lower_latency"
    PREFER_LOWER_ENERGY = "prefer_lower_energy"
    REJECT_REQUEST = "reject_request"


class GenerationRequest(BaseModel):
    """Request for text generation."""
    prompt: str = Field(..., description="Input prompt for generation", min_length=1)
    max_new_tokens: int = Field(default=128, description="Maximum new tokens to generate", ge=1, le=512)
    latency_requirement_ms: Optional[float] = Field(
        default=None, 
        description="Optional latency requirement in milliseconds"
    )
    task_type: Optional[str] = Field(
        default="unknown",
        description="Optional task type hint for classifier"
    )


class GenerationResponse(BaseModel):
    """Response from text generation with CarbonGrid telemetry."""
    response: str = Field(..., description="Generated text response")
    
    # CarbonGrid telemetry
    carbon_grid: Dict[str, Any] = Field(default_factory=dict, description="CarbonGrid decision and telemetry data")


class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    version: str
    classifier_loaded: bool
    carbon_provider_mode: str
    inference_provider_status: Dict[str, Any]


class DecisionInputSchema(BaseModel):
    """Schema for Decision Engine input (for testing)."""
    safety_probability: float
    safety_threshold: float
    carbon_intensity_gco2_per_kwh: float
    carbon_data_source: DataSourceType
    fp16_estimated_energy_wh: float
    int4_estimated_energy_wh: float
    fp16_estimated_latency_ms: float
    int4_estimated_latency_ms: float
    energy_data_source: DataSourceType = DataSourceType.ESTIMATED
    latency_data_source: DataSourceType = DataSourceType.ESTIMATED
    latency_requirement_ms: Optional[float] = None
    resource_state: Optional[str] = None
    gpu_vram_free_mb: Optional[int] = None
    vram_telemetry_available: bool = False
    fallback_policy: FallbackPolicy = FallbackPolicy.PREFER_FP16
    default_precision: Precision = Precision.FP16