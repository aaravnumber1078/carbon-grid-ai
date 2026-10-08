"""
CarbonGrid API Module.

FastAPI gateway for carbon-aware LLM inference.
"""

from carbongrid.api.main import app
from carbongrid.api.schemas import (
    GenerationRequest,
    GenerationResponse,
    HealthResponse,
    Precision,
    DataSourceType,
    DecisionReason,
    FallbackPolicy,
)

__all__ = [
    "app",
    "GenerationRequest",
    "GenerationResponse",
    "HealthResponse",
    "Precision",
    "DataSourceType",
    "DecisionReason",
    "FallbackPolicy",
]