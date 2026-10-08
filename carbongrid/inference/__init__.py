"""
CarbonGrid Inference Provider Module.

Provides the InferenceProvider class for managing Qwen2.5-1.5B-Instruct
inference at FP16 and INT4 precision.
"""

from carbongrid.inference.provider import (
    InferenceProvider,
    InferenceMetrics,
    Precision,
    GPUPowerMonitor,
    create_inference_provider,
)

__all__ = [
    "InferenceProvider",
    "InferenceMetrics",
    "Precision",
    "GPUPowerMonitor",
    "create_inference_provider",
]