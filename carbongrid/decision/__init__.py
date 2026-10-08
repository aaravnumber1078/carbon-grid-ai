"""
CarbonGrid Decision Engine Module.

Provides the core decision logic for selecting inference configuration (FP16/INT4)
based on quantization safety, carbon intensity, latency requirements, and resource constraints.
"""

from carbongrid.decision.models import (
    Precision,
    DataSourceType,
    DecisionReason,
    FallbackPolicy,
    DecisionInput,
    DecisionOutput,
    TelemetryRecord,
    DEFAULT_PROFILES,
    get_default_profile,
    estimate_co2_from_energy,
)

from carbongrid.decision.policy import (
    DecisionPolicy,
    DecisionPolicyVersion,
    DEFAULT_POLICY,
    create_policy,
    create_policy_from_dict,
)

from carbongrid.decision.engine import (
    DecisionEngine,
    create_decision_engine,
    decide_from_raw,
)

__all__ = [
    # Models
    "Precision",
    "DataSourceType",
    "DecisionReason",
    "FallbackPolicy",
    "DecisionInput",
    "DecisionOutput",
    "TelemetryRecord",
    "DEFAULT_PROFILES",
    "get_default_profile",
    "estimate_co2_from_energy",
    # Policy
    "DecisionPolicy",
    "DecisionPolicyVersion",
    "DEFAULT_POLICY",
    "create_policy",
    "create_policy_from_dict",
    # Engine
    "DecisionEngine",
    "create_decision_engine",
    "decide_from_raw",
]