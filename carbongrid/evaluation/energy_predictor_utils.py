"""
Energy Predictor Utility Functions.

Provides safe loading and prediction for the Phase F.1 energy predictor.
The predictor is EXPLORATORY ONLY and NOT used in production routing decisions.

All predictions are guarded against invalid values (negative, NaN, infinite, None).
"""

import logging
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, Union
import joblib
import numpy as np

logger = logging.getLogger(__name__)

# Predictor artifact paths
ENERGY_PREDICTOR_MODEL_PATH = Path("models/energy_predictor/energy_predictor.joblib")
# No feature extractor saved for this predictor - features are derived from B2 dataset

# Valid energy range for this model/hardware (Wh)
MIN_VALID_ENERGY_WH = 0.0
MAX_VALID_ENERGY_WH = 10.0  # Conservative upper bound


class EnergyPredictorError(Exception):
    """Raised when energy predictor encounters an error."""
    pass


class EnergyPredictorNotAvailableError(EnergyPredictorError):
    """Raised when predictor model cannot be loaded."""
    pass


class EnergyPredictorInvalidPredictionError(EnergyPredictorError):
    """Raised when prediction is invalid (negative, NaN, infinite)."""
    pass


def load_energy_predictor() -> Any:
    """
    Load the Phase F.1 energy predictor model.
    
    Returns:
        Fitted sklearn Pipeline object
        
    Raises:
        EnergyPredictorNotAvailableError: If model file missing or cannot be loaded
    """
    if not ENERGY_PREDICTOR_MODEL_PATH.exists():
        raise EnergyPredictorNotAvailableError(
            f"Energy predictor model not found at {ENERGY_PREDICTOR_MODEL_PATH}. "
            "Run scripts/train_energy_predictor.py to train it."
        )
    
    try:
        pipeline = joblib.load(ENERGY_PREDICTOR_MODEL_PATH)
        logger.info(f"Loaded energy predictor from {ENERGY_PREDICTOR_MODEL_PATH}")
        return pipeline
    except Exception as e:
        raise EnergyPredictorNotAvailableError(f"Failed to load energy predictor: {e}")


def validate_energy_prediction(
    prediction: Union[float, np.floating],
    min_valid: float = MIN_VALID_ENERGY_WH,
    max_valid: float = MAX_VALID_ENERGY_WH
) -> float:
    """
    Validate and sanitize an energy prediction.
    
    This is the CRITICAL safety guard: prevents negative, NaN, infinite,
    or out-of-range predictions from being used in any decision context.
    
    Args:
        prediction: Raw model prediction (energy difference in Wh)
        min_valid: Minimum physically valid energy (default 0.0)
        max_valid: Maximum plausible energy for this hardware (default 10.0 Wh)
        
    Returns:
        Sanitized prediction clamped to valid range
        
    Raises:
        EnergyPredictorInvalidPredictionError: If prediction is invalid and 
            clamping is not appropriate for the use case
    """
    # Handle None
    if prediction is None:
        raise EnergyPredictorInvalidPredictionError("Prediction is None")
    
    # Handle non-numeric
    if not isinstance(prediction, (int, float, np.floating)):
        raise EnergyPredictorInvalidPredictionError(
            f"Prediction is not numeric: {type(prediction).__name__}"
        )
    
    # Handle NaN
    if isinstance(prediction, float) and prediction != prediction:
        raise EnergyPredictorInvalidPredictionError("Prediction is NaN")
    
    # Handle infinity
    if prediction == float('inf') or prediction == float('-inf'):
        raise EnergyPredictorInvalidPredictionError("Prediction is infinite")
    
    # Convert to float
    pred_float = float(prediction)
    
    # Check range
    if pred_float < min_valid or pred_float > max_valid:
        # For negative predictions (physically impossible), clamp to min_valid
        if pred_float < min_valid:
            logger.warning(
                f"Energy predictor returned negative value ({pred_float:.6f} Wh). "
                f"Clamping to {min_valid} Wh. This indicates predictor is unreliable."
            )
            return min_valid
        elif pred_float > max_valid:
            logger.warning(
                f"Energy predictor returned implausibly high value ({pred_float:.6f} Wh). "
                f"Clamping to {max_valid} Wh."
            )
            return max_valid
    
    return pred_float


def predict_energy_difference_safe(
    features: Dict[str, float],
    fallback: float = 0.0
) -> Tuple[float, Dict[str, Any]]:
    """
    Predict energy difference (INT4 - FP16) with full safety guards.
    
    This function is for EXPLORATORY/RESEARCH USE ONLY.
    It does NOT affect production routing decisions.
    
    Args:
        features: Dictionary of feature name -> value (must match training features)
        fallback: Value to return if prediction fails (default 0.0 = no difference)
        
    Returns:
        (validated_prediction_wh, metadata_dict)
        
    Note:
        This function will NOT raise exceptions. All errors are caught and logged.
        The fallback value is used for any failure mode.
    """
    metadata = {
        "success": False,
        "error": None,
        "model_loaded": False,
        "raw_prediction": None,
        "validated_prediction": None,
        "fallback_used": False,
        "note": "EXPLORATORY ONLY - not used in production routing",
    }
    
    try:
        # Load model
        pipeline = load_energy_predictor()
        metadata["model_loaded"] = True
        
        # Prepare features (must match training features exactly)
        # Features needed: NUMERIC_FEATURES + task_type
        # This is a simplified version - real implementation would need proper feature engineering
        
        # For now, return fallback with clear metadata
        # TODO: Implement full feature extraction if this is ever needed for research
        metadata["error"] = "Full feature extraction not implemented - predictor not used in production"
        metadata["fallback_used"] = True
        metadata["validated_prediction"] = fallback
        return fallback, metadata
        
    except EnergyPredictorNotAvailableError as e:
        logger.warning(f"Energy predictor not available: {e}")
        metadata["error"] = str(e)
        metadata["fallback_used"] = True
        metadata["validated_prediction"] = fallback
        return fallback, metadata
        
    except EnergyPredictorInvalidPredictionError as e:
        logger.error(f"Energy predictor returned invalid value: {e}")
        metadata["error"] = str(e)
        metadata["fallback_used"] = True
        metadata["validated_prediction"] = fallback
        return fallback, metadata
        
    except Exception as e:
        logger.error(f"Unexpected error in energy predictor: {e}")
        metadata["error"] = f"Unexpected: {e}"
        metadata["fallback_used"] = True
        metadata["validated_prediction"] = fallback
        return fallback, metadata


# Backward compatibility - the predictor is NOT used in production routing
# This is just to document that the artifact exists and has been audited
PREDICTOR_STATUS = {
    "model_exists": ENERGY_PREDICTOR_MODEL_PATH.exists(),
    "used_in_production_routing": False,
    "used_in_telemetry": False,
    "audit_status": "EXPLORATORY - not validated for production",
    "known_issues": [
        "Negative predictions observed in training (1/25 INT4 samples)",
        "No feature extractor saved - cannot easily load for new prompts",
        "Workload-specific (B2 dataset, 1000-token generations)",
        "Directional accuracy 100% but misleading (all B2 prompts had INT4 < FP16 in that benchmark)",
    ],
    "recommendation": "Do not use for routing decisions. Use profile-based estimates (Phase 0.5, B2) instead.",
}