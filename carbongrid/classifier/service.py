"""
Classifier Service for CarbonGrid-AI.

Loads and manages the C.1.1 quantization-safety classifier and feature extractor.
Provides a clean interface for generating safety probabilities from prompts.
"""

import json
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import joblib

from carbongrid.classifier.features import FeatureExtractor, FeatureConfig, create_splits


# C.1.1 artifact paths
C11_CLASSIFIER_PATH = Path("models/quantization_safety/classifier_c1_1.pkl")
C11_EXTRACTOR_PATH = Path("models/quantization_safety/feature_extractor_c1_1.pkl")
C11_METADATA_PATH = Path("models/quantization_safety/metadata_c1_1.json")


# Known task types from training data (for handling unknown task types)
KNOWN_TASK_TYPES = [
    "factual", "coding", "reasoning", "explanation", "summarization",
    "scientific", "extraction", "creative",
]


def map_task_type(task_type: str) -> str:
    """Map unknown task types to a known type from training data."""
    if task_type in KNOWN_TASK_TYPES:
        return task_type
    # Default to "factual" for unknown types
    return "factual"


class ClassifierService:
    """
    Service for loading and using the C.1.1 quantization-safety classifier.
    
    The classifier predicts the probability that a prompt is safe to quantize to INT4.
    """
    
    def __init__(
        self,
        classifier_path: Optional[Path] = None,
        extractor_path: Optional[Path] = None,
    ):
        self.classifier_path = classifier_path or C11_CLASSIFIER_PATH
        self.extractor_path = extractor_path or C11_EXTRACTOR_PATH
        
        self._classifier = None
        self._extractor = None
        self._metadata = None
        self._threshold = 0.50  # Default from C.1.1
    
    def load(self):
        """Load the classifier, feature extractor, and metadata."""
        print(f"Loading C.1.1 classifier from {self.classifier_path}...")
        self._classifier = joblib.load(self.classifier_path)
        
        print(f"Loading C.1.1 feature extractor from {self.extractor_path}...")
        self._extractor = FeatureExtractor.load(str(self.extractor_path))
        
        print(f"Loading C.1.1 metadata from {C11_METADATA_PATH}...")
        with open(C11_METADATA_PATH, 'r') as f:
            self._metadata = json.load(f)
        
        # Get the production threshold from metadata
        self._threshold = self._metadata.get("threshold_selection", {}).get("selected_threshold", 0.50)
        
        print(f"C.1.1 classifier loaded successfully")
        print(f"  Model: {type(self._classifier).__name__}")
        print(f"  Features: {len(self._extractor.get_feature_names())}")
        print(f"  Production threshold: {self._threshold:.2f}")
        
        return self
    
    def is_loaded(self) -> bool:
        """Check if classifier and extractor are loaded."""
        return self._classifier is not None and self._extractor is not None
    
    def get_threshold(self) -> float:
        """Get the production safety threshold."""
        return self._threshold
    
    def get_feature_names(self) -> List[str]:
        """Get feature names from the extractor."""
        if not self.is_loaded():
            raise RuntimeError("ClassifierService not loaded. Call load() first.")
        return self._extractor.get_feature_names()
    
    def _transform(self, records: List[Dict[str, Any]]) -> np.ndarray:
        """Transform records to feature vector, removing the complexity_level feature."""
        X, _ = self._extractor.transform(records)
        
        # The C.1.1 feature extractor was saved with feature_names excluding complexity_level
        # (122 features), but transform() produces 123 columns because extract_raw_features
        # always includes complexity_level. The extra column sits at the TF-IDF boundary.
        # Find the first TF-IDF feature in feature_names to locate the boundary.
        feature_names = self._extractor.get_feature_names()
        expected_features = len(feature_names)
        
        if X.shape[1] == expected_features:
            return X
        
        if X.shape[1] != expected_features + 1:
            raise ValueError(
                f"Unexpected feature matrix shape: got {X.shape[1]} columns, "
                f"expected {expected_features} or {expected_features + 1}"
            )
        
        # Locate the first TF-IDF feature name; complexity_level occupies that index
        tfidf_start = next(
            (i for i, name in enumerate(feature_names) if name.startswith("tfidf_")),
            None,
        )
        if tfidf_start is None:
            raise ValueError("Cannot locate TF-IDF boundary in feature_names")
        
        # Remove the extra column at the TF-IDF boundary (complexity_level)
        X = np.delete(X, tfidf_start, axis=1)
        
        if X.shape[1] != expected_features:
            raise ValueError(
                f"Column removal failed: got {X.shape[1]} columns, expected {expected_features}"
            )
        
        return X
    
    def predict_safety(self, prompt: str, task_type: str = "factual") -> Dict[str, Any]:
        """
        Predict quantization safety for a single prompt.
        
        Args:
            prompt: The input prompt text
            task_type: Optional task type hint (e.g., "factual", "coding", "reasoning")
            
        Returns:
            Dictionary with safety_probability, is_safe, threshold, and metadata
        """
        if not self.is_loaded():
            raise RuntimeError("ClassifierService not loaded. Call load() first.")
        
        # Map unknown task types to known ones
        mapped_task_type = map_task_type(task_type)
        
        record = {
            "prompt": prompt,
            "task_type": mapped_task_type,
            "complexity_level": 0,
        }
        
        X = self._transform([record])
        
        # Get probability of being safe (class 1)
        safety_probability = float(self._classifier.predict_proba(X)[:, 1][0])
        
        # Apply threshold
        is_safe = safety_probability >= self._threshold
        
        return {
            "safety_probability": safety_probability,
            "is_safe": is_safe,
            "threshold": self._threshold,
            "model": type(self._classifier).__name__,
            "features_used": len(self.get_feature_names()),
        }
    
    def predict_batch(self, prompts: List[str], task_types: Optional[List[str]] = None) -> List[Dict[str, Any]]:
        """
        Predict quantization safety for multiple prompts.
        
        Args:
            prompts: List of prompt texts
            task_types: Optional list of task types (same length as prompts)
            
        Returns:
            List of prediction dictionaries
        """
        if not self.is_loaded():
            raise RuntimeError("ClassifierService not loaded. Call load() first.")
        
        if task_types is None:
            task_types = ["factual"] * len(prompts)
        
        # Map task types
        mapped_types = [map_task_type(t) for t in task_types]
        
        records = [
            {"prompt": p, "task_type": t, "complexity_level": 0}
            for p, t in zip(prompts, mapped_types)
        ]
        
        X = self._transform(records)
        probabilities = self._classifier.predict_proba(X)[:, 1]
        
        results = []
        for prob, prompt, task_type in zip(probabilities, prompts, mapped_types):
            results.append({
                "prompt": prompt,
                "task_type": task_type,
                "safety_probability": float(prob),
                "is_safe": float(prob) >= self._threshold,
                "threshold": self._threshold,
            })
        
        return results
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get C.1.1 metadata."""
        return self._metadata or {}


def create_classifier_service(
    classifier_path: Optional[str] = None,
    extractor_path: Optional[str] = None,
) -> ClassifierService:
    """Factory function to create and load ClassifierService."""
    service = ClassifierService(
        classifier_path=Path(classifier_path) if classifier_path else None,
        extractor_path=Path(extractor_path) if extractor_path else None,
    )
    return service.load()


def get_classifier_info() -> Dict[str, Any]:
    """Get information about the C.1.1 classifier without loading it."""
    if not C11_METADATA_PATH.exists():
        return {"error": "C.1.1 metadata not found"}
    
    with open(C11_METADATA_PATH, 'r') as f:
        metadata = json.load(f)
    
    return {
        "classifier_exists": C11_CLASSIFIER_PATH.exists(),
        "extractor_exists": C11_EXTRACTOR_PATH.exists(),
        "metadata": metadata,
        "threshold": metadata.get("threshold_selection", {}).get("selected_threshold", 0.50),
        "test_accuracy": metadata.get("final_test_metrics", {}).get("accuracy"),
        "test_false_safe_rate": metadata.get("final_test_metrics", {}).get("false_safe_rate"),
        "test_int4_opportunity": metadata.get("final_test_metrics", {}).get("int4_opportunity_rate"),
    }