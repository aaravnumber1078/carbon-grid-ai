"""
Phase C: Feature Extraction and Vectorization for Quantization-Safety Classifier.

Transforms raw prompt features into model-ready feature vectors.
Handles task type encoding, feature scaling, and vectorization.
"""

import json
import re
from dataclasses import dataclass
from typing import Dict, List, Any, Optional, Tuple
from collections import Counter
from pathlib import Path

import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer


@dataclass
class FeatureConfig:
    """Configuration for feature extraction."""
    use_task_type: bool = True
    use_complexity_level: bool = False
    use_tfidf: bool = True
    tfidf_max_features: int = 100
    tfidf_ngram_range: Tuple[int, int] = (1, 2)
    scale_numeric: bool = True
    random_state: int = 42


class FeatureExtractor:
    """
    Extracts and vectorizes features from prompt records.
    
    Features available BEFORE inference only.
    """
    
    def __init__(self, config: FeatureConfig = None):
        self.config = config or FeatureConfig()
        self.scaler = StandardScaler() if self.config.scale_numeric else None
        self.task_encoder = LabelEncoder()
        self.tfidf = TfidfVectorizer(
            max_features=self.config.tfidf_max_features,
            ngram_range=self.config.tfidf_ngram_range,
            stop_words='english',
            lowercase=True
        ) if self.config.use_tfidf else None
        
        self._fitted = False
        self.feature_names: List[str] = []
        self.numeric_feature_indices: List[int] = []
    
    def extract_raw_features(self, record: Dict[str, Any]) -> Dict[str, float]:
        """
        Extract raw numeric features from a record.
        
        These are features available BEFORE inference.
        """
        prompt = record.get("prompt", "")
        features = record.get("features", {})
        task_type = record.get("task_type", "unknown")
        complexity_level = record.get("complexity_level", 0)
        
        words = prompt.split()
        word_count = len(words)
        char_length = len(prompt)
        token_count_estimate = max(1, char_length // 4)
        
        # Sentence and question counts
        sentence_count = len(re.split(r'[.!?]+', prompt)) - 1
        sentence_count = max(1, sentence_count)
        question_count = prompt.count('?')
        
        # Code indicators
        code_keywords = ['def ', 'class ', 'import ', 'from ', 'function', 'return ', 'if ', 'for ', 'while ',
                         'try:', 'except:', 'lambda ', 'async ', 'await ', 'yield ', 'with ', 'as ',
                         'public ', 'private ', 'protected ', 'static ', 'void ', 'int ', 'string ',
                         'console.log', 'print(', 'system.out', 'std::', '#include', 'fn ', 'let ',
                         'const ', 'var ', '=>', 'async def', 'await ']
        code_keyword_count = sum(1 for kw in code_keywords if kw in prompt.lower())
        has_code_indicator = code_keyword_count > 0 or '```' in prompt or 'def ' in prompt
        
        # Scientific terminology
        sci_pattern = re.compile(r'\b[A-Z][a-z]{4,}\b|\b[a-z]{8,}\b')
        scientific_terms = sci_pattern.findall(prompt.lower())
        common_words = {'the', 'and', 'for', 'with', 'this', 'that', 'from', 'have', 'been', 'were',
                        'their', 'there', 'which', 'when', 'what', 'where', 'who', 'how', 'why',
                        'about', 'into', 'more', 'some', 'such', 'only', 'other', 'than', 'then',
                        'them', 'these', 'those', 'upon', 'within', 'without', 'during', 'before',
                        'after', 'since', 'until', 'while', 'under', 'over', 'between', 'among',
                        'through', 'across', 'against', 'beyond', 'because', 'therefore', 'however'}
        scientific_terms = [w for w in scientific_terms if w not in common_words]
        scientific_term_count = len(scientific_terms)
        has_scientific_terms = scientific_term_count > 0
        
        # Reasoning indicators
        reasoning_words = ['because', 'therefore', 'thus', 'hence', 'so', 'if', 'then', 'implies',
                           'follows', 'conclude', 'reason', 'logic', 'step', 'first', 'second',
                           'assume', 'suppose', 'given', 'premise', 'deduce', 'infer', 'prove',
                           'show', 'demonstrate', 'explain why', 'why does', 'how does']
        reasoning_count = sum(1 for w in reasoning_words if w in prompt.lower())
        has_reasoning_indicators = reasoning_count > 0
        
        # Numbers
        numbers = re.findall(r'\b\d+(?:\.\d+)?\b', prompt)
        number_count = len(numbers)
        has_numbers = number_count > 0
        
        # Vocabulary richness
        unique_words = len(set(w.lower() for w in words))
        unique_word_ratio = unique_words / max(word_count, 1)
        avg_word_length = sum(len(w) for w in words) / max(word_count, 1)
        
        # Length hints
        requested_length_hint = 0
        max_tokens_requested = 0
        length_patterns = [
            (r'in (\d+) sentences?', 'sentences', 20),
            (r'in (\d+) words?', 'words', 10),
            (r'briefly', 'brief', 50),
            (r'short', 'short', 30),
            (r'concise', 'concise', 40),
            (r'detailed', 'detailed', 200),
            (r'comprehensive', 'comprehensive', 300),
            (r'(\d+) paragraphs?', 'paragraphs', 50),
        ]
        for pattern, unit, mult in length_patterns:
            match = re.search(pattern, prompt.lower())
            if match:
                requested_length_hint = 1
                if match.groups() and match.group(1).isdigit():
                    max_tokens_requested = int(match.group(1)) * mult
                break
        
        # Instruction count (rough heuristic)
        instruction_keywords = ['write', 'explain', 'describe', 'summarize', 'extract', 'create',
                                'implement', 'design', 'analyze', 'compare', 'list', 'provide',
                                'give', 'show', 'find', 'calculate', 'compute', 'debug',
                                'write a', 'write an', 'create a', 'create an']
        instruction_count = sum(1 for kw in instruction_keywords if kw in prompt.lower())
        
        # Constraint indicators
        constraint_keywords = ['must', 'should', 'exactly', 'precisely', 'only', 'without',
                               'not', 'avoid', 'ensure', 'guarantee', 'require', 'limit',
                               'maximum', 'minimum', 'at least', 'at most']
        constraint_count = sum(1 for kw in constraint_keywords if kw in prompt.lower())
        
        # Task type one-hot (will be encoded later)
        raw_features = {
            # Basic length
            "char_length": float(char_length),
            "word_count": float(word_count),
            "token_count_estimate": float(token_count_estimate),
            
            # Structural
            "has_question_mark": float('?' in prompt),
            "has_code_indicator": float(has_code_indicator),
            "question_count": float(question_count),
            "sentence_count": float(sentence_count),
            
            # Content indicators
            "has_scientific_terms": float(has_scientific_terms),
            "scientific_term_count": float(scientific_term_count),
            "has_reasoning_indicators": float(has_reasoning_indicators),
            "reasoning_indicator_count": float(reasoning_count),
            "has_code_keywords": float(has_code_indicator),
            "code_keyword_count": float(code_keyword_count),
            
            # Lexical
            "avg_word_length": avg_word_length,
            "unique_word_ratio": unique_word_ratio,
            "has_numbers": float(has_numbers),
            "number_count": float(number_count),
            
            # Complexity hints
            "requested_length_hint": float(requested_length_hint),
            "max_tokens_requested": float(max_tokens_requested),
            "instruction_count": float(instruction_count),
            "constraint_count": float(constraint_count),
            
            # Task type (will be encoded)
            "task_type_raw": task_type,
            
            # Complexity level (optional)
            "complexity_level": float(complexity_level),
        }
        
        return raw_features
    
    def fit(self, records: List[Dict[str, Any]]):
        """Fit vectorizers and scalers on training data."""
        # Extract raw features for all records
        raw_features_list = [self.extract_raw_features(r) for r in records]
        
        # Extract task types for encoding
        task_types = [rf["task_type_raw"] for rf in raw_features_list]
        self.task_encoder.fit(task_types)
        
        # Build numeric feature matrix
        numeric_features = []
        for rf in raw_features_list:
            numeric_row = [v for k, v in rf.items() if k != "task_type_raw"]
            numeric_features.append(numeric_row)
        
        numeric_array = np.array(numeric_features, dtype=np.float32)
        
        if self.config.scale_numeric:
            self.scaler.fit(numeric_array)
            numeric_array = self.scaler.transform(numeric_array)
        
        # TF-IDF on prompts
        if self.config.use_tfidf:
            prompts = [r.get("prompt", "") for r in records]
            self.tfidf.fit(prompts)
            tfidf_features = self.tfidf.transform(prompts).toarray()
            
            # Combine numeric + TF-IDF
            combined = np.hstack([numeric_array, tfidf_features])
        else:
            combined = numeric_array
        
        # Build feature names
        numeric_names = [k for k in raw_features_list[0].keys() if k != "task_type_raw"]
        if self.config.use_task_type:
            numeric_names = ["task_type"] + numeric_names
        if self.config.use_tfidf:
            tfidf_names = [f"tfidf_{i}" for i in range(self.tfidf.max_features)]
            numeric_names = numeric_names + tfidf_names
        
        self.feature_names = numeric_names
        self._fitted = True
        
        return self
    
    def transform(self, records: List[Dict[str, Any]]) -> Tuple[np.ndarray, np.ndarray]:
        """Transform records to feature matrix and labels."""
        if not self._fitted:
            raise ValueError("FeatureExtractor must be fitted first")
        
        raw_features_list = [self.extract_raw_features(r) for r in records]
        
        # Task type encoding
        task_types = [rf["task_type_raw"] for rf in raw_features_list]
        task_encoded = self.task_encoder.transform(task_types).reshape(-1, 1)
        
        # Numeric features
        numeric_features = []
        for rf in raw_features_list:
            numeric_row = [v for k, v in rf.items() if k != "task_type_raw"]
            numeric_features.append(numeric_row)
        numeric_array = np.array(numeric_features, dtype=np.float32)
        
        if self.config.scale_numeric:
            numeric_array = self.scaler.transform(numeric_array)
        
        # Combine task type + numeric
        if self.config.use_task_type:
            combined_numeric = np.hstack([task_encoded, numeric_array])
        else:
            combined_numeric = numeric_array
        
        # TF-IDF
        if self.config.use_tfidf:
            prompts = [r.get("prompt", "") for r in records]
            tfidf_features = self.tfidf.transform(prompts).toarray()
            combined = np.hstack([combined_numeric, tfidf_features])
        else:
            combined = combined_numeric
        
        # Labels
        labels = np.array([1 if r.get("safe_to_quantize", False) else 0 for r in records])
        
        return combined, labels
    
    def fit_transform(self, records: List[Dict[str, Any]]) -> Tuple[np.ndarray, np.ndarray]:
        """Fit and transform in one step."""
        self.fit(records)
        return self.transform(records)
    
    def get_feature_names(self) -> List[str]:
        """Get feature names in order."""
        return self.feature_names.copy()
    
    def save(self, path: str):
        """Save fitted extractor."""
        import joblib
        joblib.dump({
            'config': self.config,
            'scaler': self.scaler,
            'task_encoder': self.task_encoder,
            'tfidf': self.tfidf,
            'feature_names': self.feature_names,
            'fitted': self._fitted
        }, path)
    
    @classmethod
    def load(cls, path: str) -> "FeatureExtractor":
        """Load fitted extractor."""
        import joblib
        data = joblib.load(path)
        extractor = cls(data['config'])
        extractor.scaler = data['scaler']
        extractor.task_encoder = data['task_encoder']
        extractor.tfidf = data['tfidf']
        extractor.feature_names = data['feature_names']
        extractor._fitted = data['fitted']
        return extractor


def create_splits(
    records: List[Dict[str, Any]],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_state: int = 42,
    stratify_by: str = "safe_to_quantize"
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Create train/validation/test splits with stratification.
    
    Avoids leakage by ensuring related prompt variants stay in same split.
    """
    from sklearn.model_selection import train_test_split
    
    # Use task_type + complexity_level as grouping key to avoid leakage
    # Prompts with same task_type and complexity_level are likely variants
    group_keys = [f"{r.get('task_type', 'unknown')}_{r.get('complexity_level', 0)}" for r in records]
    
    # Group by key
    groups = {}
    for i, key in enumerate(group_keys):
        if key not in groups:
            groups[key] = []
        groups[key].append(i)
    
    # Split groups
    group_items = list(groups.items())
    group_keys_list = [k for k, v in group_items]
    group_values_list = [v for k, v in group_items]
    
    # Get stratification labels for groups (majority label in group)
    group_labels = []
    for indices in group_values_list:
        labels = [records[i].get(stratify_by, False) for i in indices]
        # Majority vote
        group_labels.append(1 if sum(labels) > len(labels) / 2 else 0)
    
    # Split groups
    train_groups, test_groups, _, _ = train_test_split(
        group_items, group_labels,
        test_size=test_ratio,
        random_state=random_state,
        stratify=group_labels
    )
    
    # Further split train into train/val
    train_val_ratio = val_ratio / (train_ratio + val_ratio)
    train_groups_final, val_groups, _, _ = train_test_split(
        train_groups,
        [group_labels[group_keys_list.index(k)] for k, _ in train_groups],
        test_size=train_val_ratio,
        random_state=random_state,
        stratify=[group_labels[group_keys_list.index(k)] for k, _ in train_groups]
    )
    
    # Collect indices
    train_indices = [i for _, indices in train_groups_final for i in indices]
    val_indices = [i for _, indices in val_groups for i in indices]
    test_indices = [i for _, indices in test_groups for i in indices]
    
    train_records = [records[i] for i in train_indices]
    val_records = [records[i] for i in val_indices]
    test_records = [records[i] for i in test_indices]
    
    print(f"Split sizes: Train={len(train_records)}, Val={len(val_records)}, Test={len(test_records)}")
    print(f"Train safe%: {sum(1 for r in train_records if r.get('safe_to_quantize', False))/len(train_records)*100:.1f}%")
    print(f"Val safe%: {sum(1 for r in val_records if r.get('safe_to_quantize', False))/len(val_records)*100:.1f}%")
    print(f"Test safe%: {sum(1 for r in test_records if r.get('safe_to_quantize', False))/len(test_records)*100:.1f}%")
    
    return train_records, val_records, test_records


if __name__ == "__main__":
    # Test with B2 dataset
    with open("data/processed/b2_dataset.json", 'r') as f:
        records = json.load(f)
    
    print(f"Loaded {len(records)} records")
    
    # Test feature extraction
    extractor = FeatureExtractor(FeatureConfig(use_task_type=True, use_complexity_level=False))
    X, y = extractor.fit_transform(records)
    
    print(f"Feature matrix shape: {X.shape}")
    print(f"Labels shape: {y.shape}")
    print(f"Safe: {sum(y)}, Unsafe: {len(y) - sum(y)}")
    print(f"Feature names: {len(extractor.get_feature_names())}")
    
    # Test splits
    train, val, test = create_splits(records)
    
    print(f"\nTrain: {len(train)}, Val: {len(val)}, Test: {len(test)}")