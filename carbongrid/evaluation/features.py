"""
Phase B1: Feature Extraction for Quantization-Safety Classifier.

Features that are available BEFORE inference.
NO post-inference measurements (no latency, energy, quality scores, outputs).
"""

import re
from dataclasses import dataclass, asdict
from typing import Dict, List, Any, Optional
from collections import Counter


@dataclass
class PromptFeatures:
    """Pre-inference features for classifier."""
    # Basic length features
    char_length: int
    word_count: int
    token_count_estimate: int  # Rough estimate: ~4 chars per token
    
    # Structural features
    has_question_mark: bool
    has_code_indicator: bool
    question_count: int
    sentence_count: int
    
    # Content indicators
    has_scientific_terms: bool
    scientific_term_count: int
    has_reasoning_indicators: bool
    reasoning_indicator_count: int
    has_code_keywords: bool
    code_keyword_count: int
    
    # Task type (one-hot will be created later)
    task_type: str
    
    # Complexity indicators
    avg_word_length: float
    unique_word_ratio: float  # Vocabulary richness
    has_numbers: bool
    number_count: int
    
    # Output length hints
    requested_length_hint: Optional[str] = None  # e.g., "in 3 sentences", "briefly"
    max_tokens_requested: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    def to_feature_vector(self, task_types: List[str]) -> List[float]:
        """Convert to numerical feature vector for ML."""
        features = [
            self.char_length,
            self.word_count,
            self.token_count_estimate,
            float(self.has_question_mark),
            float(self.has_code_indicator),
            self.question_count,
            self.sentence_count,
            float(self.has_scientific_terms),
            self.scientific_term_count,
            float(self.has_reasoning_indicators),
            self.reasoning_indicator_count,
            float(self.has_code_keywords),
            self.code_keyword_count,
            self.avg_word_length,
            self.unique_word_ratio,
            float(self.has_numbers),
            self.number_count,
            float(self.requested_length_hint is not None),
            self.max_tokens_requested or 0,
        ]
        
        # One-hot encode task type
        for tt in task_types:
            features.append(1.0 if self.task_type == tt else 0.0)
        
        return features


def extract_features(prompt: str, task_type: str) -> PromptFeatures:
    """
    Extract pre-inference features from a prompt.
    
    These features must NOT depend on model outputs, latency, energy, or quality scores.
    """
    text = prompt.strip()
    
    # Basic counts
    char_length = len(text)
    words = text.split()
    word_count = len(words)
    token_count_estimate = max(1, char_length // 4)  # Rough estimate
    
    # Structural
    has_question_mark = '?' in text
    question_count = text.count('?')
    
    # Sentence count (rough)
    sentence_count = len(re.split(r'[.!?]+', text)) - 1
    sentence_count = max(1, sentence_count)
    
    # Code indicators
    code_keywords = ['def ', 'class ', 'import ', 'from ', 'function', 'return ', 'if ', 'for ', 'while ', 
                     'try:', 'except:', 'lambda ', 'async ', 'await ', 'yield ', 'with ', 'as ',
                     'public ', 'private ', 'protected ', 'static ', 'void ', 'int ', 'string ',
                     'console.log', 'print(', 'System.out', 'std::', '#include', 'fn ', 'let ',
                     'const ', 'var ', '=>', 'async def', 'await ']
    code_keyword_count = sum(1 for kw in code_keywords if kw in text.lower())
    has_code_indicator = code_keyword_count > 0 or '```' in text or 'def ' in text
    
    # Scientific terms (capitalized technical terms, long words)
    sci_pattern = re.compile(r'\b[A-Z][a-z]{4,}\b|\b[a-z]{8,}\b')
    scientific_terms = sci_pattern.findall(text.lower())
    # Filter common words
    common_words = {'the', 'and', 'for', 'with', 'this', 'that', 'from', 'have', 'been', 'were', 
                    'their', 'there', 'which', 'when', 'what', 'where', 'who', 'how', 'why',
                    'about', 'into', 'more', 'some', 'such', 'only', 'other', 'than', 'then',
                    'them', 'these', 'those', 'upon', 'within', 'without', 'during', 'before',
                    'after', 'since', 'until', 'while', 'under', 'over', 'between', 'among'}
    scientific_terms = [w for w in scientific_terms if w not in common_words]
    scientific_term_count = len(scientific_terms)
    has_scientific_terms = scientific_term_count > 0
    
    # Reasoning indicators
    reasoning_words = ['because', 'therefore', 'thus', 'hence', 'so', 'if', 'then', 'implies', 
                       'follows', 'conclude', 'reason', 'logic', 'step', 'first', 'second',
                       'assume', 'suppose', 'given', 'premise', 'deduce', 'infer', 'prove',
                       'show', 'demonstrate', 'explain why', 'why does', 'how does']
    reasoning_count = sum(1 for w in reasoning_words if w in text.lower())
    has_reasoning_indicators = reasoning_count > 0
    
    # Numbers
    numbers = re.findall(r'\b\d+(?:\.\d+)?\b', text)
    number_count = len(numbers)
    has_numbers = number_count > 0
    
    # Vocabulary richness
    unique_words = len(set(w.lower() for w in words))
    unique_word_ratio = unique_words / max(word_count, 1)
    
    # Average word length
    avg_word_length = sum(len(w) for w in words) / max(word_count, 1)
    
    # Output length hints
    requested_length_hint = None
    max_tokens_requested = None
    length_patterns = [
        (r'in (\d+) sentences?', 'sentences'),
        (r'in (\d+) words?', 'words'),
        (r'briefly', 'brief'),
        (r'short', 'short'),
        (r'concise', 'concise'),
        (r'detailed', 'detailed'),
        (r'comprehensive', 'comprehensive'),
        (r'(\d+) paragraphs?', 'paragraphs'),
    ]
    for pattern, unit in length_patterns:
        match = re.search(pattern, text.lower())
        if match:
            requested_length_hint = f"{match.group(1)} {unit}" if match.groups() else unit
            if match.groups() and match.group(1).isdigit():
                max_tokens_requested = int(match.group(1)) * (20 if unit == 'sentences' else 10 if unit == 'words' else 50)
            break
    
    return PromptFeatures(
        char_length=char_length,
        word_count=word_count,
        token_count_estimate=token_count_estimate,
        has_question_mark=has_question_mark,
        has_code_indicator=has_code_indicator,
        question_count=question_count,
        sentence_count=sentence_count,
        has_scientific_terms=has_scientific_terms,
        scientific_term_count=scientific_term_count,
        has_reasoning_indicators=has_reasoning_indicators,
        reasoning_indicator_count=reasoning_count,
        has_code_keywords=has_code_indicator,
        code_keyword_count=code_keyword_count,
        avg_word_length=avg_word_length,
        unique_word_ratio=unique_word_ratio,
        has_numbers=has_numbers,
        number_count=number_count,
        task_type=task_type,
        requested_length_hint=requested_length_hint,
        max_tokens_requested=max_tokens_requested
    )


def get_feature_names(task_types: List[str]) -> List[str]:
    """Get feature names in order matching to_feature_vector."""
    base_names = [
        "char_length",
        "word_count", 
        "token_count_estimate",
        "has_question_mark",
        "has_code_indicator",
        "question_count",
        "sentence_count",
        "has_scientific_terms",
        "scientific_term_count",
        "has_reasoning_indicators",
        "reasoning_indicator_count",
        "has_code_keywords",
        "code_keyword_count",
        "avg_word_length",
        "unique_word_ratio",
        "has_numbers",
        "number_count",
        "has_length_hint",
        "max_tokens_requested",
    ]
    
    # Add one-hot task type names
    for tt in task_types:
        base_names.append(f"task_{tt}")
    
    return base_names


# Test
if __name__ == "__main__":
    test_prompts = [
        ("What is the capital of France?", "factual"),
        ("Explain how photosynthesis works in plants.", "scientific"),
        ("Write a Python function to compute factorial recursively.", "code"),
        ("If all A are B and some B are C, can we conclude some A are C? Explain.", "reasoning"),
        ("Summarize climate change causes in 3 sentences.", "summarization"),
        ("Extract the company name from: Microsoft was founded in 1975.", "extraction"),
        ("Write a haiku about machine learning.", "creative"),
        ("What is the difference between a compiler and an interpreter?", "explanation"),
    ]
    
    task_types = sorted(set(t for _, t in test_prompts))
    
    for prompt, task in test_prompts:
        features = extract_features(prompt, task)
        print(f"\nPrompt: {prompt[:60]}...")
        print(f"  Task: {task}")
        print(f"  Words: {features.word_count}, Chars: {features.char_length}")
        print(f"  Question: {features.has_question_mark}, Code: {features.has_code_indicator}")
        print(f"  Scientific: {features.has_scientific_terms} ({features.scientific_term_count})")
        print(f"  Reasoning: {features.has_reasoning_indicators} ({features.reasoning_indicator_count})")
        print(f"  Unique word ratio: {features.unique_word_ratio:.2f}")
        print(f"  Length hint: {features.requested_length_hint}")
        
        vec = features.to_feature_vector(task_types)
        print(f"  Feature vector length: {len(vec)}")
    
    print(f"\nFeature names ({len(get_feature_names(task_types))}):")
    for name in get_feature_names(task_types):
        print(f"  {name}")