"""
Phase B1: Modular Quality Evaluation for Quantization-Safety Dataset.

Different task types use different evaluation methods.
No single metric is used universally.
"""

import re
import ast
import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class QualityResult:
    """Result of quality evaluation."""
    score: float                    # 0.0 to 1.0
    method: str                     # Evaluation method used
    details: Dict[str, Any]         # Method-specific details
    passed: bool                    # Whether score meets threshold
    threshold: float                # Threshold used
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "score": self.score,
            "method": self.method,
            "details": self.details,
            "passed": self.passed,
            "threshold": self.threshold
        }


class QualityEvaluator(ABC):
    """Abstract base class for quality evaluators."""
    
    @abstractmethod
    def evaluate(self, fp16_output: str, int4_output: str, reference: str = "") -> QualityResult:
        """Evaluate INT4 output quality relative to FP16."""
        pass
    
    @abstractmethod
    def get_method_name(self) -> str:
        """Return the name of the evaluation method."""
        pass


class FactualEvaluator(QualityEvaluator):
    """
    Evaluator for factual Q&A.
    
    Uses exact match or key entity presence against reference answer.
    """
    
    def __init__(self, threshold: float = 0.8):
        self.threshold = threshold
    
    def evaluate(self, fp16_output: str, int4_output: str, reference: str = "") -> QualityResult:
        # Normalize outputs
        fp16_norm = self._normalize(fp16_output)
        int4_norm = self._normalize(int4_output)
        ref_norm = self._normalize(reference)
        
        # If reference exists, check if INT4 contains key information
        if ref_norm:
            # Extract key entities from reference
            ref_entities = self._extract_entities(ref_norm)
            int4_entities = self._extract_entities(int4_norm)
            
            if not ref_entities:
                score = 1.0 if int4_norm else 0.0
            else:
                matched = len(ref_entities & int4_entities)
                score = matched / len(ref_entities)
        else:
            # No reference: use FP16 as proxy reference
            fp16_entities = self._extract_entities(fp16_norm)
            int4_entities = self._extract_entities(int4_norm)
            
            if not fp16_entities:
                score = 1.0 if int4_norm else 0.0
            else:
                matched = len(fp16_entities & int4_entities)
                score = matched / len(fp16_entities)
        
        passed = score >= self.threshold
        
        return QualityResult(
            score=score,
            method=self.get_method_name(),
            details={
                "fp16_normalized": fp16_norm[:100],
                "int4_normalized": int4_norm[:100],
                "reference": ref_norm[:100] if ref_norm else "N/A",
                "threshold": self.threshold
            },
            passed=passed,
            threshold=self.threshold
        )
    
    def _normalize(self, text: str) -> str:
        return re.sub(r'[^\w\s]', '', text.lower().strip())
    
    def _extract_entities(self, text: str) -> set:
        """Extract key entities (numbers, capitalized words, key terms)."""
        entities = set()
        # Numbers
        entities.update(re.findall(r'\b\d+\b', text))
        # Capitalized words (potential proper nouns)
        entities.update(re.findall(r'\b[A-Z][a-z]+\b', text))
        # Keywords
        keywords = re.findall(r'\b\w{4,}\b', text)
        entities.update(keywords)
        return entities
    
    def get_method_name(self) -> str:
        return "factual_entity_match"


class ExtractionEvaluator(QualityEvaluator):
    """
    Evaluator for information extraction tasks.
    
    Checks for presence of required fields in structured output.
    """
    
    def __init__(self, threshold: float = 0.8):
        self.threshold = threshold
    
    def evaluate(self, fp16_output: str, int4_output: str, reference: str = "") -> QualityResult:
        # Define expected fields based on reference or common patterns
        expected_fields = self._extract_expected_fields(reference) if reference else self._infer_fields(fp16_output)
        
        fp16_fields = self._extract_fields(fp16_output, expected_fields)
        int4_fields = self._extract_fields(int4_output, expected_fields)
        
        if not expected_fields:
            score = 1.0
        else:
            matched = sum(1 for f in expected_fields if f in int4_fields)
            score = matched / len(expected_fields)
        
        passed = score >= self.threshold
        
        return QualityResult(
            score=score,
            method=self.get_method_name(),
            details={
                "expected_fields": list(expected_fields),
                "fp16_found": list(fp16_fields),
                "int4_found": list(int4_fields),
                "threshold": self.threshold
            },
            passed=passed,
            threshold=self.threshold
        )
    
    def _extract_expected_fields(self, reference: str) -> set:
        """Extract expected field names from reference answer."""
        # Look for patterns like "Field: value" or "field = value"
        fields = set()
        for line in reference.split('\n'):
            if ':' in line:
                field = line.split(':')[0].strip().lower()
                fields.add(field)
            elif '=' in line:
                field = line.split('=')[0].strip().lower()
                fields.add(field)
        return fields
    
    def _infer_fields(self, output: str) -> set:
        """Infer expected fields from FP16 output."""
        fields = set()
        for line in output.split('\n'):
            if ':' in line:
                field = line.split(':')[0].strip().lower()
                if len(field) > 2:
                    fields.add(field)
        return fields
    
    def _extract_fields(self, text: str, expected: set) -> set:
        """Extract which expected fields are present in text."""
        text_lower = text.lower()
        found = set()
        for field in expected:
            if field in text_lower:
                found.add(field)
        return found
    
    def get_method_name(self) -> str:
        return "extraction_field_presence"


class SummarizationEvaluator(QualityEvaluator):
    """
    Evaluator for summarization tasks.
    
    Uses semantic similarity (embedding-based) as a proxy.
    WARNING: Semantic similarity is a proxy, not ground truth quality.
    """
    
    def __init__(self, threshold: float = 0.75):
        self.threshold = threshold
        self._embedder = None
    
    def evaluate(self, fp16_output: str, int4_output: str, reference: str = "") -> QualityResult:
        # Use embedding similarity between INT4 and FP16 outputs
        # (FP16 acts as reference)
        similarity = self._embedding_similarity(fp16_output, int4_output)
        
        # Also check length ratio (too short = likely incomplete)
        fp16_len = len(fp16_output.split())
        int4_len = len(int4_output.split())
        length_ratio = min(int4_len / max(fp16_len, 1), 1.0) if fp16_len > 0 else 0
        
        # Combined score: similarity weighted more
        score = 0.7 * similarity + 0.3 * length_ratio
        score = min(max(score, 0.0), 1.0)
        
        passed = score >= self.threshold
        
        return QualityResult(
            score=score,
            method=self.get_method_name(),
            details={
                "embedding_similarity": similarity,
                "length_ratio": length_ratio,
                "fp16_words": fp16_len,
                "int4_words": int4_len,
                "note": "Semantic similarity is a proxy, not ground truth quality",
                "threshold": self.threshold
            },
            passed=passed,
            threshold=self.threshold
        )
    
    def _embedding_similarity(self, text1: str, text2: str) -> float:
        """Compute cosine similarity using sentence transformers if available."""
        if not text1.strip() or not text2.strip():
            return 0.0
        
        try:
            if self._embedder is None:
                from sentence_transformers import SentenceTransformer
                self._embedder = SentenceTransformer('all-MiniLM-L6-v2')
            
            emb1 = self._embedder.encode([text1])
            emb2 = self._embedder.encode([text2])
            
            # Cosine similarity
            import numpy as np
            sim = np.dot(emb1[0], emb2[0]) / (np.linalg.norm(emb1[0]) * np.linalg.norm(emb2[0]))
            return float(sim)
        except ImportError:
            # Fallback: word overlap Jaccard similarity
            return self._jaccard_similarity(text1, text2)
        except Exception as e:
            logger.warning(f"Embedding similarity failed: {e}, using Jaccard fallback")
            return self._jaccard_similarity(text1, text2)
    
    def _jaccard_similarity(self, text1: str, text2: str) -> float:
        """Fallback Jaccard similarity on word sets."""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        if not words1 or not words2:
            return 0.0
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        return intersection / union if union > 0 else 0.0
    
    def get_method_name(self) -> str:
        return "summarization_embedding_similarity"


class CodeEvaluator(QualityEvaluator):
    """
    Evaluator for code generation tasks.
    
    Checks syntax validity and optionally executes tests.
    """
    
    def __init__(self, threshold: float = 0.7):
        self.threshold = threshold
    
    def evaluate(self, fp16_output: str, int4_output: str, reference: str = "") -> QualityResult:
        fp16_syntax = self._check_syntax(fp16_output)
        int4_syntax = self._check_syntax(int4_output)
        
        # Extract code blocks
        fp16_code = self._extract_code(fp16_output)
        int4_code = self._extract_code(int4_output)
        
        syntax_score = 1.0 if int4_syntax else 0.0
        
        # Try execution if syntax is valid
        exec_score = 0.0
        exec_details = "not_attempted"
        
        if int4_syntax and int4_code:
            try:
                exec_result = self._execute_code(int4_code)
                exec_score = 1.0 if exec_result["success"] else 0.5
                exec_details = exec_result
            except Exception as e:
                exec_score = 0.3
                exec_details = f"execution_error: {str(e)}"
        
        # Structural similarity with FP16
        structure_score = self._structure_similarity(fp16_code, int4_code)
        
        # Weighted combination
        score = 0.4 * syntax_score + 0.3 * exec_score + 0.3 * structure_score
        score = min(max(score, 0.0), 1.0)
        
        passed = score >= self.threshold
        
        return QualityResult(
            score=score,
            method=self.get_method_name(),
            details={
                "fp16_syntax_valid": fp16_syntax,
                "int4_syntax_valid": int4_syntax,
                "execution": exec_details,
                "structure_similarity": structure_score,
                "threshold": self.threshold
            },
            passed=passed,
            threshold=self.threshold
        )
    
    def _check_syntax(self, text: str) -> bool:
        """Check if text contains valid Python syntax."""
        code = self._extract_code(text)
        if not code:
            return False
        try:
            ast.parse(code)
            return True
        except SyntaxError:
            return False
    
    def _extract_code(self, text: str) -> str:
        """Extract Python code from markdown code blocks or raw text."""
        # Try markdown code blocks
        import re
        code_blocks = re.findall(r'```(?:python)?\n(.*?)\n```', text, re.DOTALL)
        if code_blocks:
            return code_blocks[0].strip()
        
        # Try indented code
        lines = text.split('\n')
        code_lines = [l for l in lines if l.startswith('    ') or l.startswith('\t') or re.match(r'^\s*(def|class|if|for|while|import|from)', l)]
        if code_lines:
            return '\n'.join(code_lines)
        
        return text.strip()
    
    def _execute_code(self, code: str) -> Dict[str, Any]:
        """Execute code in a sandboxed environment."""
        try:
            # Create a restricted namespace
            namespace = {}
            exec(code, {"__builtins__": {}}, namespace)
            return {"success": True}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _structure_similarity(self, code1: str, code2: str) -> float:
        """Compare code structure (AST-based)."""
        try:
            ast1 = ast.parse(code1) if code1 else None
            ast2 = ast.parse(code2) if code2 else None
            
            if not ast1 or not ast2:
                return 0.0
            
            # Compare node types
            nodes1 = [type(n).__name__ for n in ast.walk(ast1)]
            nodes2 = [type(n).__name__ for n in ast.walk(ast2)]
            
            if not nodes1 or not nodes2:
                return 0.0
            
            from collections import Counter
            c1 = Counter(nodes1)
            c2 = Counter(nodes2)
            
            all_nodes = set(c1.keys()) | set(c2.keys())
            if not all_nodes:
                return 0.0
            
            intersection = sum(min(c1[n], c2[n]) for n in all_nodes)
            union = sum(max(c1[n], c2[n]) for n in all_nodes)
            
            return intersection / union if union > 0 else 0.0
        except Exception:
            return 0.0
    
    def get_method_name(self) -> str:
        return "code_syntax_execution_structure"


class ReasoningEvaluator(QualityEvaluator):
    """
    Evaluator for reasoning tasks.
    
    Checks for logical structure and key reasoning steps.
    """
    
    def __init__(self, threshold: float = 0.6):
        self.threshold = threshold
    
    def evaluate(self, fp16_output: str, int4_output: str, reference: str = "") -> QualityResult:
        # Check for reasoning keywords/structure
        reasoning_indicators = [
            'because', 'therefore', 'thus', 'hence', 'so',
            'if', 'then', 'implies', 'follows', 'conclude',
            'reason', 'logic', 'step', 'first', 'second',
            'assume', 'suppose', 'given', 'premise'
        ]
        
        fp16_indicators = sum(1 for w in reasoning_indicators if w in fp16_output.lower())
        int4_indicators = sum(1 for w in reasoning_indicators if w in int4_output.lower())
        
        # Also check length (reasoning typically needs some explanation)
        fp16_words = len(fp16_output.split())
        int4_words = len(int4_output.split())
        
        structure_score = min(int4_indicators / max(fp16_indicators, 1), 1.0) if fp16_indicators > 0 else 0.5
        length_score = min(int4_words / max(fp16_words, 1), 1.0) if fp16_words > 0 else 0.5
        
        # If reference exists, check for key answer
        answer_score = 1.0
        if reference:
            ref_key = self._extract_answer_key(reference)
            if ref_key and ref_key.lower() not in int4_output.lower():
                answer_score = 0.0
        
        score = 0.4 * structure_score + 0.3 * length_score + 0.3 * answer_score
        score = min(max(score, 0.0), 1.0)
        
        passed = score >= self.threshold
        
        return QualityResult(
            score=score,
            method=self.get_method_name(),
            details={
                "fp16_reasoning_indicators": fp16_indicators,
                "int4_reasoning_indicators": int4_indicators,
                "fp16_words": fp16_words,
                "int4_words": int4_words,
                "answer_present": answer_score > 0,
                "threshold": self.threshold
            },
            passed=passed,
            threshold=self.threshold
        )
    
    def _extract_answer_key(self, reference: str) -> str:
        """Extract key answer from reference."""
        # Simple heuristic: last number or short phrase
        words = reference.split()
        for w in reversed(words):
            if len(w) > 1:
                return w
        return ""
    
    def get_method_name(self) -> str:
        return "reasoning_structure_keywords"


class CreativeEvaluator(QualityEvaluator):
    """
    Evaluator for creative writing tasks.
    
    Uses semantic similarity as a loose proxy.
    WARNING: Very approximate - creative quality is subjective.
    """
    
    def __init__(self, threshold: float = 0.6):
        self.threshold = threshold
        self._embedder = None
    
    def evaluate(self, fp16_output: str, int4_output: str, reference: str = "") -> QualityResult:
        similarity = self._embedding_similarity(fp16_output, int4_output)
        
        # Check for creative elements
        creative_markers = ['metaphor', 'imagery', 'rhythm', 'rhyme', 'stanza', 'verse']
        int4_creative = sum(1 for m in creative_markers if m in int4_output.lower())
        fp16_creative = sum(1 for m in creative_markers if m in fp16_output.lower())
        
        creativity_score = min(int4_creative / max(fp16_creative, 1), 1.0) if fp16_creative > 0 else 0.5
        
        score = 0.7 * similarity + 0.3 * creativity_score
        score = min(max(score, 0.0), 1.0)
        
        passed = score >= self.threshold
        
        return QualityResult(
            score=score,
            method=self.get_method_name(),
            details={
                "embedding_similarity": similarity,
                "creativity_markers_fp16": fp16_creative,
                "creativity_markers_int4": int4_creative,
                "note": "Creative quality is subjective; semantic similarity is a loose proxy",
                "threshold": self.threshold
            },
            passed=passed,
            threshold=self.threshold
        )
    
    def _embedding_similarity(self, text1: str, text2: str) -> float:
        if not text1.strip() or not text2.strip():
            return 0.0
        try:
            if self._embedder is None:
                from sentence_transformers import SentenceTransformer
                self._embedder = SentenceTransformer('all-MiniLM-L6-v2')
            
            import numpy as np
            emb1 = self._embedder.encode([text1])
            emb2 = self._embedder.encode([text2])
            sim = np.dot(emb1[0], emb2[0]) / (np.linalg.norm(emb1[0]) * np.linalg.norm(emb2[0]))
            return float(sim)
        except Exception:
            # Jaccard fallback
            words1 = set(text1.lower().split())
            words2 = set(text2.lower().split())
            if not words1 or not words2:
                return 0.0
            return len(words1 & words2) / len(words1 | words2)
    
    def get_method_name(self) -> str:
        return "creative_embedding_similarity"


class ScientificEvaluator(QualityEvaluator):
    """
    Evaluator for scientific/technical explanations.
    
    Checks for key concepts and technical accuracy.
    """
    
    def __init__(self, threshold: float = 0.7):
        self.threshold = threshold
    
    def evaluate(self, fp16_output: str, int4_output: str, reference: str = "") -> QualityResult:
        # Extract technical terms from FP16 output
        tech_terms = self._extract_technical_terms(fp16_output)
        int4_terms = self._extract_technical_terms(int4_output)
        
        if tech_terms:
            concept_coverage = len(tech_terms & int4_terms) / len(tech_terms)
        else:
            concept_coverage = 1.0
        
        # Check for incorrect statements (basic heuristic)
        # This is a placeholder - real fact-checking would need external knowledge
        fp16_len = len(fp16_output.split())
        int4_len = len(int4_output.split())
        length_score = min(int4_len / max(fp16_len, 1), 1.0) if fp16_len > 0 else 0.5
        
        score = 0.7 * concept_coverage + 0.3 * length_score
        score = min(max(score, 0.0), 1.0)
        
        passed = score >= self.threshold
        
        return QualityResult(
            score=score,
            method=self.get_method_name(),
            details={
                "fp16_technical_terms": list(tech_terms),
                "int4_technical_terms": list(int4_terms),
                "concept_coverage": concept_coverage,
                "length_score": length_score,
                "threshold": self.threshold
            },
            passed=passed,
            threshold=self.threshold
        )
    
    def _extract_technical_terms(self, text: str) -> set:
        """Extract technical/scientific terms (capitalized, longer words)."""
        words = re.findall(r'\b[A-Z][a-z]{3,}\b|\b[a-z]{6,}\b', text.lower())
        # Filter common words
        common = {'the', 'and', 'for', 'with', 'this', 'that', 'from', 'have', 'been', 'were', 'their', 'there', 'which', 'when', 'what', 'where', 'who', 'how', 'why', 'can', 'will', 'would', 'could', 'should', 'may', 'might', 'must', 'shall', 'about', 'into', 'more', 'some', 'such', 'only', 'other', 'than', 'then', 'them', 'these', 'those', 'upon', 'within', 'without', 'during', 'before', 'after', 'since', 'until', 'while', 'under', 'over', 'between', 'among', 'through', 'across', 'against', 'beyond'}
        return {w for w in words if w not in common and len(w) > 3}
    
    def get_method_name(self) -> str:
        return "scientific_concept_coverage"


class ExplanationEvaluator(QualityEvaluator):
    """
    Evaluator for explanation tasks.
    
    Checks for structural completeness and clarity.
    """
    
    def __init__(self, threshold: float = 0.7):
        self.threshold = threshold
    
    def evaluate(self, fp16_output: str, int4_output: str, reference: str = "") -> QualityResult:
        # Check for explanation structure
        structure_words = ['first', 'second', 'then', 'next', 'finally', 'because', 'therefore', 'however', 'moreover', 'furthermore', 'additionally', 'consequently', 'in contrast', 'similarly', 'for example', 'for instance', 'specifically', 'in particular', 'namely', 'that is', 'i.e.', 'e.g.']
        
        fp16_struct = sum(1 for w in structure_words if w in fp16_output.lower())
        int4_struct = sum(1 for w in structure_words if w in int4_output.lower())
        
        structure_score = min(int4_struct / max(fp16_struct, 1), 1.0) if fp16_struct > 0 else 0.5
        
        # Length
        fp16_words = len(fp16_output.split())
        int4_words = len(int4_output.split())
        length_score = min(int4_words / max(fp16_words, 1), 1.0) if fp16_words > 0 else 0.5
        
        score = 0.6 * structure_score + 0.4 * length_score
        score = min(max(score, 0.0), 1.0)
        
        passed = score >= self.threshold
        
        return QualityResult(
            score=score,
            method=self.get_method_name(),
            details={
                "fp16_structure_markers": fp16_struct,
                "int4_structure_markers": int4_struct,
                "fp16_words": fp16_words,
                "int4_words": int4_words,
                "threshold": self.threshold
            },
            passed=passed,
            threshold=self.threshold
        )
    
    def get_method_name(self) -> str:
        return "explanation_structure_length"


class DefaultEvaluator(QualityEvaluator):
    """Fallback evaluator using embedding similarity."""
    
    def __init__(self, threshold: float = 0.7):
        self.threshold = threshold
        self._embedder = None
    
    def evaluate(self, fp16_output: str, int4_output: str, reference: str = "") -> QualityResult:
        similarity = self._embedding_similarity(fp16_output, int4_output)
        score = similarity
        
        passed = score >= self.threshold
        
        return QualityResult(
            score=score,
            method=self.get_method_name(),
            details={
                "embedding_similarity": similarity,
                "note": "Default evaluator - embedding similarity only",
                "threshold": self.threshold
            },
            passed=passed,
            threshold=self.threshold
        )
    
    def _embedding_similarity(self, text1: str, text2: str) -> float:
        if not text1.strip() or not text2.strip():
            return 0.0
        try:
            if self._embedder is None:
                from sentence_transformers import SentenceTransformer
                self._embedder = SentenceTransformer('all-MiniLM-L6-v2')
            
            import numpy as np
            emb1 = self._embedder.encode([text1])
            emb2 = self._embedder.encode([text2])
            sim = np.dot(emb1[0], emb2[0]) / (np.linalg.norm(emb1[0]) * np.linalg.norm(emb2[0]))
            return float(sim)
        except Exception:
            words1 = set(text1.lower().split())
            words2 = set(text2.lower().split())
            if not words1 or not words2:
                return 0.0
            return len(words1 & words2) / len(words1 | words2)
    
    def get_method_name(self) -> str:
        return "default_embedding_similarity"


# Registry of evaluators by task type
EVALUATORS = {
    "factual": FactualEvaluator,
    "extraction": ExtractionEvaluator,
    "summarization": SummarizationEvaluator,
    "code": CodeEvaluator,
    "reasoning": ReasoningEvaluator,
    "creative": CreativeEvaluator,
    "scientific": ScientificEvaluator,
    "explanation": ExplanationEvaluator,
}


def get_evaluator(task_type: str, threshold: float = 0.7) -> QualityEvaluator:
    """Get appropriate evaluator for task type."""
    evaluator_class = EVALUATORS.get(task_type, DefaultEvaluator)
    return evaluator_class(threshold=threshold)


def evaluate_pair(
    fp16_output: str,
    int4_output: str,
    task_type: str,
    reference: str = "",
    threshold: float = 0.7
) -> QualityResult:
    """Convenience function to evaluate a single pair."""
    evaluator = get_evaluator(task_type, threshold)
    return evaluator.evaluate(fp16_output, int4_output, reference)


# Example usage and testing
if __name__ == "__main__":
    # Quick test
    fp16 = "The capital of France is Paris. It is located in the northwestern part of the country."
    int4 = "The capital of France is Paris. It's a famous city in Europe."
    
    result = evaluate_pair(fp16, int4, "factual", "Paris", threshold=0.8)
    print(f"Factual test: score={result.score:.3f}, passed={result.passed}, method={result.method}")
    
    # Code test
    fp16_code = "def factorial(n):\n    if n == 0:\n        return 1\n    return n * factorial(n-1)"
    int4_code = "def factorial(n):\n    if n == 0:\n        return 1\n    else:\n        return n * factorial(n-1)"
    
    result = evaluate_pair(fp16_code, int4_code, "code", threshold=0.7)
    print(f"Code test: score={result.score:.3f}, passed={result.passed}, method={result.method}")