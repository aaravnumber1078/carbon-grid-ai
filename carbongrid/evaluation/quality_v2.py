"""
Phase B2: Improved Quality Evaluation for Quantization-Safety Dataset.

Modular evaluators per task type with reference answers and semantic similarity.
Includes evaluation confidence reporting.
"""

import re
import ast
import json
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
from collections import Counter

logger = logging.getLogger(__name__)


@dataclass
class QualityResult:
    """Result of quality evaluation."""
    score: float                    # 0.0 to 1.0
    method: str                     # Evaluation method used
    details: Dict[str, Any]         # Method-specific details
    passed: bool                    # Whether score meets threshold
    threshold: float                # Threshold used
    confidence: float               # Confidence in this evaluation (0.0-1.0)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "score": self.score,
            "method": self.method,
            "details": self.details,
            "passed": self.passed,
            "threshold": self.threshold,
            "confidence": self.confidence
        }


class QualityEvaluator(ABC):
    """Abstract base class for quality evaluators."""
    
    @abstractmethod
    def evaluate(self, fp16_output: str, int4_output: str, reference: str = "") -> QualityResult:
        pass
    
    @abstractmethod
    def get_method_name(self) -> str:
        pass


# ==================== Shared Utilities ====================

class EmbeddingSimilarity:
    """Lazy-loaded sentence transformer for semantic similarity."""
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        self.model_name = model_name
        self._embedder = None
    
    def _get_embedder(self):
        if self._embedder is None:
            from sentence_transformers import SentenceTransformer
            self._embedder = SentenceTransformer(self.model_name)
        return self._embedder
    
    def similarity(self, text1: str, text2: str) -> float:
        if not text1.strip() or not text2.strip():
            return 0.0
        try:
            import numpy as np
            embedder = self._get_embedder()
            emb1 = embedder.encode([text1])
            emb2 = embedder.encode([text2])
            sim = np.dot(emb1[0], emb2[0]) / (np.linalg.norm(emb1[0]) * np.linalg.norm(emb2[0]))
            return float(sim)
        except Exception as e:
            logger.warning(f"Embedding similarity failed: {e}, using Jaccard fallback")
            return self._jaccard(text1, text2)
    
    def _jaccard(self, text1: str, text2: str) -> float:
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        if not words1 or not words2:
            return 0.0
        return len(words1 & words2) / len(words1 | words2)


# Global instance for reuse
_embedding = EmbeddingSimilarity()


def normalize_text(text: str) -> str:
    """Normalize text for comparison."""
    return re.sub(r'[^\w\s]', '', text.lower().strip())


def extract_entities(text: str) -> set:
    """Extract key entities from text."""
    text = normalize_text(text)
    entities = set()
    # Numbers
    entities.update(re.findall(r'\b\d+\b', text))
    # Capitalized words (proper nouns)
    entities.update(re.findall(r'\b[a-z]{4,}\b', text))
    # Longer words
    entities.update(re.findall(r'\b\w{5,}\b', text))
    return entities


def extract_key_concepts(text: str, min_len: int = 4) -> set:
    """Extract technical/scientific concepts."""
    words = re.findall(r'\b[a-z]{%d,}\b' % min_len, text.lower())
    common = {'the', 'and', 'for', 'with', 'this', 'that', 'from', 'have', 'been', 'were',
              'their', 'there', 'which', 'when', 'what', 'where', 'who', 'how', 'why',
              'about', 'into', 'more', 'some', 'such', 'only', 'other', 'than', 'then',
              'them', 'these', 'those', 'upon', 'within', 'without', 'during', 'before',
              'after', 'since', 'until', 'while', 'under', 'over', 'between', 'among',
              'through', 'across', 'against', 'beyond', 'because', 'therefore', 'however'}
    return {w for w in words if w not in common and len(w) >= min_len}


# ==================== Task-Specific Evaluators ====================

class FactualEvaluator(QualityEvaluator):
    """
    Factual Q&A evaluator.
    
    Combines: reference answer entity match + semantic similarity
    """
    
    def __init__(self, threshold: float = 0.7):
        self.threshold = threshold
    
    def evaluate(self, fp16_output: str, int4_output: str, reference: str = "") -> QualityResult:
        fp16_norm = normalize_text(fp16_output)
        int4_norm = normalize_text(int4_output)
        
        # Semantic similarity (primary)
        sem_sim = _embedding.similarity(fp16_output, int4_output)
        
        # Entity match against reference (if available)
        if reference:
            ref_entities = extract_entities(reference)
            int4_entities = extract_entities(int4_norm)
            entity_score = len(ref_entities & int4_entities) / len(ref_entities) if ref_entities else 1.0
        else:
            # Use FP16 as reference
            fp16_entities = extract_entities(fp16_norm)
            int4_entities = extract_entities(int4_norm)
            entity_score = len(fp16_entities & int4_entities) / len(fp16_entities) if fp16_entities else 1.0
        
        # Combined score (semantic weighted more)
        score = 0.6 * sem_sim + 0.4 * entity_score
        score = min(max(score, 0.0), 1.0)
        
        passed = score >= self.threshold
        confidence = 0.8 if reference else 0.6
        
        return QualityResult(
            score=score,
            method=self.get_method_name(),
            details={
                "semantic_similarity": sem_sim,
                "entity_match": entity_score,
                "has_reference": bool(reference),
                "threshold": self.threshold
            },
            passed=passed,
            threshold=self.threshold,
            confidence=confidence
        )
    
    def get_method_name(self) -> str:
        return "factual_semantic_entity"


class ScientificEvaluator(QualityEvaluator):
    """
    Scientific/technical explanation evaluator.
    
    Combines: reference concept coverage + semantic similarity
    """
    
    def __init__(self, threshold: float = 0.7):
        self.threshold = threshold
    
    def evaluate(self, fp16_output: str, int4_output: str, reference: str = "") -> QualityResult:
        # Semantic similarity
        sem_sim = _embedding.similarity(fp16_output, int4_output)
        
        # Concept coverage
        if reference:
            ref_concepts = extract_key_concepts(reference)
            int4_concepts = extract_key_concepts(int4_output)
        else:
            ref_concepts = extract_key_concepts(fp16_output)
            int4_concepts = extract_key_concepts(int4_output)
        
        concept_score = len(ref_concepts & int4_concepts) / len(ref_concepts) if ref_concepts else 1.0
        
        # Length adequacy
        fp16_words = len(fp16_output.split())
        int4_words = len(int4_output.split())
        length_score = min(int4_words / max(fp16_words, 1), 1.0) if fp16_words > 0 else 0.5
        
        # Combined
        score = 0.5 * sem_sim + 0.3 * concept_score + 0.2 * length_score
        score = min(max(score, 0.0), 1.0)
        
        passed = score >= self.threshold
        confidence = 0.7 if reference else 0.5
        
        return QualityResult(
            score=score,
            method=self.get_method_name(),
            details={
                "semantic_similarity": sem_sim,
                "concept_coverage": concept_score,
                "length_ratio": length_score,
                "ref_concepts": list(ref_concepts)[:10],
                "int4_concepts": list(int4_concepts)[:10],
                "threshold": self.threshold
            },
            passed=passed,
            threshold=self.threshold,
            confidence=confidence
        )
    
    def get_method_name(self) -> str:
        return "scientific_semantic_concept"


class ExtractionEvaluator(QualityEvaluator):
    """
    Information extraction evaluator.
    
    Exact field/value comparison against reference.
    """
    
    def __init__(self, threshold: float = 0.8):
        self.threshold = threshold
    
    def evaluate(self, fp16_output: str, int4_output: str, reference: str = "") -> QualityResult:
        # Parse expected fields from reference
        expected_fields = self._parse_fields(reference) if reference else self._infer_fields(fp16_output)
        
        int4_fields = self._parse_fields(int4_output)
        fp16_fields = self._parse_fields(fp16_output)
        
        if not expected_fields:
            # Fallback to semantic similarity
            sem_sim = _embedding.similarity(fp16_output, int4_output)
            return QualityResult(
                score=sem_sim,
                method="extraction_fallback_semantic",
                details={"fallback": True, "semantic_similarity": sem_sim},
                passed=sem_sim >= self.threshold,
                threshold=self.threshold,
                confidence=0.4
            )
        
        # Check each expected field
        matched = 0
        field_details = {}
        for field, expected_val in expected_fields.items():
            found = False
            actual_val = ""
            if field in int4_fields:
                actual_val = int4_fields[field]
                if self._values_match(expected_val, actual_val):
                    matched += 1
                    found = True
            field_details[field] = {"expected": expected_val, "found": actual_val, "matched": found}
        
        score = matched / len(expected_fields)
        passed = score >= self.threshold
        
        return QualityResult(
            score=score,
            method=self.get_method_name(),
            details={
                "expected_fields": expected_fields,
                "field_details": field_details,
                "matched": matched,
                "total": len(expected_fields),
                "threshold": self.threshold
            },
            passed=passed,
            threshold=self.threshold,
            confidence=0.9
        )
    
    def _parse_fields(self, text: str) -> Dict[str, str]:
        """Parse key:value or field=value pairs from text."""
        fields = {}
        for line in text.split('\n'):
            line = line.strip()
            if ':' in line:
                parts = line.split(':', 1)
                fields[parts[0].strip().lower()] = parts[1].strip()
            elif '=' in line:
                parts = line.split('=', 1)
                fields[parts[0].strip().lower()] = parts[1].strip()
        return fields
    
    def _infer_fields(self, text: str) -> Dict[str, str]:
        """Infer expected fields from FP16 output."""
        return self._parse_fields(text)
    
    def _values_match(self, expected: str, actual: str) -> bool:
        """Fuzzy match for field values."""
        exp_norm = normalize_text(expected)
        act_norm = normalize_text(actual)
        if exp_norm == act_norm:
            return True
        # Partial match for longer values
        if len(exp_norm) > 10 and exp_norm in act_norm:
            return True
        if len(act_norm) > 10 and act_norm in exp_norm:
            return True
        # Number match
        exp_nums = re.findall(r'\d+', exp_norm)
        act_nums = re.findall(r'\d+', act_norm)
        if exp_nums and set(exp_nums) & set(act_nums):
            return True
        return False
    
    def get_method_name(self) -> str:
        return "extraction_exact_field_value"


class SummarizationEvaluator(QualityEvaluator):
    """
    Summarization evaluator.
    
    Semantic similarity + length constraint + key concept coverage
    """
    
    def __init__(self, threshold: float = 0.75):
        self.threshold = threshold
    
    def evaluate(self, fp16_output: str, int4_output: str, reference: str = "") -> QualityResult:
        # Primary: semantic similarity to FP16 (or reference)
        target = reference if reference else fp16_output
        sem_sim = _embedding.similarity(target, int4_output)
        
        # Length adequacy
        target_words = len(target.split())
        int4_words = len(int4_output.split())
        length_score = min(int4_words / max(target_words, 1), 1.0) if target_words > 0 else 0.5
        
        # Concept coverage
        if reference:
            target_concepts = extract_key_concepts(reference)
        else:
            target_concepts = extract_key_concepts(fp16_output)
        int4_concepts = extract_key_concepts(int4_output)
        concept_score = len(target_concepts & int4_concepts) / len(target_concepts) if target_concepts else 1.0
        
        # Combined
        score = 0.6 * sem_sim + 0.2 * length_score + 0.2 * concept_score
        score = min(max(score, 0.0), 1.0)
        
        passed = score >= self.threshold
        confidence = 0.7 if reference else 0.5
        
        return QualityResult(
            score=score,
            method=self.get_method_name(),
            details={
                "semantic_similarity": sem_sim,
                "length_ratio": length_score,
                "concept_coverage": concept_score,
                "threshold": self.threshold,
                "note": "Semantic similarity is a proxy, not ground truth quality"
            },
            passed=passed,
            threshold=self.threshold,
            confidence=confidence
        )
    
    def get_method_name(self) -> str:
        return "summarization_semantic_length_concept"


class ExplanationEvaluator(QualityEvaluator):
    """
    Explanation evaluator.
    
    Semantic similarity + structural completeness + concept coverage
    """
    
    def __init__(self, threshold: float = 0.7):
        self.threshold = threshold
    
    def evaluate(self, fp16_output: str, int4_output: str, reference: str = "") -> QualityResult:
        target = reference if reference else fp16_output
        
        # Semantic similarity
        sem_sim = _embedding.similarity(target, int4_output)
        
        # Structural markers
        structure_markers = [
            'first', 'second', 'then', 'next', 'finally', 'because', 'therefore',
            'however', 'moreover', 'furthermore', 'additionally', 'consequently',
            'in contrast', 'similarly', 'for example', 'for instance', 'specifically',
            'in particular', 'namely', 'that is', 'i.e.', 'e.g.', 'step', 'reason'
        ]
        target_struct = sum(1 for m in structure_markers if m in target.lower())
        int4_struct = sum(1 for m in structure_markers if m in int4_output.lower())
        struct_score = min(int4_struct / max(target_struct, 1), 1.0) if target_struct > 0 else 0.5
        
        # Concept coverage
        target_concepts = extract_key_concepts(target)
        int4_concepts = extract_key_concepts(int4_output)
        concept_score = len(target_concepts & int4_concepts) / len(target_concepts) if target_concepts else 1.0
        
        # Length
        target_words = len(target.split())
        int4_words = len(int4_output.split())
        length_score = min(int4_words / max(target_words, 1), 1.0) if target_words > 0 else 0.5
        
        score = 0.5 * sem_sim + 0.2 * struct_score + 0.2 * concept_score + 0.1 * length_score
        score = min(max(score, 0.0), 1.0)
        
        passed = score >= self.threshold
        confidence = 0.6 if reference else 0.4
        
        return QualityResult(
            score=score,
            method=self.get_method_name(),
            details={
                "semantic_similarity": sem_sim,
                "structure_score": struct_score,
                "concept_coverage": concept_score,
                "length_ratio": length_score,
                "threshold": self.threshold
            },
            passed=passed,
            threshold=self.threshold,
            confidence=confidence
        )
    
    def get_method_name(self) -> str:
        return "explanation_semantic_structure_concept"


class ReasoningEvaluator(QualityEvaluator):
    """
    Reasoning evaluator.
    
    Reference answer match + semantic similarity + logical structure
    """
    
    def __init__(self, threshold: float = 0.6):
        self.threshold = threshold
    
    def evaluate(self, fp16_output: str, int4_output: str, reference: str = "") -> QualityResult:
        target = reference if reference else fp16_output
        
        # Semantic similarity
        sem_sim = _embedding.similarity(target, int4_output)
        
        # Check for key answer from reference
        answer_score = 1.0
        if reference:
            answer_key = self._extract_answer_key(reference)
            if answer_key and answer_key.lower() not in int4_output.lower():
                answer_score = 0.0
        
        # Logical structure markers
        logic_markers = ['because', 'therefore', 'thus', 'hence', 'so', 'if', 'then',
                         'implies', 'follows', 'conclude', 'reason', 'logic', 'step',
                         'first', 'second', 'assume', 'suppose', 'given', 'premise',
                         'deduce', 'infer', 'prove', 'show', 'demonstrate']
        target_logic = sum(1 for m in logic_markers if m in target.lower())
        int4_logic = sum(1 for m in logic_markers if m in int4_output.lower())
        logic_score = min(int4_logic / max(target_logic, 1), 1.0) if target_logic > 0 else 0.5
        
        # Length adequacy
        target_words = len(target.split())
        int4_words = len(int4_output.split())
        length_score = min(int4_words / max(target_words, 1), 1.0) if target_words > 0 else 0.5
        
        score = 0.4 * sem_sim + 0.3 * answer_score + 0.2 * logic_score + 0.1 * length_score
        score = min(max(score, 0.0), 1.0)
        
        passed = score >= self.threshold
        confidence = 0.7 if reference else 0.5
        
        return QualityResult(
            score=score,
            method=self.get_method_name(),
            details={
                "semantic_similarity": sem_sim,
                "answer_present": answer_score > 0,
                "logic_score": logic_score,
                "length_ratio": length_score,
                "threshold": self.threshold
            },
            passed=passed,
            threshold=self.threshold,
            confidence=confidence
        )
    
    def _extract_answer_key(self, reference: str) -> str:
        # Look for final answer patterns
        patterns = [
            r'answer[:\s]+([^\n]+)',
            r'result[:\s]+([^\n]+)',
            r'=\s*([\d\.\-]+)',
            r'is\s+([\w\.\-]+)\.?$'
        ]
        for pattern in patterns:
            match = re.search(pattern, reference, re.IGNORECASE)
            if match:
                return match.group(1).strip()
        # Fallback: last sentence
        sentences = reference.split('.')
        if sentences:
            return sentences[-1].strip()
        return ""
    
    def get_method_name(self) -> str:
        return "reasoning_semantic_answer_logic"


class CodeEvaluator(QualityEvaluator):
    """
    Code evaluator.
    
    Syntax validation + execution (where possible) + structural similarity
    """
    
    def __init__(self, threshold: float = 0.7):
        self.threshold = threshold
    
    def evaluate(self, fp16_output: str, int4_output: str, reference: str = "") -> QualityResult:
        # Extract code
        fp16_code = self._extract_code(fp16_output)
        int4_code = self._extract_code(int4_output)
        ref_code = self._extract_code(reference) if reference else fp16_code
        
        # Syntax check
        fp16_syntax = self._check_syntax(fp16_code)
        int4_syntax = self._check_syntax(int4_code)
        
        syntax_score = 1.0 if int4_syntax else 0.0
        
        # Execution test (if we have a test case)
        exec_score = 0.0
        exec_details = "not_attempted"
        if int4_syntax and int4_code:
            try:
                result = self._execute_code(int4_code)
                exec_score = 1.0 if result["success"] else 0.3
                exec_details = result
            except Exception as e:
                exec_score = 0.2
                exec_details = f"execution_error: {str(e)}"
        
        # Structural similarity (AST)
        structure_score = self._ast_similarity(ref_code, int4_code)
        
        # Semantic similarity fallback
        sem_sim = _embedding.similarity(fp16_code or fp16_output, int4_code or int4_output)
        
        # Combined
        score = 0.3 * syntax_score + 0.3 * exec_score + 0.2 * structure_score + 0.2 * sem_sim
        score = min(max(score, 0.0), 1.0)
        
        passed = score >= self.threshold
        confidence = 0.8 if exec_score > 0 else 0.5
        
        return QualityResult(
            score=score,
            method=self.get_method_name(),
            details={
                "fp16_syntax_valid": fp16_syntax,
                "int4_syntax_valid": int4_syntax,
                "execution": exec_details,
                "ast_similarity": structure_score,
                "semantic_similarity": sem_sim,
                "threshold": self.threshold
            },
            passed=passed,
            threshold=self.threshold,
            confidence=confidence
        )
    
    def _extract_code(self, text: str) -> str:
        code_blocks = re.findall(r'```(?:python)?\n(.*?)\n```', text, re.DOTALL)
        if code_blocks:
            return code_blocks[0].strip()
        # Indented or def/class lines
        lines = text.split('\n')
        code_lines = []
        for l in lines:
            stripped = l.lstrip()
            if stripped.startswith(('def ', 'class ', 'if ', 'for ', 'while ', 'import ', 'from ',
                                    'try:', 'except:', 'with ', 'return ', 'print(')):
                code_lines.append(l)
        if code_lines:
            return '\n'.join(code_lines)
        return text.strip()
    
    def _check_syntax(self, code: str) -> bool:
        if not code:
            return False
        try:
            ast.parse(code)
            return True
        except SyntaxError:
            return False
    
    def _execute_code(self, code: str) -> Dict[str, Any]:
        try:
            namespace = {}
            exec(code, {"__builtins__": {}}, namespace)
            return {"success": True}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def _ast_similarity(self, code1: str, code2: str) -> float:
        if not code1 or not code2:
            return 0.0
        try:
            ast1 = ast.parse(code1)
            ast2 = ast.parse(code2)
            nodes1 = [type(n).__name__ for n in ast.walk(ast1)]
            nodes2 = [type(n).__name__ for n in ast.walk(ast2)]
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
        return "code_syntax_ast_execution"


class CreativeEvaluator(QualityEvaluator):
    """
    Creative writing evaluator.
    
    Semantic similarity only (explicitly a proxy)
    """
    
    def __init__(self, threshold: float = 0.65):
        self.threshold = threshold
    
    def evaluate(self, fp16_output: str, int4_output: str, reference: str = "") -> QualityResult:
        sem_sim = _embedding.similarity(fp16_output, int4_output)
        
        # Length check
        fp16_words = len(fp16_output.split())
        int4_words = len(int4_output.split())
        length_score = min(int4_words / max(fp16_words, 1), 1.0) if fp16_words > 0 else 0.5
        
        score = 0.8 * sem_sim + 0.2 * length_score
        score = min(max(score, 0.0), 1.0)
        
        passed = score >= self.threshold
        confidence = 0.3  # Low - creative quality is subjective
        
        return QualityResult(
            score=score,
            method=self.get_method_name(),
            details={
                "semantic_similarity": sem_sim,
                "length_ratio": length_score,
                "fp16_words": fp16_words,
                "int4_words": int4_words,
                "threshold": self.threshold,
                "note": "Creative quality is subjective; semantic similarity is a loose proxy"
            },
            passed=passed,
            threshold=self.threshold,
            confidence=confidence
        )
    
    def get_method_name(self) -> str:
        return "creative_semantic_proxy"


# ==================== Registry ====================

EVALUATORS_V2 = {
    "factual": (FactualEvaluator, 0.7),
    "scientific": (ScientificEvaluator, 0.7),
    "extraction": (ExtractionEvaluator, 0.8),
    "summarization": (SummarizationEvaluator, 0.75),
    "explanation": (ExplanationEvaluator, 0.7),
    "reasoning": (ReasoningEvaluator, 0.6),
    "coding": (CodeEvaluator, 0.7),
    "creative": (CreativeEvaluator, 0.65),
}


def get_evaluator_v2(task_type: str, threshold: Optional[float] = None) -> QualityEvaluator:
    """Get appropriate evaluator for task type with default threshold."""
    evaluator_class, default_thresh = EVALUATORS_V2.get(task_type, (None, 0.7))
    if evaluator_class is None:
        from carbongrid.evaluation.quality import DefaultEvaluator
        return DefaultEvaluator(threshold=threshold or 0.7)
    return evaluator_class(threshold=threshold or default_thresh)


def evaluate_pair_v2(
    fp16_output: str,
    int4_output: str,
    task_type: str,
    reference: str = "",
    threshold: Optional[float] = None
) -> QualityResult:
    """Evaluate with improved evaluators."""
    evaluator = get_evaluator_v2(task_type, threshold)
    return evaluator.evaluate(fp16_output, int4_output, reference)


# ==================== Calibration Utility ====================

def calibrate_threshold(dataset_records: List[Dict], thresholds: List[float] = None) -> Dict[str, Any]:
    """
    Show how labels change under different thresholds.
    
    Args:
        dataset_records: List of records with quality_score_int4 and quality_score_fp16
        thresholds: List of thresholds to test
    """
    if thresholds is None:
        thresholds = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]  # degradation tolerances
    
    results = {}
    for t in thresholds:
        safe = 0
        unsafe = 0
        for r in dataset_records:
            # Use quality_difference (int4 - fp16, negative means degradation)
            diff = r.get("quality_difference", 0)
            if diff >= -t:  # degradation within tolerance
                safe += 1
            else:
                unsafe += 1
        results[t] = {"safe": safe, "unsafe": unsafe, "total": safe + unsafe, "safe_pct": safe / (safe + unsafe) * 100}
    
    return results


if __name__ == "__main__":
    # Quick test
    fp16 = "The capital of France is Paris. It is located in the northwestern part of the country."
    int4 = "The capital of France is Paris. It's a famous city in Europe."
    
    result = evaluate_pair_v2(fp16, int4, "factual", "Paris", threshold=0.7)
    print(f"Factual test: score={result.score:.3f}, passed={result.passed}, confidence={result.confidence}")
    print(f"  Method: {result.method}")
    print(f"  Details: {result.details}")