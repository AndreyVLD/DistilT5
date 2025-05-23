import logging
import re
import orjson
import code_bert_score
import torch
from rouge_score import rouge_scorer
from difflib import SequenceMatcher
from tqdm import tqdm
from typing import TypedDict, Callable, cast
from codebleu import calc_codebleu

# Only show ERROR+ messages, hide WARNINGs (including the data-flow error from codebleu)
logging.getLogger().setLevel(logging.ERROR)


class AssertionEvalResult(TypedDict):
    """
    TypedDict for the result of evaluate_assertions().
    """
    exact_matches: int  # number of assertions that matched exactly
    generated_count: int  # total assertions generated
    reference_count: int  # total reference assertions
    precision: float  # exact match / generated_count
    recall: float  # exact match / reference_count
    f1: float  # harmonic mean of precision & recall
    accuracy: float  # exact match / max(generated_count, reference_count)
    similarity_score_avg: float  # mean of best per-assertion similarity scores
    similarity_scores: list[float]  # list of best similarity per generated assertion
    codeblue_score: float  # CodeBLEU score
    codebert_score: float  # CodeBERT score
    rougeL: float  # ROUGE-L score
    # Add new metrics here as needed, e.g.:


class ComputeAllResult(TypedDict):
    """
    Return type for MetricsEvaluator.compute_all()
    """
    precision: float  # overall exact‐match precision
    recall: float  # overall exact‐match recall
    f1: float  # harmonic mean of precision & recall
    accuracy: float  # mean of per‐sample accuracy scores
    similarity_score_avg: float  # mean of per‐sample similarity scores
    codeblue_avg: float  # mean of per‐sample CodeBLEU scores
    codebert_avg: float  # mean of per‐sample CodeBERT scores
    rougeL_avg: float  # mean of per‐sample ROUGE-L scores
    # Add new metrics here as needed, e.g.:
    # mutation_score_avg: float


class MetricsEvaluator:
    exact_matches: int
    generated_count: int
    reference_count: int
    similarity_scores: list[float]
    accuracy_scores: list[float]
    f1_scores: list[float]
    codebleu_scores: list[float]
    codebert_scores: list[float]
    rougeL_scores: list[float]
    _metric_funcs: dict[str, Callable[[], float]]
    _rouge_scorer: rouge_scorer.RougeScorer

    # Extend with new metrics here

    def __init__(self):
        self.reset()

    def reset(self) -> None:
        """
        Reset the metrics evaluator to its initial state.
        """

        # Initialize accumulators for metrics
        self.exact_matches: int = 0
        self.generated_count: int = 0
        self.reference_count: int = 0
        self.similarity_scores: list[float] = []
        self.accuracy_scores: list[float] = []
        self.f1_scores: list[float] = []
        self.codebleu_scores: list[float] = []
        self.codebert_scores: list[float] = []
        self.rougeL_scores: list[float] = []
        # Initialize new metrics accumulators here

        # Register metric computation methods
        self._metric_funcs = {
            'precision': self._precision,
            'recall': self._recall,
            'f1': self._f1,
            'accuracy': self._accuracy,
            'similarity_score_avg': self._similarity_score_avg,
            'codeblue_avg': self._codeblue_avg,
            'codebert_avg': self._codebert_avg,
            'rougeL_avg': self._rougeL_avg,  # Placeholder for ROUGE-L
            # Extend with new metrics aggregation functions here
        }

        self._rouge_scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

    @staticmethod
    def normalize_assertion(assertion: str) -> str:
        """Normalize assertion text for more reliable comparison"""
        # Remove whitespace
        assertion = re.sub(r'\s+', ' ', assertion).strip()

        # Remove variable names in certain cases
        assertion = re.sub(r'assertEquals\(\s*[^,]+,\s*([^)]+)\)', r'assertEquals(VALUE, \1)', assertion)

        # Normalize assertion method names
        assertion = re.sub(r'assert(Equals|That|True|False)', r'assert\1', assertion, flags=re.IGNORECASE)

        return assertion

    @staticmethod
    def calculate_similarity(reference: str, candidate: str) -> float:
        """Calculate string similarity using SequenceMatcher"""
        return SequenceMatcher(None, reference, candidate).ratio()

    def evaluate_assertions(self, generated_assertions: str | list[str],
                            reference_assertions: str | list[str]) -> AssertionEvalResult:
        """Evaluate the quality of generated assertions against reference assertions"""
        # Parse individual assertions if provided as multiline string
        if isinstance(generated_assertions, str):
            # Split by semicolons or newlines
            generated_list = re.split(r'[;\n]', generated_assertions)
            generated_list = [a.strip() + ';' for a in generated_list if a.strip()]
        else:
            generated_list = generated_assertions

        if isinstance(reference_assertions, str):
            reference_list = re.split(r'[;\n]', reference_assertions)
            reference_list = [a.strip() + ';' for a in reference_list if a.strip()]
        else:
            reference_list = reference_assertions

        # Special case handling for empty lists
        if not generated_list or not reference_list:
            return {
                "exact_matches": 0,
                "generated_count": len(generated_list) if generated_list else 0,
                "reference_count": len(reference_list) if reference_list else 0,
                "precision": 0,
                "recall": 0,
                "f1": 0,
                "accuracy": 0,
                "similarity_score_avg": 0,
                "similarity_scores": [],
                "codeblue_score": 0,
                "codebert_score": 0,
                "rougeL": 0,
            }

        # Normalize assertions
        normalized_generated = [self.normalize_assertion(a) for a in generated_list]
        normalized_reference = [self.normalize_assertion(a) for a in reference_list]

        # Calculate exact matches
        exact_matches = 0
        for gen in normalized_generated:
            if gen in normalized_reference:
                exact_matches += 1

        # Calculate similarity scores
        similarity_scores = []
        for gen in normalized_generated:
            best_sim = 0
            for ref in normalized_reference:
                sim = self.calculate_similarity(gen, ref)
                best_sim = max(best_sim, sim)
            similarity_scores.append(best_sim)

        # Calculate metrics
        precision = exact_matches / len(normalized_generated) if normalized_generated else 0
        recall = exact_matches / len(normalized_reference) if normalized_reference else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        accuracy = exact_matches / max(len(normalized_generated), len(normalized_reference)) if max(
            len(normalized_generated), len(normalized_reference)) > 0 else 0

        # Compute advanced metrics
        joined_generated = '\n.'.join(normalized_generated)
        joined_reference = '\n.'.join(normalized_reference)
        codebleu_score = calc_codebleu([joined_reference],
                                       [joined_generated], 'java')
        codebert_score: torch.Tensor = code_bert_score.score([joined_generated], [joined_reference], lang='java')
        rougeL_score = self._rouge_scorer.score(joined_generated, joined_reference)['rougeL'].fmeasure

        return {
            "exact_matches": exact_matches,
            "generated_count": len(normalized_generated),
            "reference_count": len(normalized_reference),
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "accuracy": accuracy,
            "similarity_score_avg": sum(similarity_scores) / len(similarity_scores) if similarity_scores else 0,
            "similarity_scores": similarity_scores,
            "codeblue_score": codebleu_score['codebleu'],
            "codebert_score": codebert_score[2].item(),  # F1
            "rougeL": rougeL_score,
        }

    def update(self, result: AssertionEvalResult) -> None:
        """
        Update internal state from evaluate_assertions() output.
        """
        self.exact_matches += result.get("exact_matches", 0)
        self.generated_count += result.get("generated_count", 0)
        self.reference_count += result.get("reference_count", 0)
        self.similarity_scores.extend(result.get("similarity_scores", []))
        self.accuracy_scores.append(result.get("accuracy", 0))
        self.f1_scores.append(result.get("f1", 0))
        self.codebleu_scores.append(result.get("codeblue_score", 0))
        self.codebert_scores.append(result.get("codebert_score", 0))
        self.rougeL_scores.append(result.get("rougeL", 0))

    def _precision(self) -> float:
        return self.exact_matches / self.generated_count if self.generated_count else 0

    def _recall(self) -> float:
        return self.exact_matches / self.reference_count if self.reference_count else 0

    def _f1(self) -> float:
        p = self._precision()
        r = self._recall()
        return (2 * p * r / (p + r)) if (p + r) else 0

    def _accuracy(self) -> float:
        return sum(self.accuracy_scores) / len(self.accuracy_scores) if self.accuracy_scores else 0

    def _similarity_score_avg(self) -> float:
        return sum(self.similarity_scores) / len(self.similarity_scores) if self.similarity_scores else 0

    def _codeblue_avg(self) -> float:
        return sum(self.codebleu_scores) / len(self.codebleu_scores) if self.codebleu_scores else 0

    def _codebert_avg(self) -> float:
        return sum(self.codebert_scores) / len(self.codebert_scores) if self.codebert_scores else 0

    def _rougeL_avg(self) -> float:
        return sum(self.rougeL_scores) / len(self.rougeL_scores) if self.rougeL_scores else 0

    # Add new metric functions here

    def compute_all(self) -> ComputeAllResult:
        """
        Compute all registered metrics in one call.
        """
        return cast(ComputeAllResult, {name: func() for name, func in self._metric_funcs.items()})


def evaluate_teacher(file_path: str) -> ComputeAllResult:
    """
    Evaluate the teacher model assertions by comparing generated assertions with reference assertions.
    Args:
        file_path (str): Path to the JSONL file containing the data. It needs to have the following fields:
            - "original_target": The original target assertions.
            - "predicted_assertions": The generated assertions from the teacher model.
    Returns:
        dict: A dictionary containing the evaluation metrics
    """

    evaluator = MetricsEvaluator()

    with open(file_path, 'r') as f:
        data = [orjson.loads(line) for line in f]

    for entry in tqdm(data, desc="Evaluating teacher model assertions"):
        # Extract assertions
        reference_assertions = entry.get("original_target", "")
        generated_assertions = entry.get("predicted_assertions", "")

        # Evaluate assertions
        metrics = evaluator.evaluate_assertions(generated_assertions, reference_assertions)

        # Update overall metrics
        evaluator.update(metrics)

    results = evaluator.compute_all()
    return results
