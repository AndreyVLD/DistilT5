import re
import orjson

from difflib import SequenceMatcher
from tqdm import tqdm


def normalize_assertion(assertion: str) -> str:
    """Normalize assertion text for more reliable comparison"""
    # Remove whitespace
    assertion = re.sub(r'\s+', ' ', assertion).strip()

    # Remove variable names in certain cases
    assertion = re.sub(r'assertEquals\(\s*[^,]+,\s*([^)]+)\)', r'assertEquals(VALUE, \1)', assertion)

    # Normalize assertion method names
    assertion = re.sub(r'assert(Equals|That|True|False)', r'assert\1', assertion, flags=re.IGNORECASE)

    return assertion


def calculate_similarity(reference: str, candidate: str) -> float:
    """Calculate string similarity using SequenceMatcher"""
    return SequenceMatcher(None, reference, candidate).ratio()


# TODO Type this return value with the correct TypedDict
def evaluate_assertions(generated_assertions: str | list[str], reference_assertions: str | list[str]) \
        -> dict[str, float | list[float]]:
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
            "similarity_scores": []
        }

    # Normalize assertions
    normalized_generated = [normalize_assertion(a) for a in generated_list]
    normalized_reference = [normalize_assertion(a) for a in reference_list]

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
            sim = calculate_similarity(gen, ref)
            best_sim = max(best_sim, sim)
        similarity_scores.append(best_sim)

    # Calculate metrics
    precision = exact_matches / len(normalized_generated) if normalized_generated else 0
    recall = exact_matches / len(normalized_reference) if normalized_reference else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = exact_matches / max(len(normalized_generated), len(normalized_reference)) if max(
        len(normalized_generated), len(normalized_reference)) > 0 else 0

    return {
        "exact_matches": exact_matches,
        "generated_count": len(normalized_generated),
        "reference_count": len(normalized_reference),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
        "similarity_score_avg": sum(similarity_scores) / len(similarity_scores) if similarity_scores else 0,
        "similarity_scores": similarity_scores
    }


# TODO Type this return value with the correct TypedDict
# TODO Modularize this function to avoid code duplication with the evaluation in the pipeline
def evaluate_teacher(file_path: str) -> dict[str, float]:
    """
    Evaluate the teacher model assertions by comparing generated assertions with reference assertions.
    Args:
        file_path (str): Path to the JSONL file containing the data. It needs to have the following fields:
            - "original_target": The original target assertions.
            - "predicted_assertions": The generated assertions from the teacher model.
    Returns:
        dict: A dictionary containing the evaluation metrics:
            - precision
            - recall
            - f1
            - accuracy
            - similarity_score_avg
            - total_exact_matches
            - total_generated
            - total_reference
    """

    all_metrics = {
        "exact_matches": 0,
        "generated_count": 0,
        "reference_count": 0,
        "similarity_scores": [],
        "accuracy_scores": [],
        "f1_scores": []
    }

    with open(file_path, 'r') as f:
        data = [orjson.loads(line) for line in f]

    for entry in tqdm(data, desc="Evaluating teacher model assertions"):
        # Extract assertions
        reference_assertions = entry.get("original_target", "")
        generated_assertions = entry.get("predicted_assertions", "")

        # Evaluate assertions
        metrics = evaluate_assertions(generated_assertions, reference_assertions)

        # Update overall metrics
        all_metrics["exact_matches"] += metrics["exact_matches"]
        all_metrics["generated_count"] += metrics["generated_count"]
        all_metrics["reference_count"] += metrics["reference_count"]
        all_metrics["similarity_scores"].extend(metrics["similarity_scores"])
        all_metrics["accuracy_scores"].append(metrics["accuracy"])
        all_metrics["f1_scores"].append(metrics["f1"])

    if not all_metrics["similarity_scores"]:
        return {"precision": 0, "recall": 0, "f1": 0, "accuracy": 0, "similarity_score_avg": 0}

    # Calculate aggregate metrics
    overall_precision = all_metrics["exact_matches"] / all_metrics["generated_count"] if all_metrics[
        "generated_count"] else 0
    overall_recall = all_metrics["exact_matches"] / all_metrics["reference_count"] if all_metrics[
        "reference_count"] else 0
    overall_f1 = 2 * overall_precision * overall_recall / (overall_precision + overall_recall) \
        if (overall_precision + overall_recall) > 0 else 0

    # Average per-sample metrics
    avg_similarity = sum(all_metrics["similarity_scores"]) / len(all_metrics["similarity_scores"])
    avg_accuracy = sum(all_metrics["accuracy_scores"]) / len(all_metrics["accuracy_scores"]) if all_metrics[
        "accuracy_scores"] else 0

    results = {
        "precision": overall_precision,
        "recall": overall_recall,
        "f1": overall_f1,
        "accuracy": avg_accuracy,
        "similarity_score_avg": avg_similarity,
        "total_exact_matches": all_metrics["exact_matches"],
        "total_generated": all_metrics["generated_count"],
        "total_reference": all_metrics["reference_count"]
    }

    return results
