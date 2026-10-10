"""
Evaluation harness for the aspect/polarity extraction task — §14 of the
reference doc. One implementation, imported by both the zero-shot baseline
notebook and the post-fine-tune evaluation notebook, so the two conditions
are scored identically (the §10.3 fairness requirement applies to scoring
code, not just prompts).

Built and unit-tested (see tests/test_eval_metrics.py) before this project
calls a model for real — §12.1's reasoning: validate the harness on data you
already know the right answer for, not on the first live model output.
"""

from __future__ import annotations

import json
import re

ALLOWED_POLARITIES = {"positive", "negative", "neutral"}


# ---------------------------------------------------------------------------
# Parsing (§12.4)
# ---------------------------------------------------------------------------

def _is_schema_valid(obj) -> bool:
    """Structural check for M1: must be a dict with an "aspects" list, each
    item a dict with a non-empty "aspect" string and a "sentiment" drawn
    from the three permitted values. A bare list, or a dict shaped like
    {"battery": "positive"}, is schema drift, not a valid parse — even
    though json.loads() would happily accept the bare list on its own.
    """
    if not isinstance(obj, dict):
        return False
    if "aspects" not in obj or not isinstance(obj["aspects"], list):
        return False
    for item in obj["aspects"]:
        if not isinstance(item, dict):
            return False
        if not isinstance(item.get("aspect"), str) or not item["aspect"].strip():
            return False
        if item.get("sentiment") not in ALLOWED_POLARITIES:
            return False
    return True


def parse_strict(raw: str) -> dict | None:
    """The output must parse and conform to schema as-is. No cleanup."""
    try:
        obj = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return None
    return obj if _is_schema_valid(obj) else None


_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL)


def parse_lenient(raw: str) -> dict | None:
    """Strips markdown code fences and/or leading/trailing prose, then
    applies the same strict schema check. Recovers *presentation* failures
    (§12.3: code fences, "Sure! Here's the JSON:") — it does NOT attempt to
    coerce a wrong structure (e.g. a bare list) into the right one. That
    distinction is the whole point of reporting strict vs. lenient
    separately (§12.4): fixing presentation is fair; fixing comprehension
    would flatter the model instead of measuring it.
    """
    text = raw.strip()
    fence_match = _FENCE_RE.search(text)
    if fence_match:
        text = fence_match.group(1).strip()
    else:
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end != -1 and end > start:
            text = text[start:end + 1]
    return parse_strict(text)


# ---------------------------------------------------------------------------
# Scoring (§14.1)
# ---------------------------------------------------------------------------

def _normalize(aspect: str) -> str:
    # M2's definition: exact string match, case-insensitive, whitespace-normalised.
    return re.sub(r"\s+", " ", aspect.strip().lower())


def _prf(tp: int, n_pred: int, n_gold: int) -> tuple[float, float, float]:
    p = tp / n_pred if n_pred else 0.0
    r = tp / n_gold if n_gold else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f1


def score_records(records: list[dict], parser=parse_strict) -> dict:
    """
    records: list of {
        "sentence_id": str,
        "gold_aspects": [{"aspect": str, "sentiment": str}, ...],  # [] for zero-aspect gold
        "raw_output": str,   # the model's raw generation for this sentence
    }
    parser: parse_strict or parse_lenient — which failure modes count against M1.

    Returns M1-M5 plus the supporting precision/recall for M2 and M4.
    Unparseable output contributes zero predictions and is NOT excluded from
    any denominator (§14.2) — it scores as fully wrong, not ignored.
    """
    n = len(records)
    n_valid = 0
    n_gold_aspects = n_pred_aspects = tp_aspect = tp_joint = 0
    n_correct_extractions = n_correct_polarity = 0
    n_empty_gold = n_empty_correct = 0

    for r in records:
        gold_pairs = [(_normalize(a["aspect"]), a["sentiment"]) for a in r["gold_aspects"]]
        gold_polarity = dict(gold_pairs)  # first occurrence wins on duplicate spans
        gold_set = set(a for a, _ in gold_pairs)
        is_gold_empty = len(gold_pairs) == 0
        n_empty_gold += is_gold_empty

        parsed = parser(r["raw_output"])
        if parsed is not None:
            n_valid += 1
            pred_pairs = [
                (_normalize(item["aspect"]), item.get("sentiment"))
                for item in parsed.get("aspects", [])
            ]
        else:
            pred_pairs = []
        pred_polarity = dict(pred_pairs)
        pred_set = set(a for a, _ in pred_pairs)

        if is_gold_empty and len(pred_set) == 0:
            n_empty_correct += 1

        n_gold_aspects += len(gold_set)
        n_pred_aspects += len(pred_set)
        matched = gold_set & pred_set
        tp_aspect += len(matched)

        for a in matched:
            n_correct_extractions += 1
            if pred_polarity.get(a) == gold_polarity.get(a):
                n_correct_polarity += 1
                tp_joint += 1

    m2_p, m2_r, m2_f1 = _prf(tp_aspect, n_pred_aspects, n_gold_aspects)
    m4_p, m4_r, m4_f1 = _prf(tp_joint, n_pred_aspects, n_gold_aspects)

    return {
        "n_records": n,
        "M1_json_validity_rate": n_valid / n if n else 0.0,
        "M2_aspect_precision": m2_p,
        "M2_aspect_recall": m2_r,
        "M2_aspect_f1": m2_f1,
        "M3_polarity_accuracy": (n_correct_polarity / n_correct_extractions
                                  if n_correct_extractions else 0.0),
        "M4_joint_precision": m4_p,
        "M4_joint_recall": m4_r,
        "M4_joint_f1": m4_f1,
        "M5_empty_case_accuracy": n_empty_correct / n_empty_gold if n_empty_gold else 0.0,
    }


def evaluate_strict(records: list[dict]) -> dict:
    return score_records(records, parse_strict)


def evaluate_lenient(records: list[dict]) -> dict:
    return score_records(records, parse_lenient)


def score_extraction_only(records: list[dict], parser=parse_strict) -> dict:
    """
    For the phaseB bonus check (Decision D17): this gold data has spans but
    no polarity, so M3/M4/M5 are not defined here. Scores M1 (schema
    validity — unchanged definition) and aspect-only precision/recall/F1
    (M2's span-matching rule, sentiment ignored entirely).

    records: list of {
        "sentence_id": str,
        "gold_aspects": [str, ...],   # plain aspect strings, no polarity
        "raw_output": str,
    }
    """
    n = len(records)
    n_valid = 0
    n_gold = n_pred = tp = 0

    for r in records:
        gold_set = set(_normalize(a) for a in r["gold_aspects"])

        parsed = parser(r["raw_output"])
        if parsed is not None:
            n_valid += 1
            pred_set = set(_normalize(item["aspect"]) for item in parsed.get("aspects", []))
        else:
            pred_set = set()

        n_gold += len(gold_set)
        n_pred += len(pred_set)
        tp += len(gold_set & pred_set)

    p, r_, f1 = _prf(tp, n_pred, n_gold)
    return {
        "n_records": n,
        "M1_json_validity_rate": n_valid / n if n else 0.0,
        "aspect_precision": p,
        "aspect_recall": r_,
        "aspect_f1": f1,
    }
