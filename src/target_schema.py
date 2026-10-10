"""
JSON-target construction for the aspect/polarity extraction task. Pure
functions, no I/O — kept separate so evaluation code (Stage 4/5) imports the
exact same schema definition used to build training targets here, instead
of re-deriving it and risking the two disagreeing on what "correct" means.
 
Schema matches src/model_config.py's PROMPT_TEMPLATE exactly:
{"aspects": [{"aspect": "...", "sentiment": "positive|negative|neutral"}], "overall": "..."}
"""

from __future__ import annotations

ALLOWED_POLARITIES = {"positive", "negative", "neutral"}


def derive_overall(polarities: list[str]) -> str:
    """§9.2 Decision 5: derived deterministically, not gold, not evaluated
    directly. all-positive (optionally + neutral) -> positive;
    all-negative (optionally + neutral) -> negative; both signs present ->
    mixed; empty or all-neutral -> neutral.
    """
    if not polarities:
        return "neutral"
    unique = set(polarities)
    if unique <= {"neutral"}:
        return "neutral"
    has_positive = "positive" in unique
    has_negative = "negative" in unique
    if has_positive and has_negative:
        return "mixed"
    if has_positive:
        return "positive"
    if has_negative:
        return "negative"
    return "neutral"


def build_target(aspects: list[tuple[str, str]]) -> dict:
    """aspects: list of (aspect_term, polarity) tuples, already verbatim
    substrings of the source sentence (§9.2 Decision 3 — never normalised).
    """
    return {
        "aspects": [{"aspect": a, "sentiment": p} for a, p in aspects],
        "overall": derive_overall([p for _, p in aspects]),
    }
