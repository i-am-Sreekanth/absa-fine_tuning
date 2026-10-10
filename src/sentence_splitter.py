"""
Shared sentence-splitting heuristic. Same logic validated in
notebooks/00_feasibility_and_eda.ipynb's feasibility check (§16.1 step 3),
pulled out here so 06_application.ipynb imports one implementation instead
of maintaining a second copy.

Heuristic, not a real sentence tokenizer (e.g. nltk/spaCy) — sufficient for
this project's purposes (estimating volume, splitting reviews into units
short enough for the prompt template), not linguistically rigorous.
"""

from __future__ import annotations

import re


def split_sentences(text: str) -> list[str]:
    text = re.sub(r"\s+", " ", text).strip()
    if not text:
        return []
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [p.strip() for p in parts if p.strip()]
