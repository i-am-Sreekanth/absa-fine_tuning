"""
Shared SemEval-2014 ABSA XML parsing. This is the same logic validated in
notebooks/00_feasibility_and_eda.ipynb, pulled out here so 01_data_prep.ipynb
(and anything else that needs it later) imports one implementation instead
of maintaining a second copy that can drift from what was actually verified.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET

import pandas as pd


def parse_semeval_xml(path) -> pd.DataFrame:
    """Flatten a SemEval-2014 ABSA XML file to one row per (sentence, aspectTerm).
    Sentences with zero aspectTerms produce a single row with aspect fields = None.
    """
    tree = ET.parse(path)
    root = tree.getroot()
    rows = []
    for sent in root.findall("sentence"):
        sid = sent.get("id")
        text_el = sent.find("text")
        text = text_el.text if text_el is not None else ""
        aspect_terms = sent.find("aspectTerms")
        terms = aspect_terms.findall("aspectTerm") if aspect_terms is not None else []
        if not terms:
            rows.append({"sentence_id": sid, "text": text, "aspect": None,
                         "polarity": None, "from": None, "to": None})
        else:
            for t in terms:
                rows.append({
                    "sentence_id": sid,
                    "text": text,
                    "aspect": t.get("term"),
                    "polarity": t.get("polarity"),
                    "from": int(t.get("from")),
                    "to": int(t.get("to")),
                })
    return pd.DataFrame(rows)


def drop_conflict_sentences(df: pd.DataFrame) -> pd.DataFrame:
    """Implements Decision D10: any sentence containing a conflict-polarity
    aspect is removed *whole* (all its rows), not just that one aspect —
    dropping only the aspect would leave a visible attribute in the text
    that the model is trained to silently ignore.
    """
    conflict_ids = set(df.loc[df["polarity"] == "conflict", "sentence_id"])
    return df[~df["sentence_id"].isin(conflict_ids)].reset_index(drop=True)
