"""
Hand-constructed cases for src/eval_metrics.py — the doc's own Phase 2 exit
criterion (§18): "Metrics correct on hand-constructed cases," done before
any real model output exists. No Ollama, no network, no model.

Run: python3 tests/test_eval_metrics.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from eval_metrics import parse_lenient, parse_strict, score_extraction_only, score_records

failures = []


def check(name, condition, detail=""):
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {name}" + (f" — {detail}" if detail and not condition else ""))
    if not condition:
        failures.append(name)


# ---------------------------------------------------------------------------
# Parser-level cases
# ---------------------------------------------------------------------------

check(
    "strict parse: well-formed JSON matching schema",
    parse_strict('{"aspects":[{"aspect":"battery","sentiment":"positive"}]}') is not None,
)

check(
    "strict parse: empty aspects list is valid",
    parse_strict('{"aspects":[]}') is not None,
)

check(
    "strict parse REJECTS code-fenced JSON (no cleanup allowed)",
    parse_strict('```json\n{"aspects":[]}\n```') is None,
)

check(
    "lenient parse RECOVERS code-fenced JSON",
    parse_lenient('```json\n{"aspects":[]}\n```') is not None,
)

check(
    "strict/lenient both REJECT schema drift (bare array, not {\"aspects\":[...]})"
    " — json.loads() would accept this, the schema check must not",
    parse_strict('[{"aspect":"battery","sentiment":"positive"}]') is None
    and parse_lenient('[{"aspect":"battery","sentiment":"positive"}]') is None,
)

check(
    "strict/lenient both REJECT wrong-shape schema drift ({\"battery\":\"positive\"})",
    parse_strict('{"battery":"positive"}') is None
    and parse_lenient('{"battery":"positive"}') is None,
)

check(
    "strict/lenient both REJECT an invented polarity label",
    parse_strict('{"aspects":[{"aspect":"battery","sentiment":"very positive"}]}') is None,
)

check(
    "lenient parse strips leading prose (\"Sure! Here's the JSON:\")",
    parse_lenient('Sure! Here is the JSON you requested:\n{"aspects":[]}') is not None,
)

check(
    "lenient parse still returns None on genuinely broken JSON",
    parse_lenient("not json at all, sorry") is None,
)


# ---------------------------------------------------------------------------
# Scoring-level cases — one record at a time, hand-verified expected values
# ---------------------------------------------------------------------------

def rec(sentence_id, gold_aspects, raw_output):
    return {"sentence_id": sentence_id, "gold_aspects": gold_aspects, "raw_output": raw_output}


# Case 1: perfect match on a single multi-aspect sentence.
r1 = [rec(
    "1",
    [{"aspect": "keyboard", "sentiment": "negative"}, {"aspect": "battery life", "sentiment": "positive"}],
    '{"aspects":[{"aspect":"keyboard","sentiment":"negative"},{"aspect":"battery life","sentiment":"positive"}]}',
)]
s1 = score_records(r1, parse_strict)
check("perfect match: M1=1.0", s1["M1_json_validity_rate"] == 1.0)
check("perfect match: M2 F1=1.0", s1["M2_aspect_f1"] == 1.0)
check("perfect match: M3=1.0", s1["M3_polarity_accuracy"] == 1.0)
check("perfect match: M4 F1=1.0", s1["M4_joint_f1"] == 1.0)

# Case 2: zero-aspect gold, model correctly restrains.
r2 = [rec("2", [], '{"aspects":[]}')]
s2 = score_records(r2, parse_strict)
check("correct restraint: M5=1.0", s2["M5_empty_case_accuracy"] == 1.0)

# Case 3: zero-aspect gold, model hallucinates an aspect.
r3 = [rec("3", [], '{"aspects":[{"aspect":"shipping","sentiment":"positive"}]}')]
s3 = score_records(r3, parse_strict)
check("hallucination on empty sentence: M5=0.0", s3["M5_empty_case_accuracy"] == 0.0)
check("hallucination on empty sentence: M2 precision=0.0 (1 wrong prediction, 0 gold)",
      s3["M2_aspect_precision"] == 0.0)

# Case 4: boundary error — gold "battery life", predicted "battery" (no match, exact-match rule).
r4 = [rec("4", [{"aspect": "battery life", "sentiment": "positive"}],
           '{"aspects":[{"aspect":"battery","sentiment":"positive"}]}')]
s4 = score_records(r4, parse_strict)
check("boundary error counts as a full miss: M2 F1=0.0", s4["M2_aspect_f1"] == 0.0)

# Case 5: polarity flip — aspect span correct, sentiment wrong.
r5 = [rec("5", [{"aspect": "screen", "sentiment": "positive"}],
           '{"aspects":[{"aspect":"screen","sentiment":"negative"}]}')]
s5 = score_records(r5, parse_strict)
check("polarity flip: M2 F1=1.0 (span still correct)", s5["M2_aspect_f1"] == 1.0)
check("polarity flip: M3=0.0", s5["M3_polarity_accuracy"] == 0.0)
check("polarity flip: M4 F1=0.0 (joint requires both)", s5["M4_joint_f1"] == 0.0)

# Case 6: unparseable output — must NOT be excluded from the denominator (§14.2).
r6 = [rec("6", [{"aspect": "price", "sentiment": "positive"}], "Sure, the sentiment is positive!")]
s6 = score_records(r6, parse_strict)
check("unparseable output: M1=0.0", s6["M1_json_validity_rate"] == 0.0)
check("unparseable output: M2 recall=0.0 (gold aspect still counted, not skipped)",
      s6["M2_aspect_recall"] == 0.0)

# Case 7: case-insensitive, whitespace-normalised aspect matching (M2's own definition).
r7 = [rec("7", [{"aspect": "Battery Life", "sentiment": "positive"}],
           '{"aspects":[{"aspect":"battery   life","sentiment":"positive"}]}')]
s7 = score_records(r7, parse_strict)
check("case/whitespace-normalised match: M2 F1=1.0", s7["M2_aspect_f1"] == 1.0)

# Case 8: strict vs. lenient on the SAME raw output should differ when it's fenced.
r8 = [rec("8", [{"aspect": "price", "sentiment": "positive"}],
           '```json\n{"aspects":[{"aspect":"price","sentiment":"positive"}]}\n```')]
s8_strict = score_records(r8, parse_strict)
s8_lenient = score_records(r8, parse_lenient)
check("fenced output: strict M1=0.0, lenient M1=1.0 (the whole point of §12.4)",
      s8_strict["M1_json_validity_rate"] == 0.0 and s8_lenient["M1_json_validity_rate"] == 1.0)
check("fenced output: lenient still recovers full Joint F1",
      s8_lenient["M4_joint_f1"] == 1.0)

# Case 9: aggregate over a small mixed batch — sanity check the averaging, not just single records.
# Only r6 is genuinely unparseable; r3 (the hallucination) is well-formed,
# schema-valid JSON with wrong *content* — M1 measures structure, not
# correctness, so it counts as valid here. That's intentional, not a bug:
# it's the exact M1-vs-M2 distinction the doc draws.
batch = r1 + r2 + r3 + r4 + r5 + r6 + r7
s_batch = score_records(batch, parse_strict)
check("batch M1 = 6/7 (only r6 is unparseable; r3's hallucination is still valid JSON)",
      abs(s_batch["M1_json_validity_rate"] - (6 / 7)) < 1e-9,
      detail=f"got {s_batch['M1_json_validity_rate']}")

# ---------------------------------------------------------------------------
# score_extraction_only — the phaseB bonus check (Decision D17): gold has
# spans but no polarity, so only M1 + aspect P/R/F1 are defined.
# ---------------------------------------------------------------------------

def erec(sentence_id, gold_aspects, raw_output):
    return {"sentence_id": sentence_id, "gold_aspects": gold_aspects, "raw_output": raw_output}


# Case 10: perfect extraction match, no polarity involved at all.
e1 = [erec("e1", ["boot time"], '{"aspects":[{"aspect":"boot time","sentiment":"positive"}]}')]
se1 = score_extraction_only(e1, parse_strict)
check("extraction-only perfect match: M1=1.0", se1["M1_json_validity_rate"] == 1.0)
check("extraction-only perfect match: aspect_f1=1.0", se1["aspect_f1"] == 1.0)
check("extraction-only ignores sentiment entirely (result unaffected by predicted polarity)",
      score_extraction_only(
          [erec("e1b", ["boot time"], '{"aspects":[{"aspect":"boot time","sentiment":"negative"}]}')],
          parse_strict,
      )["aspect_f1"] == 1.0)

# Case 11: missed aspect -> recall hit, still M1-valid.
e2 = [erec("e2", ["boot time", "keyboard"], '{"aspects":[{"aspect":"boot time","sentiment":"neutral"}]}')]
se2 = score_extraction_only(e2, parse_strict)
check("extraction-only missed aspect: recall=0.5", abs(se2["aspect_recall"] - 0.5) < 1e-9)
check("extraction-only missed aspect: precision=1.0", se2["aspect_precision"] == 1.0)

# Case 12: unparseable output on extraction-only — same §14.2 rule, not excluded.
e3 = [erec("e3", ["boot time"], "not json")]
se3 = score_extraction_only(e3, parse_strict)
check("extraction-only unparseable: M1=0.0", se3["M1_json_validity_rate"] == 0.0)
check("extraction-only unparseable: recall=0.0 (gold still counted)", se3["aspect_recall"] == 0.0)

# Case 13: lenient recovers a fenced response same as the main scorer does.
e4 = [erec("e4", ["boot time"], '```json\n{"aspects":[{"aspect":"boot time","sentiment":"neutral"}]}\n```')]
check("extraction-only fenced: strict M1=0, lenient M1=1",
      score_extraction_only(e4, parse_strict)["M1_json_validity_rate"] == 0.0
      and score_extraction_only(e4, parse_lenient)["M1_json_validity_rate"] == 1.0)

print()
if failures:
    print(f"{len(failures)} check(s) FAILED: {failures}")
    sys.exit(1)
else:
    print("All checks passed.")
