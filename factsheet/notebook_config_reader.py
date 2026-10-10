"""Extracts real training-hyperparameter values by parsing the *code* cells
of the fine-tuning notebook with `ast` — not by reading prose/markdown.

We deliberately do not hardcode numbers here: every value returned traces
back to a specific `keyword=value` argument in the notebook's own
`TrainingArguments(...)` call, found by walking the AST of the cell source.
If that call (or an argument on it) isn't there, the field comes back
missing rather than guessed.
"""
from __future__ import annotations

import ast
import json


def _literal_or_source(node: ast.AST) -> object:
    """Try to resolve a constant; fall back to the exact source expression
    (e.g. `not torch.cuda.is_bf16_supported()`) when it isn't a literal."""
    try:
        return ast.literal_eval(node)
    except (ValueError, TypeError):
        return ast.unparse(node)


def find_training_arguments_kwargs(
    notebook_path: str, target_var: str = "full_args", call_name: str = "TrainingArguments"
) -> dict | None:
    """Scan every code cell for `<target_var> = <call_name>(...)` and return
    its keyword arguments as a dict. Returns None if not found."""
    with open(notebook_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        try:
            tree = ast.parse(source)
        except SyntaxError:
            continue

        for node in ast.walk(tree):
            if not isinstance(node, ast.Assign):
                continue
            if not any(isinstance(t, ast.Name) and t.id == target_var for t in node.targets):
                continue
            call = node.value
            if not (isinstance(call, ast.Call) and getattr(call.func, "id", None) == call_name):
                continue
            return {kw.arg: _literal_or_source(kw.value) for kw in call.keywords if kw.arg}

    return None


def find_peft_get_peft_model_kwargs(notebook_path: str) -> dict | None:
    """Scan for `model = FastLanguageModel.get_peft_model(model, r=..., ...)`
    and return its keyword arguments (LoRA rank/alpha/dropout/targets)."""
    with open(notebook_path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    for cell in nb.get("cells", []):
        if cell.get("cell_type") != "code":
            continue
        source = "".join(cell.get("source", []))
        try:
            tree = ast.parse(source)
        except SyntaxError:
            continue

        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            if isinstance(func, ast.Attribute) and func.attr == "get_peft_model":
                return {kw.arg: _literal_or_source(kw.value) for kw in node.keywords if kw.arg}

    return None


def cell_has_executed_outputs(notebook_path: str) -> bool:
    """True if ANY code cell in the notebook has recorded outputs — i.e. the
    notebook was actually run, not just authored. Used to decide whether
    execution-only facts (wall-clock training time, eval loss, GPU memory)
    can exist at all."""
    with open(notebook_path, "r", encoding="utf-8") as f:
        nb = json.load(f)
    return any(cell.get("outputs") for cell in nb.get("cells", []) if cell.get("cell_type") == "code")
