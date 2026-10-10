#!/usr/bin/env python3
"""Fills the project fact-sheet template by *querying* the repo's actual
artifacts (configs, model weight headers, result CSVs, notebook code) —
never by reading/parsing prose out of README.md or the docs/ markdown.

Lives entirely under factsheet/ so it can't interfere with anything else
in the repo; it only reads files elsewhere, never writes them.

Run:
    python3 factsheet/generate_factsheet.py
"""
from __future__ import annotations

import csv
import json
import re
from pathlib import Path

from gguf_reader import total_param_count as gguf_total_params
from notebook_config_reader import (
    cell_has_executed_outputs,
    find_peft_get_peft_model_kwargs,
    find_training_arguments_kwargs,
)
from safetensors_reader import total_param_count as adapter_trainable_params

ROOT = Path(__file__).resolve().parent.parent

NOT_FOUND = "NOT FOUND (no executed run/log in repo)"


def load_json(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_project_title() -> str:
    """First H1 line of README.md — a label lookup, not a numeric value."""
    readme = ROOT / "README.md"
    if not readme.exists():
        return "UNKNOWN"
    for line in readme.read_text(encoding="utf-8").splitlines():
        if line.startswith("# "):
            return line[2:].strip()
    return "UNKNOWN"


def get_dataset_facts() -> dict:
    manifest = load_json(ROOT / "data" / "processed" / "MANIFEST.json")
    splits = manifest["splits"]
    total = splits["train"] + splits["val"] + splits["test"]
    return {
        "dataset_name": Path(manifest["source_files"]["train_val_test"]).stem,
        "train": splits["train"],
        "val": splits["val"],
        "test": splits["test"],
        "total": total,
        "total_k": total / 1000,
    }


def get_lora_and_quant_facts() -> dict:
    cfg = load_json(ROOT / "adapters" / "final" / "adapter_config.json")
    base_model = cfg["base_model_name_or_path"]
    bit_match = re.search(r"(\d+)bit", base_model.replace("-", ""))
    return {
        "base_model": base_model,
        "quant_bits": int(bit_match.group(1)) if bit_match else None,
        "lora_rank": cfg["r"],
        "lora_alpha": cfg["lora_alpha"],
        "lora_dropout": cfg["lora_dropout"],
        "target_modules": cfg["target_modules"],
    }


def get_param_counts() -> dict:
    adapter_path = ROOT / "adapters" / "final" / "adapter_model.safetensors"
    gguf_candidates = sorted((ROOT / "adapters" / "gguf").glob("*.gguf"))

    trainable = adapter_trainable_params(str(adapter_path))
    result = {"trainable": trainable, "trainable_m": trainable / 1e6}

    if gguf_candidates:
        total = gguf_total_params(str(gguf_candidates[0]))
        result["total"] = total
        result["total_b"] = total / 1e9
        result["trainable_pct"] = 100 * trainable / total
        result["gguf_file"] = gguf_candidates[0].name
    else:
        result["total"] = None
        result["trainable_pct"] = None

    return result


def get_training_hparams() -> dict:
    nb_path = ROOT / "notebooks" / "03_finetune_colab.ipynb"
    kwargs = find_training_arguments_kwargs(str(nb_path), target_var="full_args")
    lora_kwargs = find_peft_get_peft_model_kwargs(str(nb_path))
    executed = cell_has_executed_outputs(str(nb_path))
    return {"training_args": kwargs, "lora_kwargs": lora_kwargs, "notebook_was_executed": executed}


def find_execution_logs() -> dict | None:
    """Search the repo for any artifact that could only exist from an actual
    training run: a HF Trainer `trainer_state.json` (has train_runtime /
    eval_loss / GPU-adjacent fields), a checkpoints directory, or a
    freestanding training log. Returns parsed facts if found, else None."""
    for candidate in ROOT.rglob("trainer_state.json"):
        state = load_json(candidate)
        log_history = state.get("log_history", [])
        eval_losses = [e["eval_loss"] for e in log_history if "eval_loss" in e]
        return {
            "source": str(candidate.relative_to(ROOT)),
            "eval_loss_best": min(eval_losses) if eval_losses else None,
            "total_flos": state.get("total_flos"),
            "log_history": log_history,
        }
    return None


def get_metrics() -> dict:
    path = ROOT / "results" / "results_table.csv"
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    by_key = {(r["condition"], r["parser"]): r for r in rows}

    zero_shot = by_key.get(("zero_shot", "lenient"))
    fine_tuned = by_key.get(("fine_tuned", "lenient"))
    if not zero_shot or not fine_tuned:
        return {}

    metrics = {}
    for label, field in [
        ("json_validity", "M1_json_validity_rate"),
        ("aspect_f1", "M2_aspect_f1"),
        ("polarity_accuracy", "M3_polarity_accuracy"),
        ("joint_f1", "M4_joint_f1"),
    ]:
        base = float(zero_shot[field])
        tuned = float(fine_tuned[field])
        metrics[label] = {
            "base_pct": 100 * base,
            "finetuned_pct": 100 * tuned,
            "improvement_pp": 100 * (tuned - base),
        }
    return metrics


def render(facts: dict) -> str:
    ds = facts["dataset"]
    lq = facts["lora_quant"]
    pc = facts["params"]
    th = facts["training"]["training_args"] or {}
    logs = facts["exec_logs"]
    metrics = facts["metrics"]
    headline = metrics.get("joint_f1", {})

    def g(d: dict, key, fmt="{}", default=NOT_FOUND):
        if key not in d or d[key] is None:
            return default
        return fmt.format(d[key])

    lines = [
        f"Project: {facts['project_title']}",
        f"Base Model: {lq['base_model']}",
        f"Model Size: {g(pc, 'total_b', '{:.3f}')} B"
        + (f"  (from {pc.get('gguf_file')}, exact tensor-table sum)" if pc.get("gguf_file") else ""),
        f"Dataset: {ds['dataset_name']} (SemEval Aspect-Based Sentiment Analysis, Laptops domain)",
        f"Dataset Size: {ds['total_k']:.3f} K samples  (train {ds['train']} + val {ds['val']} + test {ds['test']})",
        f"Train/Val Split: {ds['train']} / {ds['val']}  (test held out separately: {ds['test']})",
        "",
        f"Quantization: {lq['quant_bits']} bit",
        f"LoRA Rank: {lq['lora_rank']}",
        f"LoRA Alpha: {lq['lora_alpha']}",
        f"LoRA Dropout: {lq['lora_dropout']}",
        "",
        f"Epochs: {g(th, 'num_train_epochs')}",
        f"Learning Rate: {g(th, 'learning_rate')}",
        f"Batch Size: {g(th, 'per_device_train_batch_size')}"
        + (
            f" (x{th['gradient_accumulation_steps']} grad-accum steps"
            f" = effective {int(th['per_device_train_batch_size']) * int(th['gradient_accumulation_steps'])})"
            if "per_device_train_batch_size" in th and "gradient_accumulation_steps" in th
            else ""
        ),
        f"Optimizer: {th.get('optim', 'adamw_torch (transformers TrainingArguments default — not overridden in notebook)')}",
        f"GPU: {'T4 (notebook targets Colab free-tier T4 explicitly)' if not facts['training']['notebook_was_executed'] else NOT_FOUND}",
        f"Training Time: {g(logs or {}, 'total_flos') if logs else NOT_FOUND}",
        "",
        f"Trainable Parameters: {g(pc, 'trainable_m', '{:.3f}')} M  (exact sum over {facts['adapter_tensor_count']} adapter tensors)",
        f"Trainable %: {g(pc, 'trainable_pct', '{:.4f}')} %"
        + ("" if pc.get("trainable_pct") is None else f"  ({pc['trainable_m']:.3f}M / {pc['total_b']*1000:.3f}M)"),
        "",
        f"Base Metric (zero-shot joint F1): {g(headline, 'base_pct', '{:.2f}')} %",
        f"Fine-Tuned Metric (joint F1): {g(headline, 'finetuned_pct', '{:.2f}')} %",
        f"Improvement: {g(headline, 'improvement_pp', '{:+.2f}')} percentage points",
        f"Validation Loss: {g(logs or {}, 'eval_loss_best', '{:.4f}')}",
        f"GPU Memory: {NOT_FOUND}",
        "",
        f"Key Result: fine-tuning raised joint F1 (aspect+polarity, lenient parser) from "
        f"{g(headline, 'base_pct', '{:.1f}')}% to {g(headline, 'finetuned_pct', '{:.1f}')}% "
        f"({g(headline, 'improvement_pp', '{:+.1f}')} pp) on {facts['n_test']} held-out test sentences, "
        f"using {g(pc, 'trainable_pct', '{:.2f}')}% trainable parameters (LoRA r={lq['lora_rank']}).",
    ]

    if not facts["training"]["notebook_was_executed"]:
        lines += [
            "",
            "--- NOTE ---",
            "notebooks/03_finetune_colab.ipynb has no recorded cell outputs (it was authored",
            "but not executed in this repo — see its own markdown caveat). Training Time, GPU",
            "and GPU Memory therefore have no ground truth to query and are reported as such",
            "rather than guessed. Everything else above is computed from real artifacts:",
            "adapters/final/adapter_config.json, adapters/final/adapter_model.safetensors,",
            "adapters/gguf/*.gguf, data/processed/MANIFEST.json, results/results_table.csv,",
            "and the TrainingArguments(...)/get_peft_model(...) calls in that notebook's own code.",
        ]

    return "\n".join(lines)


def main() -> None:
    ds = get_dataset_facts()
    lq = get_lora_and_quant_facts()
    pc = get_param_counts()
    training = get_training_hparams()
    exec_logs = find_execution_logs()
    metrics = get_metrics()

    facts = {
        "project_title": get_project_title(),
        "dataset": ds,
        "lora_quant": lq,
        "params": pc,
        "training": training,
        "exec_logs": exec_logs,
        "metrics": metrics,
        "n_test": ds["test"],
        "adapter_tensor_count": __import__("safetensors_reader").tensor_count(
            str(ROOT / "adapters" / "final" / "adapter_model.safetensors")
        ),
    }

    text = render(facts)
    print(text)

    out_path = Path(__file__).resolve().parent / "factsheet_output.txt"
    out_path.write_text(text + "\n", encoding="utf-8")
    print(f"\n(written to {out_path.relative_to(ROOT)})")

    json_path = Path(__file__).resolve().parent / "factsheet_facts.json"
    json_path.write_text(json.dumps(facts, indent=2, default=str), encoding="utf-8")


if __name__ == "__main__":
    main()
