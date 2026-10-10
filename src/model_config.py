from __future__ import annotations

import json
import urllib.error
import urllib.request

# ---------------------------------------------------------------------------
# Change this one line to switch models everywhere at once.
# ---------------------------------------------------------------------------
ACTIVE_MODEL = "qwen2.5:1.5b-instruct"

# The actual fine-tuned artifact (adapter merged + GGUF-exported, imported
# into Ollama via adapters/gguf/Modelfile — Q4_K_M, same quantization level
# as ACTIVE_MODEL) — separate from ACTIVE_MODEL, which is for base-model
# experimentation. Stage 8 (the application layer) should import and use
# this name specifically, since the whole point of that stage is running
# the *fine-tuned* model over the earphone corpus.
#
# Registered and verified reachable (2026-09-20) via `ollama create` +
# `generate()` against known test sentences. Caveat: outputs are close to
# but not always byte-identical to the raw generations already scored in
# results/results_table.csv — those were captured before GGUF quantization,
# this serves the post-quantization model. Minor drift (e.g. dropping a
# weakly-signalled aspect, a neutral/positive wobble) is expected
# quantization noise, not a pipeline bug.
FINETUNED_MODEL = "qwen2.5-1.5b-absa-finetuned"

KNOWN_MODELS = {
    "qwen2.5:1.5b-instruct": "primary — this is the model the project plans to fine-tune",
    "qwen2.5:3b-instruct":   "same family, one size up; backup fine-tune target",
    "llama3.2:3b-instruct":  "doc's named alternative model",
    "llama3.2:1b-instruct":  "lighter Llama sibling",
    "qwen3:1.7b":            "newer-gen Qwen, closest size match to the primary",
    "phi3.5":                "Microsoft, MIT license, strong instruction-following for size",
    "qwen3:0.6b":            "ultra-light floor/ablation reference",
    "gemma3:1b-it-q4_K_M":   "Google family, cross-architecture diversity check",
    "smollm2:1.7b":          "HF efficiency-focused, weaker benchmarks at this size",
    "qwen3:4b":              "largest that still fits 4GB VRAM comfortably",
}

OLLAMA_HOST = "http://localhost:11434"

DEFAULT_OPTIONS = {"temperature": 0}

PROMPT_TEMPLATE = """### Instruction:
Extract every product aspect mentioned in the following review sentence
and the sentiment expressed towards each. Respond only with JSON in the
schema: {{"aspects":[{{"aspect":"...","sentiment":"positive|negative|neutral"}}]}}
If no product aspect is mentioned, return {{"aspects":[]}}.

### Input:
{sentence}

### Response:
"""


def build_prompt(sentence: str) -> str:
    return PROMPT_TEMPLATE.format(sentence=sentence)


def is_reachable(host: str = OLLAMA_HOST, timeout: float = 2.0) -> bool:
    """Check the Ollama server is up. Hits /api/tags (metadata only) —
    does not load or invoke any model."""
    try:
        urllib.request.urlopen(f"{host}/api/tags", timeout=timeout)
        return True
    except (urllib.error.URLError, OSError):
        return False


def list_local_models(host: str = OLLAMA_HOST, timeout: float = 2.0) -> list[str]:
    """Names of models currently pulled locally, as Ollama reports them —
    fully qualified with a tag (e.g. "qwen2.5:1.5b-instruct",
    "qwen2.5-1.5b-absa-finetuned:latest"). Metadata only, no inference."""
    with urllib.request.urlopen(f"{host}/api/tags", timeout=timeout) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return [m["name"] for m in data.get("models", [])]


def model_available(name: str, host: str = OLLAMA_HOST, timeout: float = 2.0) -> bool:
    """True if `name` is pulled/registered locally. Handles Ollama's
    implicit ":latest" tagging correctly — a bare name like
    "qwen2.5-1.5b-absa-finetuned" (no explicit tag, as `generate()` accepts
    and resolves server-side) won't exact-string-match the fully-qualified
    "qwen2.5-1.5b-absa-finetuned:latest" that list_local_models() returns,
    so a plain `name in list_local_models()` check silently fails for any
    model created/pulled without an explicit tag."""
    local = list_local_models(host, timeout)
    if name in local:
        return True
    if ":" not in name:
        return f"{name}:latest" in local
    return False


def generate(sentence: str, model: str | None = None, host: str = OLLAMA_HOST,
             options: dict | None = None, timeout: float = 60.0) -> dict:
    """
    Run one review sentence through `model` (defaults to ACTIVE_MODEL) via
    Ollama's /api/generate, using the project's fixed prompt template and
    greedy decoding by default. Returns Ollama's raw parsed JSON response;
    the model's own output is in the returned dict's "response" key and
    still needs schema parsing (see §12.4 — strict vs. lenient parsing).
    """
    model = model or ACTIVE_MODEL
    payload = {
        "model": model,
        "prompt": build_prompt(sentence),
        "stream": False,
        "options": {**DEFAULT_OPTIONS, **(options or {})},
    }
    req = urllib.request.Request(
        f"{host}/api/generate",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))
