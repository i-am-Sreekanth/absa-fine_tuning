# Aspect-Level Customer Review Intelligence via Fine-Tuned Small Language Models

Fine-tunes a small open-weight instruction-tuned LLM to read a customer
product review and emit a structured, machine-readable breakdown of every
product aspect mentioned and the sentiment expressed toward each — full
project reference in [`docs/Customer_Review_Intelligence_Review0.md`](docs/Customer_Review_Intelligence_Review0.md).

## Status

_Last checked against actual repo contents: 2026-09-22._

Every pipeline stage through the application layer is done, including a
real before/after fine-tuning comparison on the actual business data (not
just the benchmark). What's left is the full-corpus run and the write-up.

- [x] Stage 0/1 — Feasibility & EDA (`notebooks/00_feasibility_and_eda.ipynb`)
- [x] Data-sourcing decision finalized (see `docs/…md`, Appendix C) — train/val/test
      carved from `Laptop_Train_v2.xml`; the official SemEval test file
      lacks gold polarity in this mirror, and `laptops-trial.xml` turned out to
      be a subsample of train, not independent data.
- [x] Model selected: `qwen2.5:1.5b-instruct`, pulled locally via Ollama and
      confirmed reachable (`ollama list`, plus a live smoke-test call using
      the project's exact prompt template). Centralized in `src/model_config.py`
      so switching models is a one-line change, not a find-and-replace.
- [x] Stage 2 — Data prep / format conversion (`notebooks/01_data_prep.ipynb`) —
      3,002 non-conflict sentences → fixed-seed (42) 80/10/10 split →
      `data/processed/{train,val,test}.jsonl` (2,402 / 300 / 300), all §9.4
      sanity checks passed, plus `phaseb_extraction_check.jsonl` (800 sentences,
      extraction-only bonus set, Decision D17).
- [x] Evaluation harness (`src/eval_metrics.py`) — M1–M5 + strict/lenient
      parsing, unit-tested against 25 hand-constructed cases
      (`tests/test_eval_metrics.py`, all passing) before touching any real
      model output, per the doc's own Phase 2 exit criterion (§18).
- [x] Stage 3 — Zero-shot baseline (`notebooks/02_baseline.ipynb`) — 300/300
      generations, 0 failed calls. Strict M1=12.7%/lenient M1=88.3% (most
      "failures" were code-fence wrapping, not fundamental invalidity);
      lenient M5=14.8% (the base model hallucinates an aspect onto ~85% of
      aspect-free sentences — strict M5 looked like 91.4% but that was a
      parse-failure artifact, not real restraint). Full numbers in
      `results/results_table.csv`, raw generations in
      `outputs/raw_generations/zero_shot/`.
- [x] Stage 4 — QLoRA fine-tuning. Run by the user on Colab's free T4 GPU
      via `notebooks/03_finetune_colab.ipynb` (local-file-upload flow, no
      Drive). Adapter confirmed real in `adapters/final/` (r=16, alpha=16,
      dropout=0, all 7 attention+MLP target modules — matches §13.1
      exactly). 1,100 fine-tuned generations captured:
      `outputs/raw_generations/finetuned/{test_generations,phaseb_generations}.jsonl`
      (300 + 800 records). Not executed by me — I have no GPU in this
      session; this is the one stage verified only by inspecting the
      artifacts it produced, not by running it myself.
- [x] Stage 5 — Evaluation (`notebooks/04_evaluate.ipynb`) — the real
      comparison, same `src/eval_metrics.py` for both conditions:

      | Metric (lenient) | Zero-Shot | Fine-Tuned |
      |---|---|---|
      | M1 JSON validity | 88.3% | **100%** (0/300 parse failures, independently re-verified) |
      | M2 aspect F1 | 0.304 | **0.449** |
      | M3 polarity accuracy | 77.9% | 78.9% (flat — see below) |
      | M4 joint F1 | 0.237 | **0.354** |
      | M5 empty-case accuracy | 14.8% | **100%** |

      M3 barely moving while M1/M5 jump dramatically is exactly H2's
      prediction, not a weak result: the base model already understood
      sentiment reasonably well zero-shot — fine-tuning's real work was
      format compliance and restraint. Bonus phaseB check (800 genuinely
      unseen sentences, extraction-only, no polarity): aspect F1 = 0.483,
      comparable to the in-domain test set. 50-sample error analysis
      (`results/error_analysis.csv`, first-pass heuristic — needs an actual
      read): dominant issue is `missed_aspect` (23/50), not hallucination —
      consistent with the model leaning conservative rather than
      over-extracting.
- [x] Fine-tuned model served locally via Ollama (GGUF export). Path:
      `notebooks/05_export_gguf_colab.ipynb` (Colab, T4 GPU) merges the
      adapter and exports Q4_K_M GGUF — hit a real bug on the first run
      (`peft.PeftModel.from_pretrained()` produces a model Unsloth's
      `save_pretrained_gguf` doesn't recognize as PEFT-wrapped, so it skips
      the merge and fails serializing the still-quantized base directly —
      `NotImplementedError`), fixed by loading via
      `FastLanguageModel.from_pretrained(model_name=<adapter_dir>)` instead
      (Unsloth's own adapter-aware path). Re-run clean, GGUF downloaded to
      `adapters/gguf/Qwen2.5-1.5B-Instruct.Q4_K_M.gguf` (986MB — matches the
      base model's own Q4_K_M size almost exactly, as a correctly-merged
      same-quantization export should).
      `ollama create qwen2.5-1.5b-absa-finetuned -f adapters/gguf/Modelfile`
      run and verified: registered in `ollama list`, and cross-checked
      against Colab's own captured generations for two known test sentences
      via `mcfg.generate(text, model=mcfg.FINETUNED_MODEL)` — one matched
      byte-for-byte, the other was close but not identical (dropped one weak
      aspect, one neutral/positive wobble), which is expected quantization
      drift versus the pre-GGUF numbers already in `results_table.csv`, not
      a bug. Modelfile uses an explicit passthrough `TEMPLATE` since the
      model trained on raw text, never Qwen's chat template — without it,
      Ollama would silently wrap every prompt in a format never seen in
      training.
- [x] Stage 5 chart — added after the fact: the M1–M5 comparison above had
      only ever existed as text/CSV. `figures/benchmark_zero_shot_vs_finetuned.png`
      is the same data, actually plotted.
- [x] Stage 6 — Application layer (`notebooks/06_application.ipynb`) — ran
      on a 500-review sample (fixed seed 42; full corpus deferred, see
      below), 1,113 sentences, fine-tuned model only at first:
      - Real-world JSON validity held at 100% (both parsers) on messy real
        review text, not just the curated benchmark.
      - Star rating vs. aspect sentiment (Chart 3) is genuinely mixed at
        3★ — 47% negative, 40% positive aspects, not a uniform "meh" block.
        Real evidence, on real customer data, for the project's opening
        claim (§2.1).
      - Top aspects show unmerged variants (`"sound quality"` vs `"sound"`,
        `"battery life"` vs `"battery backup"`) — light grouping is a fair
        next refinement, now data-justified rather than guessed upfront.
      - **Gap found and closed:** every chart initially used only the
        fine-tuned model — no before/after evidence on the application data
        itself. Added Sections 13-14: ran the *zero-shot* base model over
        the identical 1,113 sentences, same settings, and charted the
        contrast directly —
        `figures/05_finetuning_effect_on_application_corpus.png`:

        | | Zero-Shot | Fine-Tuned |
        |---|---|---|
        | JSON validity (strict) | 19% | **100%** |
        | Share of outputs claiming 0 aspects | 2% | **65%** |
        | Avg. aspects / sentence | 1.62 | **0.67** |

        Zero-shot essentially never says "no aspect here" (implausible for
        real review text); fine-tuning changed that dramatically. No gold
        labels exist on this corpus, so this is a real, large, *measured
        behavioral difference* — not a claim that 65%-empty is "correct."
        The spot-check sample already found real under-extraction cases on
        this same data, so both things are true: zero-shot over-extracts
        everywhere, fine-tuned likely swings somewhat too far the other way.
      - Hit and fixed a real bug along the way: `src/model_config.py`'s
        `list_local_models()`/membership check didn't handle Ollama's
        implicit `:latest` tagging — added `model_available()` as the
        correct, reusable check.
- [ ] **Full corpus run** — same notebook, `SAMPLE_REVIEWS = None`
      (~12,720 reviews / ~28,400 sentences after cleaning). Not started.
      Estimated ~62 min for the fine-tuned pass (measured 6.35 sentences/sec
      on the sample) + ~157 min for a matching zero-shot pass (measured
      3.02 sentences/sec) if the comparison is extended too — a real
      time commitment, worth confirming before starting.
- [ ] Aspect variant-grouping refinement (optional, data-justified — see
      Stage 6 above).
- [ ] `REPORT.md` / final write-up — not started.

## Directory structure

```
.
├── README.md                    this file
├── docs/
│   ├── Customer_Review_Intelligence_Review0.md       full project reference (living doc — see its Appendix B/C for the decision log)
│   └── Customer_Review_Intelligence_Review0.txt.txt  short-form proposal
├── data/
│   ├── raw/
│   │   ├── semeval/             SemEval-2014 Task 4 Laptops (train/test/trial XML+CSV) — Dataset A, train/eval source
│   │   └── earphones/           Amazon Earphones Reviews (AllProductReviews.csv, ProductInfo.csv) — Dataset B, application/demo only
│   └── processed/                train/val/test.jsonl + phaseb_extraction_check.jsonl + MANIFEST.json (written by 01_data_prep.ipynb)
├── notebooks/
│   ├── 00_feasibility_and_eda.ipynb   Stage 0/1: integrity checks + EDA on both datasets
│   ├── 01_data_prep.ipynb             Stage 2: XML → JSON targets, fixed-seed split, sanity checks
│   ├── 02_baseline.ipynb              Stage 5: zero-shot baseline (executed — see Status above)
│   ├── 03_finetune_colab.ipynb        Stage 6: QLoRA fine-tune — run on Colab (T4 GPU), results brought back
│   ├── 04_evaluate.ipynb              Stage 7: zero-shot vs. fine-tuned comparison + chart (executed — see Status above)
│   ├── 05_export_gguf_colab.ipynb     Merge adapter + export Q4_K_M GGUF — for Colab (T4 GPU), executed, GGUF brought back
│   └── 06_application.ipynb           Stage 6: application-layer charts + zero-shot-vs-finetuned comparison on real data (executed — see Status above)
├── src/
│   ├── model_config.py          ACTIVE_MODEL, FINETUNED_MODEL, prompt template, Ollama generate() wrapper — single source of truth
│   ├── semeval_parser.py        shared SemEval XML parsing + conflict-sentence drop (D10)
│   ├── target_schema.py         JSON target construction (aspects + derived "overall" field)
│   └── eval_metrics.py          M1–M5 scoring + strict/lenient parsing + extraction-only variant, shared by baseline and post-fine-tune evaluation
├── tests/
│   └── test_eval_metrics.py     33 hand-constructed cases for eval_metrics.py — no model access
├── adapters/
│   ├── final/                    LoRA adapter from the Colab fine-tune (~74MB safetensors + tokenizer files)
│   └── gguf/
│       ├── Qwen2.5-1.5B-Instruct.Q4_K_M.gguf   merged + quantized fine-tuned model (986MB)
│       └── Modelfile                            `ollama create qwen2.5-1.5b-absa-finetuned -f Modelfile` — bare relative FROM path (project dir has a space in it; avoids shebang/parser space issues)
├── outputs/raw_generations/
│   ├── zero_shot/                 raw model output from 02_baseline.ipynb (benchmark), saved verbatim before parsing (§12.3)
│   ├── finetuned/                 raw output from the Colab fine-tune (benchmark), same JSONL shape as zero_shot/
│   ├── application/                fine-tuned model's raw output on the 1,113-sentence earphone sample
│   └── application_zero_shot/      zero-shot model's raw output on the SAME 1,113 sentences — the before/after pair
├── results/
│   ├── results_table.csv                    M1–M5, strict + lenient, both conditions + the phaseB bonus check (benchmark)
│   ├── error_analysis.csv                   50-sample heuristic error categorization (§14.4) — a starting point for manual review, not a substitute
│   ├── application_aspect_sentiment.csv     long-format (review, aspect, sentiment) table from the fine-tuned application run
│   └── application_spot_check_sample.csv    50 examples for manual reading — no gold data on this corpus, so this isn't auto-scored
├── figures/
│   ├── benchmark_zero_shot_vs_finetuned.png            Stage 7's M1–M5 comparison, charted
│   ├── 01_aspect_frequency.png … 04_per_product_comparison.png   the four application-layer charts (§16.2)
│   └── 05_finetuning_effect_on_application_corpus.png  zero-shot vs. fine-tuned, on the actual business data
└── .venv/                       local Python env for running the notebooks (not a deliverable)
```

## Data provenance

| | Dataset A — SemEval-2014 Laptops | Dataset B — Amazon Earphones |
|---|---|---|
| Path | `data/raw/semeval/` | `data/raw/earphones/` |
| Role | Train + evaluate (has gold labels) | Application/demo only (star ratings only, **no aspect labels**) |
| Source | kaggle.com/datasets/charitarth/semeval-2014-task-4-aspectbasedsentimentanalysis | kaggle.com/datasets/shitalkat/amazonearphonesreviews |
| File dates on disk | 2022-01-05 (dataset version date from file metadata) | 2019-10-27 (dataset version date from file metadata) |
| Actual download date | not recorded — gap against the reproducibility checklist (§20 of the reference doc); record this if re-downloading |

Integrity check results (go/no-go gate, §8.2 of the reference doc) and the
full cross-file audit of every SemEval split (train / PhaseA / phaseB /
trial), including why `laptops-trial.xml` was excluded, live in
`notebooks/00_feasibility_and_eda.ipynb` and `docs/…md` Appendix C.

## Running this

> **Note:** the project directory now contains a space (`SEM 7`). `.venv`'s
> `activate` script and console-script shebangs (`jupyter`, `pip`, etc.) were
> generated for the old path and both hardcode it *and* would break on the
> space even if fixed — Unix shebangs don't reliably support spaces in the
> interpreter path. Use module invocation (`python3 -m ...`) instead of
> `source activate` + bare commands until the venv is regenerated:

```bash
cd "notebooks"
../.venv/bin/python3 -m jupyter notebook 00_feasibility_and_eda.ipynb
# or, headless:
../.venv/bin/python3 -m nbconvert --to notebook --execute --inplace 00_feasibility_and_eda.ipynb
```

If regenerating `.venv` from scratch (recommended — fixes the hardcoded old
path; module invocation is still the safer habit afterward given the space):

```bash
python3 -m venv .venv
./.venv/bin/python3 -m pip install pandas numpy matplotlib jupyter nbformat nbconvert ipykernel
```

Ollama, if used for baseline prompting or later inference:

```bash
ollama list                       # confirm qwen2.5:1.5b-instruct is present
ollama pull qwen2.5:1.5b-instruct # if not
```

## Reproducibility notes so far

- Random seed: `42`, fixed in `notebooks/01_data_prep.ipynb`, used for both
  the train/val/test shuffle and the 10-example manual-read sample. Recorded
  in `data/processed/MANIFEST.json`.
- Split sizes: train 2,402 / val 300 / test 300 (from 3,002 non-conflict
  sentences), plus 800 sentences held separately as the extraction-only
  bonus check (`phaseb_extraction_check.jsonl`).
- Model: `qwen2.5:1.5b-instruct`, Q4_K_M quantization, pulled via Ollama
  (`ollama list` → digest `65ec06548149`).
- Hardware: RTX 3050 Laptop GPU (4GB VRAM), Ryzen 7 7445HS, 14GB system RAM —
  see `docs/…md` for how this constrained model selection.
