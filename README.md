# Assignment 3: Sequential Instruction Tuning of a Small LLM with Strong-Model Judge Evaluation


This repository implements a two-stage instruction tuning pipeline using LoRA fine-tuning on a small language model, followed by a comprehensive evaluation of catastrophic forgetting across three checkpoints.

---

# 📌 Project Overview

We study the effect of sequential fine-tuning on:

* **C0:** Base model (microsoft/Phi-3.5-mini-instruct)
* **C1:** After Stage 1 (Alpaca-style instruction tuning)
* **C2:** After Stage 2 (JSON structured output tuning)

Key goal:

> Measure how specialization for structured outputs impacts general instruction-following ability.

---

# Model Details

* Base Model: `microsoft/Phi-3.5-mini-instruct`
* Fine-tuning method: LoRA (PEFT)
* Framework: HuggingFace Transformers
* Precision: FP16
* Training environment: UTSA HPC GPU nodes

---

# Repository Structure

```
scripts/
  eval_all_checkpoints.py
  judge_model.py
  json_metrics.py
  nlp_metrics.py
  forgetting_analysis.py
  build_stage2_data.py
  eval_c0.py
  eval_c1.py
  eval_c2.py
  generate_json_data.py
  prepare_alpaca.py
  train_stage1.py
  train_stage2.py

results/
  C0_alpaca.json
  C1_alpaca.json
  C2_alpaca.json
  C0_json.json
  C1_json.json
  C2_json.json
  nlp_metrics_safe.csv

outputs/
  stage1_debug/
  stage2_adapter/

REPORT.md
README.md
```

---

# Setup Instructions

## 1. Create environment

```bash
conda create -n llm_env python=3.10 -y
conda activate llm_env
```

## 2. Install dependencies

```bash
pip install -r requirements.txt
```

Main packages:

* torch
* transformers
* peft
* evaluate
* rouge-score
* bert-score
* datasets

````
# How to Run Everything

## 1. Run full evaluation (C0, C1, C2)
```bash
python scripts/eval_all_checkpoints.py
````

Outputs saved to:

```
results/C*_alpaca.json
results/C*_json.json
```

---

## 2. Run judge model (pairwise comparison)

```bash
python scripts/judge_model.py
```

Compares:

* C0 vs C1
* C1 vs C2
* JSON variants

Outputs win/tie counts.

---

## 3. JSON structured evaluation

```bash
python scripts/json_metrics.py
```

Computes:

* JSON validity
* schema correctness
* exact match

---

## 4. ROUGE + BERTScore evaluation

```bash
python scripts/nlp_metrics.py
```

Outputs:

* ROUGE-L
* BERTScore

Saved to:

```
results/nlp_metrics_safe.csv
```

---

## 5. Forgetting analysis (C1 vs C2)

```bash
python scripts/forgetting_analysis.py
```

Outputs:

* Alpaca win rate drop
* forgetting delta
* qualitative comparison

---

# Evaluation Summary

We evaluate across three checkpoints:

| Model | Description         |
| ----- | ------------------- |
| C0    | Base model          |
| C1    | After Alpaca tuning |
| C2    | After JSON tuning   |

---

# Metrics

## Alpaca Evaluation

* Pairwise LLM judge comparison
* Win / tie / loss rate

## Automatic Metrics

* ROUGE-L
* BERTScore
* Output length

## JSON Evaluation

* Valid JSON rate
* Schema compliance
* Exact match accuracy

---

# Key Experiment Goal

We study:

> Does improving structured output (JSON) degrade general instruction-following ability?

This is analyzed via:

* C1 → C2 performance drop
* Forgetting delta
* qualitative failure cases

---

# Key Finding (Expected Result)

* C1 improves general instruction following
* C2 improves structured format behavior
* BUT C2 shows **catastrophic forgetting** on Alpaca tasks

---

# Reproducibility

To reproduce all results:

```bash
python scripts/eval_all_checkpoints.py
python scripts/judge_model.py
python scripts/json_metrics.py
python scripts/nlp_metrics.py
python scripts/forgetting_analysis.py
```

---

# Outputs

All results are saved in:

```
/results
```

Includes:

* model outputs
* evaluation scores
* judge comparisons

---

# Report

Full analysis is provided in:

```
REPORT.md
```

Includes:

* methodology
* experiments
* forgetting analysis
* ablation discussion
* prompt engineering

---

# Notes

* Flash-attention warnings are expected on HPC systems
* Some JSON outputs may fail strict schema due to unconstrained decoding
* BERTScore model loads RoBERTa automatically on first run

---


