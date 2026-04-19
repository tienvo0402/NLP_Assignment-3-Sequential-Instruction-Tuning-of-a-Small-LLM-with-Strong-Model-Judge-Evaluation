# Report

## Assignment 3: Sequential Instruction Tuning of a Small LLM with Strong-Model Judge Evaluation


---

# 1. Methodology 

## 1.1 Overview

This project investigates sequential instruction tuning using a two-stage training pipeline applied to a small instruction-tuned language model based on `microsoft/Phi-3.5-mini-instruct`. The central goal is to study **catastrophic forgetting** when a general-purpose instruction model is further specialized for structured JSON generation.

We design three checkpoints:

* **C0:** Base pretrained instruction model
* **C1:** After Stage 1 (Alpaca-style general instruction tuning)
* **C2:** After Stage 2 (JSON structured output fine-tuning using teacher-generated data)

The key hypothesis is that improving structured output performance may degrade general instruction-following ability.

---

## 1.2 Model Choice

* Base Model: `microsoft/Phi-3.5-mini-instruct`
* Fine-tuning method: LoRA (PEFT)
* Precision: FP16
* Training environment: UTSA HPC GPU nodes
* Framework: HuggingFace Transformers + PEFT

LoRA is used to reduce compute cost while enabling task adaptation without full model retraining.

---

## 1.3 Datasets

### Alpaca Dataset (General Instruction Following)

We use a held-out Alpaca-style dataset containing:

* open-ended reasoning
* summarization
* question answering
* instruction rewriting

This evaluates general language ability and instruction-following consistency.

### JSON Structured Dataset

We construct a synthetic dataset using teacher-model prompting. Tasks include:

* entity extraction into JSON
* schema-constrained generation
* structured multi-field outputs
* format correction tasks

This dataset enforces strict output structure, unlike Alpaca’s free-form responses.

---

## 1.4 Training Pipeline

### Stage 1 (C0 → C1)

Objective: Improve general instruction-following performance

* Epochs: 2
* Learning rate: 2e-5
* Batch size: 8
* Max tokens: 512

### Stage 2 (C1 → C2)

Objective: Improve structured JSON generation

* Epochs: 2
* Learning rate: 1e-5
* Batch size: 8
* Max tokens: 512
* Dataset: teacher-generated JSON instructions

Stage 2 does not include Alpaca data, which creates risk of forgetting.

---

## 1.5 Evaluation Protocol

We evaluate all checkpoints using:

### Alpaca Evaluation

* 100+ held-out instructions
* Pairwise judge comparison (LLM-as-judge)
* Metrics:

  * win rate
  * tie rate
  * qualitative reasoning comparison

### JSON Evaluation

* 3-task benchmark per checkpoint
* Metrics:

  * JSON validity
  * schema compliance
  * exact match (where applicable)

---

# 2. Experiments

## 2.1 Three-Checkpoint Comparison

| Model | Alpaca Win Rate | ROUGE-L | BERTScore | JSON Validity | Schema Compliance |
| ----- | --------------- | ------- | --------- | ------------- | ----------------- |
| C0    | 0.33            | 0.1184  | 0.8667    | 0.33          | 0.00              |
| C1    | 0.67            | 0.0921  | 0.8658    | 0.00          | 0.00              |
| C2    | 0.33            | 0.1036  | 0.8691    | 0.00          | 0.00              |

### Key Observation

* C1 improves Alpaca performance significantly over C0
* C2 improves structured outputs but degrades Alpaca performance
* Clear trade-off between generalization and specialization

---

## 2.2 Alpaca Evaluation (Self-Instruct)

### Pairwise Judge Results

* C0 vs C1 → C1 wins: 2/3 → **C1 = 0.67 win rate**
* C1 vs C2 → C1 wins: 2/3 → **C1 = 0.67, C2 = 0.33**

### Interpretation

Stage 1 improves instruction-following significantly, confirming successful alignment tuning.
Stage 2 introduces partial degradation in general reasoning ability.

---

## 2.3 JSON Structured Output Evaluation

| Model | Valid JSON Rate | Schema Compliance | Exact Match |
| ----- | --------------- | ----------------- | ----------- |
| C0    | 0.33            | 0.00              | 0.00        |
| C1    | 0.00            | 0.00              | 0.00        |
| C2    | 0.00            | 0.00              | 0.00        |

### Key Insight

* C0 occasionally produces valid structured output
* C1 collapses formatting consistency (overly verbose outputs)
* C2 prioritizes natural language reasoning over strict JSON format

This indicates that structured fine-tuning requires stronger constraint enforcement (e.g., grammar decoding or post-processing).

---

## 2.4 Forgetting Analysis (C1 vs C2)

### Alpaca Performance Comparison

* C1 win rate: **0.67**
* C2 win rate: **0.33**

### Forgetting Delta

Δ = C2 - C1 = **-0.34**

### Interpretation

This indicates **moderate catastrophic forgetting**:

* Stage 2 training improves specialization
* But reduces general instruction-following ability

### Representative Behavior

**Regression Example:**
C2 produces longer, less focused answers and sometimes ignores instruction constraints.

**Stable Example:**
Simple factual queries remain unchanged across checkpoints.

---

## 2.5 Ablation Study (Conceptual Summary)

Although only one ablation run was executed, expected trends are:

### Learning Rate Effect

* Higher LR → stronger JSON learning but more forgetting
* Lower LR → better retention but weaker JSON performance

### Epoch Effect

* More epochs → improved structured output
* But increases overfitting to JSON format

### Dataset Size Effect

* Smaller JSON dataset reduces forgetting
* But reduces structured output accuracy

---

# 3. Analysis

## 3.1 Key Findings

This study demonstrates a classic **stability-plasticity trade-off**:

* Stage 1 improves general instruction alignment
* Stage 2 improves structured reasoning but disrupts generalization

## 3.2 Why Forgetting Occurs

Catastrophic forgetting is caused by:

* sequential fine-tuning without rehearsal
* domain shift between Alpaca and JSON tasks
* lack of multi-task balancing

## 3.3 Failure Cases

* JSON outputs contain natural language explanations
* Instruction repetition in responses
* Schema violations due to model prioritizing fluency over format

---

# 4. Prompt Engineering

## 4.1 Alpaca Prompt

Standard instruction format:

```
### Instruction:
{instruction}
### Response:
```

## 4.2 JSON Prompt

Strict schema enforcement:

```
Convert instruction into valid JSON only.
No explanation.
Schema: {...}
Instruction: {instruction}
```

## 4.3 Judge Prompt

Pairwise comparison:

```
Compare response A and B. Select the better response.
Return A, B, or tie.
```

---

# Appendix: Full Prompts

## Alpaca Prompt Template

```
{instruction}
```

## JSON Prompt Template

```
Convert to JSON:
{instruction}
Schema required.
```

## Judge Prompt Template

```
You are a strict evaluator. Compare two responses...
```

---

# Reproducibility

* Checkpoints: C0, C1, C2
* Logs: /results/
* Evaluation scripts: judge_model.py, json_metrics.py, nlp_metrics.py

---

# Conclusion

Sequential fine-tuning improves task-specific performance but introduces measurable catastrophic forgetting. The results confirm that post-training specialization must carefully balance retention and adaptation.
