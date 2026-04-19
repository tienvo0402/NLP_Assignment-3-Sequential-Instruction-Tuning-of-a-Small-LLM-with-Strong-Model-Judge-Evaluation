import os
import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# ======================
# PATHS
# ======================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

train_path = os.path.join(BASE_DIR, "data", "alpaca_train.json")
eval_path = os.path.join(BASE_DIR, "data", "alpaca_eval.json")

MODEL_NAME = "microsoft/Phi-3.5-mini-instruct"

# ======================
# LOAD DATASET (FULL)
# ======================
dataset = load_dataset(
    "json",
    data_files={"train": train_path, "test": eval_path}
)

# ======================
# TOKENIZER
# ======================
tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True
)

tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# ======================
# TOKENIZATION
# ======================
def tokenize(example):
    instruction = example["instruction"]
    input_text = example["input"] if example["input"] else ""
    output_text = example["output"]

    prompt = f"### Instruction:\n{instruction}\n\n### Input:\n{input_text}\n\n### Response:\n"
    full_text = prompt + output_text

    tokenized = tokenizer(
        full_text,
        truncation=True,
        max_length=512,
        padding="max_length"
    )

    prompt_tokens = tokenizer(
        prompt,
        truncation=True,
        max_length=512,
        add_special_tokens=False
    )["input_ids"]

    labels = tokenized["input_ids"].copy()

    # Mask prompt part
    prompt_len = min(len(prompt_tokens), len(labels))
    labels[:prompt_len] = [-100] * prompt_len

    tokenized["labels"] = labels
    return tokenized

dataset = dataset.map(tokenize, remove_columns=dataset["train"].column_names)

# ======================
# MODEL (QLoRA)
# ======================
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
    attn_implementation="eager"
)

model = prepare_model_for_kbit_training(model)

# ======================
# LoRA CONFIG
# ======================
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# ======================
# TRAINING ARGS
# ======================
training_args = TrainingArguments(
    output_dir="outputs/stage1",

    per_device_train_batch_size=2,
    gradient_accumulation_steps=2,

    learning_rate=2e-5,
    max_grad_norm=1.0,

    num_train_epochs=1,

    logging_steps=10,
    save_steps=500,
    save_total_limit=2,

    fp16=True,
    report_to="none",

    remove_unused_columns=False
)

# ======================
# TRAINER
# ======================
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"]
)

# ======================
# TRAIN
# ======================
trainer.train()

# ======================
# SAVE
# ======================
model.save_pretrained("outputs/stage1")
tokenizer.save_pretrained("outputs/stage1")

print("STAGE 1 TRAINING COMPLETE")