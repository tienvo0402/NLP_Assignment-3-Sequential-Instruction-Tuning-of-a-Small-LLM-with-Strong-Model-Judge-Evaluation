import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from peft import PeftModel, prepare_model_for_kbit_training

# ======================
# CONFIG
# ======================
BASE_MODEL = "microsoft/Phi-3.5-mini-instruct"
STAGE1_PATH = "outputs/stage1_debug"   # MUST be local folder
DATA_PATH = "stage2_data.json"

OUTPUT_DIR = "outputs/stage2_adapter"

device = "cuda" if torch.cuda.is_available() else "cpu"

# ======================
# TOKENIZER
# ======================
tokenizer = AutoTokenizer.from_pretrained(
    BASE_MODEL,
    trust_remote_code=True
)
tokenizer.pad_token = tokenizer.eos_token

# ======================
# LOAD BASE MODEL (QLoRA)
# ======================
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True
)

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    device_map="auto",
    quantization_config=bnb_config,
    trust_remote_code=True,
    attn_implementation="eager"
)

base_model = prepare_model_for_kbit_training(base_model)

# ======================
# LOAD STAGE 1 ADAPTER (LOCAL FIX)
# ======================
model = PeftModel.from_pretrained(
    base_model,
    STAGE1_PATH,
    is_trainable=True,
    local_files_only=True   # 
)

print("Stage 1 adapter loaded successfully!")

# ======================
# LOAD DATASET
# ======================
dataset = load_dataset("json", data_files=DATA_PATH)

# ======================
# FORMAT DATA
# ======================
def format_example(example):
    instruction = example["instruction"]
    inp = example.get("input", "")
    output = example["output"]

    text = f"""### Instruction:
{instruction}

### Input:
{inp}

### Response:
{output}"""

    return tokenizer(
        text,
        truncation=True,
        max_length=512,
        padding="max_length"
    )

dataset = dataset.map(format_example, remove_columns=dataset["train"].column_names)

# ======================
# DATA COLLATOR
# ======================
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

# ======================
# TRAINING ARGS
# ======================
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    num_train_epochs=1,
    learning_rate=1e-5,
    logging_steps=10,
    save_steps=200,
    save_total_limit=2,
    fp16=True,
    report_to="none"
)

# ======================
# TRAINER
# ======================
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    data_collator=data_collator
)

# ======================
# TRAIN
# ======================
trainer.train()

# ======================
# SAVE FINAL MODEL
# ======================
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("STAGE 2 TRAINING COMPLETE")