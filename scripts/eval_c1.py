import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

# CONFIG
BASE_MODEL = "microsoft/Phi-3.5-mini-instruct"
ADAPTER_PATH = "outputs/stage1"

# TOKENIZER
tokenizer = AutoTokenizer.from_pretrained(
    BASE_MODEL,
    trust_remote_code=True
)

tokenizer.pad_token = tokenizer.eos_token

# LOAD BASE MODEL (QLoRA)
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

# LOAD LORA ADAPTER
model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
model.eval()

print("Checkpoint 1 loaded successfully!\n")

# PROMPT FORMAT
# MUST MATCH TRAINING
def build_prompt(text):
    return f"""### Instruction:
{text}

### Input:


### Response:
"""

# GENERATION FUNCTION
def generate(text):
    prompt = build_prompt(text)

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )

    input_ids = inputs["input_ids"].to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            max_new_tokens=150,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id
        )

    generated_tokens = outputs[0][input_ids.shape[-1]:]

    return tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()

# TEST CASES
test_prompts = [
    "Explain what a neural network is in simple terms.",
    "Write a short summary of machine learning.",
    "List three uses of artificial intelligence."
]

# RUN EVAL
print("=" * 50)
print("CHECKPOINT 1 EVALUATION")
print("=" * 50)

for i, p in enumerate(test_prompts, 1):
    print(f"\n--- SAMPLE {i} ---")
    print("INPUT:\n", p)
    print("\nOUTPUT:\n", generate(p))
    print("\n" + "-" * 50)
