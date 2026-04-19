import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

# CONFIG
BASE_MODEL = "microsoft/Phi-3.5-mini-instruct"

STAGE2_PATH = "outputs/stage2_adapter"

device = "cuda" if torch.cuda.is_available() else "cpu"

# TOKENIZER

tokenizer = AutoTokenizer.from_pretrained(
    BASE_MODEL,
    trust_remote_code=True
)

tokenizer.pad_token = tokenizer.eos_token


# LOAD BASE MODEL (4-bit QLoRA style)

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


# LOAD STAGE 2 ADAPTER

model = PeftModel.from_pretrained(base_model, STAGE2_PATH)
model.eval()

print("Checkpoint 2 loaded successfully!\n")


# PROMPT FORMAT (SAME AS TRAINING)

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
    ).to(device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=150,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            repetition_penalty=1.1,
            eos_token_id=tokenizer.eos_token_id
        )

    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # remove prompt section
    return decoded.split("### Response:")[-1].strip()


# TEST SET (SAME AS C0 / C1)
test_prompts = [
    "Explain what a neural network is in simple terms.",
    "Write a short summary of machine learning.",
    "List three uses of artificial intelligence."
]

# RUN EVALUATION
print("=" * 50)
print("CHECKPOINT 2 EVALUATION (STAGE 2 MODEL)")
print("=" * 50)

for i, prompt in enumerate(test_prompts, 1):
    print(f"\n--- SAMPLE {i} ---")
    print("INPUT:\n", prompt)
    print("\nOUTPUT:\n")
    print(generate(prompt))
    print("\n" + "-" * 50)
