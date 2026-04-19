import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# ======================
# MODEL
# ======================
MODEL_NAME = "microsoft/Phi-3.5-mini-instruct"

print("Loading Checkpoint 0 (base model)...")

tokenizer = AutoTokenizer.from_pretrained(
    MODEL_NAME,
    trust_remote_code=True
)

tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    torch_dtype=torch.float16,
    trust_remote_code=True,
    attn_implementation="eager"
)

model.eval()

print("Checkpoint 0 loaded successfully!\n")


# ======================
# GENERATION
# ======================
def generate(prompt):

    messages = [
        {"role": "user", "content": prompt}
    ]

    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    )

    if isinstance(inputs, dict):
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        input_ids = inputs["input_ids"]
    else:
        inputs = inputs.to(model.device)
        input_ids = inputs

    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            max_new_tokens=120,
            do_sample=False,
            use_cache=False,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id
        )

    # ✅ remove prompt from output
    generated_tokens = outputs[0][input_ids.shape[-1]:]

    return tokenizer.decode(generated_tokens, skip_special_tokens=True)


# ======================
# TEST SET
# ======================
test_prompts = [
    "Explain what a neural network is in simple terms.",
    "Write a short summary of machine learning.",
    "List three uses of artificial intelligence."
]


# ======================
# EVALUATION
# ======================
print("=" * 50)
print("CHECKPOINT 0 EVALUATION (BASE MODEL)")
print("=" * 50)

for i, prompt in enumerate(test_prompts):
    print(f"\n--- SAMPLE {i+1} ---")
    print("INPUT:\n", prompt)
    print("\nOUTPUT:\n")
    print(generate(prompt))
    print("\n" + "-" * 50)