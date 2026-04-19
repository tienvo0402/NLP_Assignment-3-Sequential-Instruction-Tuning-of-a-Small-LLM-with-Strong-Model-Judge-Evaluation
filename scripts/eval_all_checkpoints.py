import os
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# PATHS
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(BASE_DIR, "../data")
RESULTS_DIR = os.path.join(BASE_DIR, "../results")
os.makedirs(RESULTS_DIR, exist_ok=True)

ALPACA_TEST = os.path.join(DATA_DIR, "alpaca_eval.json")
JSON_TEST = os.path.join(DATA_DIR, "json_eval.json")

CHECKPOINTS = {
    "C0": None,
    "C1": os.path.join(BASE_DIR, "outputs/stage1_debug"),
    "C2": os.path.join(BASE_DIR, "outputs/stage2_adapter")
}

BASE_MODEL = "microsoft/Phi-3.5-mini-instruct"

# DATA LOADER

def load_dataset(path):
    if not os.path.exists(path):
        print(f"[ERROR] Missing dataset: {path}")
        return []

    with open(path, "r") as f:
        content = f.read().strip()

    try:
        return json.loads(content)
    except:
        return [json.loads(line) for line in content.splitlines() if line.strip()]

# MODEL LOADER

def load_model_and_tokenizer(path):
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )

    if path is not None:
        model = PeftModel.from_pretrained(model, path)

    model.eval()
    return model, tokenizer

# GENERATION

def generate(model, tokenizer, prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.7,
            do_sample=True
        )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# SAVE RESULTS

def save_results(checkpoint_name, task_name, results):
    path = os.path.join(RESULTS_DIR, f"{checkpoint_name}_{task_name}.json")

    with open(path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"[SAVED] {path}")

# EVALUATION

def evaluate(model, tokenizer, checkpoint_name, dataset_path, task_name):
    data = load_dataset(dataset_path)

    print(f"\n=== {checkpoint_name} - {task_name} ===")

    results = []

    for i, item in enumerate(data[:3]):
        prompt = item["instruction"]

        output = generate(model, tokenizer, prompt)

        print(f"\n--- SAMPLE {i+1} ---")
        print("INPUT:", prompt)
        print("OUTPUT:", output[:300], "...")

        results.append({
            "instruction": prompt,
            "output": output
        })

    save_results(checkpoint_name, task_name, results)

# RUN

def run_checkpoint(name, path):
    print("\n" + "#" * 60)
    print(f"CHECKPOINT: {name}")
    print("#" * 60)

    model, tokenizer = load_model_and_tokenizer(path)

    evaluate(model, tokenizer, name, ALPACA_TEST, "alpaca")
    evaluate(model, tokenizer, name, JSON_TEST, "json")


def main():
    for name, path in CHECKPOINTS.items():
        run_checkpoint(name, path)


if __name__ == "__main__":
    main()
