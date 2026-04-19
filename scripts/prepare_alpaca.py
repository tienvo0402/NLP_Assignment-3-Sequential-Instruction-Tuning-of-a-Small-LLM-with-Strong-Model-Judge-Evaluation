from datasets import load_dataset

# Step 1.2 - Load dataset
dataset = load_dataset("yahma/alpaca-cleaned")

# Step 1.3 - Inspect
print("Sample example:")
print(dataset["train"][0])

# Step 1.4 - Clean dataset
def is_valid(example):
    return (
        example["instruction"] is not None and
        example["output"] is not None and
        len(example["output"].strip()) > 0
    )

dataset = dataset["train"].filter(is_valid)

print(f"Dataset size after cleaning: {len(dataset)}")

# OPTIONAL: limit size for faster training
dataset = dataset.select(range(20000))
print(f"Dataset size after limiting: {len(dataset)}")

# Step 1.5 - Train/test split
dataset = dataset.train_test_split(test_size=0.1, seed=42)

train_data = dataset["train"]
eval_data = dataset["test"]

print(f"Train size: {len(train_data)}")
print(f"Eval size: {len(eval_data)}")

# Step 1.6 - Format for training
def format_example(example):
    if example["input"]:
        return {
            "text": f"### Instruction:\n{example['instruction']}\n\n### Input:\n{example['input']}\n\n### Response:\n{example['output']}"
        }
    else:
        return {
            "text": f"### Instruction:\n{example['instruction']}\n\n### Response:\n{example['output']}"
        }

train_data = train_data.map(format_example)
eval_data = eval_data.map(format_example)

# Step 1.7 - Save to JSON
train_data.to_json("alpaca_train.json")
eval_data.to_json("alpaca_eval.json")

