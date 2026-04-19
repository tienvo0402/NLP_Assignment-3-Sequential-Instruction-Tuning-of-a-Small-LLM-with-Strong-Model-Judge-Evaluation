import json
import random
import os
from tqdm import tqdm
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()  # load .env in root folder

client = OpenAI(
    api_key=os.getenv("UTSA_API_KEY"),
    base_url=os.getenv("UTSA_BASE_URL")
)

MODEL = os.getenv("UTSA_MODEL")



def is_valid_json(text):
    try:
        json.loads(text)
        return True
    except:
        return False


def build_prompt(instruction, input_text):
    return f"""
You are a strict JSON generator.

Return ONLY valid JSON.
Do not include explanations, markdown, or text.

Instruction:
{instruction}

Input:
{input_text}
"""


def generate_extraction():
    text = random.choice([
        "John met Mary on Jan 5, 2020 in New York.",
        "Alice visited Paris on March 10, 2021.",
        "Bob and Charlie traveled to Texas in 2019."
    ])
    return {
        "instruction": "Extract people and dates into JSON",
        "input": text
    }

def generate_schema():
    return {
        "instruction": "Generate JSON following schema {name: string, price: float, in_stock: bool}",
        "input": "Create a product entry"
    }

def generate_classification():
    text = random.choice([
        "I love this product!",
        "This is terrible.",
        "It's okay, nothing special."
    ])
    return {
        "instruction": "Classify sentiment into {positive, neutral, negative}",
        "input": text
    }

def generate_repair():
    broken = random.choice([
        "{name: John, age: 30",
        '{"city": "Austin", "temp": 85',
        "{'invalid': True"
    ])
    return {
        "instruction": "Fix the JSON",
        "input": broken
    }

def generate_tool():
    text = random.choice([
        "What's the weather in Austin in Celsius?",
        "Get weather for New York in Fahrenheit",
        "Weather in Dallas in Celsius"
    ])
    return {
        "instruction": "Generate arguments for function get_weather(city, unit)",
        "input": text
    }

task_functions = [
    generate_extraction,
    generate_schema,
    generate_classification,
    generate_repair,
    generate_tool
]

# =========================
# 🤖 CALL TEACHER MODEL
# =========================

def call_model(prompt):
    response = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7
    )
    return response.choices[0].message.content.strip()


dataset = []

TARGET_SIZE = 300  # change if needed

for _ in tqdm(range(TARGET_SIZE)):
    task_fn = random.choice(task_functions)
    example = task_fn()

    prompt = build_prompt(example["instruction"], example["input"])

    try:
        output = call_model(prompt)

        if is_valid_json(output):
            dataset.append({
                "instruction": example["instruction"],
                "input": example["input"],
                "output": output
            })
        else:
            print("Invalid JSON skipped")

    except Exception as e:
        print("Error:", e)



split = int(0.9 * len(dataset))

train_data = dataset[:split]
eval_data = dataset[split:]

with open("json_train.json", "w") as f:
    json.dump(train_data, f, indent=2)

with open("json_eval.json", "w") as f:
    json.dump(eval_data, f, indent=2)
