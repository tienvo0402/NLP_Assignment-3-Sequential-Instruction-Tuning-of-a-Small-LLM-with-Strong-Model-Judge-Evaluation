import json
import random

# =========================
# JSON TASK TEMPLATES
# =========================

def json_extraction():
    return {
        "instruction": "Extract entities from the text into JSON format",
        "input": "John Doe visited New York on March 3rd 2024 and met Jane Smith.",
        "output": json.dumps({
            "persons": ["John Doe", "Jane Smith"],
            "locations": ["New York"],
            "dates": ["March 3rd 2024"]
        }, indent=2)
    }

def schema_generation():
    return {
        "instruction": "Generate a JSON object following this schema: {name, age, city}",
        "input": "Create a profile for a student",
        "output": json.dumps({
            "name": "Alice",
            "age": 21,
            "city": "San Antonio"
        }, indent=2)
    }

def classification():
    return {
        "instruction": "Classify sentiment as positive, negative, or neutral in JSON format",
        "input": "The movie was surprisingly good and enjoyable.",
        "output": json.dumps({
            "sentiment": "positive"
        }, indent=2)
    }

def json_repair():
    return {
        "instruction": "Fix the malformed JSON",
        "input": "{name: John, age: 25 city: New York}",
        "output": json.dumps({
            "name": "John",
            "age": 25,
            "city": "New York"
        }, indent=2)
    }

def tool_call():
    return {
        "instruction": "Convert request into API call JSON",
        "input": "Book a flight from NYC to LA on May 5th",
        "output": json.dumps({
            "tool": "book_flight",
            "parameters": {
                "from": "NYC",
                "to": "LA",
                "date": "May 5th"
            }
        }, indent=2)
    }

# =========================
# BUILD DATASET
# =========================

dataset = []

for _ in range(400):
    dataset.append(json_extraction())
    dataset.append(schema_generation())
    dataset.append(classification())
    dataset.append(json_repair())
    dataset.append(tool_call())

random.shuffle(dataset)

# =========================
# SAVE
# =========================

with open("stage2_data.json", "w") as f:
    json.dump(dataset, f, indent=2)

print("Stage 2 dataset created:", len(dataset))