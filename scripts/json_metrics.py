import os
import json
import re

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE_DIR, "results")


# =========================
# Extract JSON safely from model output
# =========================
def extract_json(text):
    if not isinstance(text, str):
        return None

    # remove markdown code blocks first
    text = text.replace("```json", "").replace("```", "")

    # try direct parse
    try:
        return json.loads(text)
    except:
        pass

    # try to extract array JSON first (IMPORTANT for your case)
    match = re.search(r"\[[\s\S]*\]", text)
    if match:
        try:
            return json.loads(match.group(0))
        except:
            pass

    # fallback: object JSON
    match = re.search(r"\{[\s\S]*\}", text)
    if match:
        try:
            return json.loads(match.group(0))
        except:
            pass

    return None


# =========================
# Load results file
# =========================
def load_results(path):
    if not os.path.exists(path):
        print(f"[MISSING] {path}")
        return []

    with open(path, "r") as f:
        return json.load(f)


# =========================
# Schema validation (lightweight)
# =========================
def check_schema(obj):
    if not isinstance(obj, dict):
        return False

    # flexible schema check (covers your assignment types)
    allowed_keys_sets = [
        {"name", "price", "in_stock"},
        {"people", "dates"},
        {"event_name", "date", "participants"}
    ]

    obj_keys = set(obj.keys())

    for schema in allowed_keys_sets:
        if schema.issubset(obj_keys):
            return True

    return False


# =========================
# Evaluate file
# =========================
def evaluate_file(path):
    data = load_results(path)

    total = len(data)
    valid_json = 0
    schema_ok = 0

    for item in data:
        output = item.get("output", "")

        parsed = extract_json(output)

        if parsed is not None:
            valid_json += 1

            if check_schema(parsed):
                schema_ok += 1

    return {
        "total": total,
        "valid_json_rate": valid_json / total if total else 0,
        "schema_rate": schema_ok / total if total else 0
    }


# =========================
# MAIN
# =========================
def main():
    print("\n===== JSON METRICS =====")
    print("RESULTS DIR:", RESULTS_DIR)

    files = ["C0_json.json", "C1_json.json", "C2_json.json"]

    for f in files:
        path = os.path.join(RESULTS_DIR, f)

        metrics = evaluate_file(path)

        print(f"\n{f}")
        print(f"  Total: {metrics['total']}")
        print(f"  Valid JSON Rate: {metrics['valid_json_rate']:.2f}")
        print(f"  Schema Correct Rate: {metrics['schema_rate']:.2f}")


if __name__ == "__main__":
    main()