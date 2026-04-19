import os
import json

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import torch

# ==================================================
# FIXED PATH HANDLING (IMPORTANT)
# ==================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.abspath(os.path.join(BASE_DIR, "../results"))

ALPACA_FILES = {
    "C0": os.path.join(RESULTS_DIR, "C0_alpaca.json"),
    "C1": os.path.join(RESULTS_DIR, "C1_alpaca.json"),
    "C2": os.path.join(RESULTS_DIR, "C2_alpaca.json"),
}

JSON_FILES = {
    "C0": os.path.join(RESULTS_DIR, "C0_json.json"),
    "C1": os.path.join(RESULTS_DIR, "C1_json.json"),
    "C2": os.path.join(RESULTS_DIR, "C2_json.json"),
}

BASE_MODEL = "microsoft/Phi-3.5-mini-instruct"


# ==================================================
# LOAD RESULTS SAFE
# ==================================================
def load_results(path):
    if not os.path.exists(path):
        print(f"[MISSING] {path}")
        return []

    with open(path, "r") as f:
        return json.load(f)


# ==================================================
# SIMPLE JUDGE (SAFE BASELINE)
# ==================================================
def judge(a, b):
    # simple heuristic (you can upgrade later if needed)
    if len(a) > len(b):
        return "A"
    elif len(b) > len(a):
        return "B"
    return "Tie"


# ==================================================
# COMPARE FUNCTION
# ==================================================
def compare(file_a, file_b):

    print("\n" + "=" * 40)
    print(f"COMPARE: {file_a} vs {file_b}")
    print("=" * 40)

    data_a = load_results(file_a)
    data_b = load_results(file_b)

    if len(data_a) == 0 or len(data_b) == 0:
        print("No data found — check paths!")
        return

    a_win, b_win, tie = 0, 0, 0

    for i in range(min(len(data_a), len(data_b))):
        out_a = data_a[i]["output"]
        out_b = data_b[i]["output"]

        result = judge(out_a, out_b)

        if result == "A":
            a_win += 1
        elif result == "B":
            b_win += 1
        else:
            tie += 1

    print(f"A wins: {a_win}")
    print(f"B wins: {b_win}")
    print(f"Ties: {tie}")


# ==================================================
# MAIN
# ==================================================
def main():

    print("\nRESULTS DIR:", RESULTS_DIR)

    # Alpaca comparisons
    compare(ALPACA_FILES["C0"], ALPACA_FILES["C1"])
    compare(ALPACA_FILES["C1"], ALPACA_FILES["C2"])

    # JSON comparisons
    compare(JSON_FILES["C0"], JSON_FILES["C1"])
    compare(JSON_FILES["C1"], JSON_FILES["C2"])


if __name__ == "__main__":
    main()