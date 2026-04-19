import os
import json

RESULTS_DIR = "/work/qrc637/NLP_Assignment3/results"


# Load JSON
def load(path):
    with open(path, "r") as f:
        return json.load(f)


# Simple heuristic judge
def simple_score(text):
    """
    Heuristic scoring:
    - longer + structured answer = better proxy
    """
    if not text:
        return 0

    score = 0

    # length reward
    score += min(len(text) / 500, 2)

    # structure reward
    if "1." in text or "2." in text:
        score += 1

    if "##" in text or ":" in text:
        score += 0.5

    return score


# Compare two checkpoints
def compare(c1_file, c2_file):

    c1 = load(os.path.join(RESULTS_DIR, c1_file))
    c2 = load(os.path.join(RESULTS_DIR, c2_file))

    assert len(c1) == len(c2), "Mismatch dataset size"

    c1_wins = 0
    c2_wins = 0
    ties = 0

    for a, b in zip(c1, c2):

        s1 = simple_score(a["output"])
        s2 = simple_score(b["output"])

        if abs(s1 - s2) < 0.1:
            ties += 1
        elif s1 > s2:
            c1_wins += 1
        else:
            c2_wins += 1

    total = len(c1)

    print("\n========================================")
    print(f"FORGETTING: {c1_file} vs {c2_file}")
    print("========================================")

    print(f"C1 wins: {c1_wins}")
    print(f"C2 wins: {c2_wins}")
    print(f"Ties: {ties}")

    c1_rate = c1_wins / total
    c2_rate = c2_wins / total

    print("\nWIN RATES:")
    print(f"C1: {c1_rate:.3f}")
    print(f"C2: {c2_rate:.3f}")

    delta = c2_rate - c1_rate

    print("\nFORGETTING DELTA:")
    print(f"{delta:.3f}")

    if delta < 0:
        print("Forgetting detected (Alpaca degradation)")
    else:
        print("No forgetting or improvement")


# MAIN
if __name__ == "__main__":
    compare("C1_alpaca.json", "C2_alpaca.json")
