import os
import json
import pandas as pd
from tqdm import tqdm
import re

from bert_score import score as bert_score
from collections import Counter


# PATH
RESULTS_DIR = "/work/qrc637/NLP_Assignment3/results"


# SIMPLE ROUGE (NO NLTK)

def simple_rouge(pred, ref):
    pred_tokens = pred.lower().split()
    ref_tokens = ref.lower().split()

    pred_counter = Counter(pred_tokens)
    ref_counter = Counter(ref_tokens)

    overlap = sum((pred_counter & ref_counter).values())

    precision = overlap / max(len(pred_tokens), 1)
    recall = overlap / max(len(ref_tokens), 1)

    if precision + recall == 0:
        f1 = 0.0
    else:
        f1 = 2 * precision * recall / (precision + recall)

    return f1


# LOAD DATA
def load(path):
    with open(path, "r") as f:
        return json.load(f)


# EVALUATE FILE
def evaluate(path):
    data = load(path)

    rouge_scores = []
    preds = []
    refs = []

    for item in tqdm(data, desc=os.path.basename(path)):
        pred = item.get("output", "")
        ref = item.get("instruction", "")  # fallback reference

        rouge_scores.append(simple_rouge(pred, ref))
        preds.append(pred)
        refs.append(ref)

    # BERTScore
    P, R, F1 = bert_score(preds, refs, lang="en", verbose=False)

    return {
        "ROUGE-L(simple)": sum(rouge_scores) / len(rouge_scores),
        "BERTScore": float(F1.mean())
    }


# MAIN
def main():
    files = [
        "C0_alpaca.json",
        "C1_alpaca.json",
        "C2_alpaca.json"
    ]

    results = []

    print("\n===== SAFE NLP METRICS =====\n")

    for f in files:
        path = os.path.join(RESULTS_DIR, f)

        if not os.path.exists(path):
            print(f"[MISSING] {path}")
            continue

        metrics = evaluate(path)

        print(f"\n{f}")
        print(f"  ROUGE-L (simple): {metrics['ROUGE-L(simple)']:.4f}")
        print(f"  BERTScore: {metrics['BERTScore']:.4f}")

        results.append({
            "checkpoint": f,
            **metrics
        })

    df = pd.DataFrame(results)
    out_path = os.path.join(RESULTS_DIR, "nlp_metrics_safe.csv")
    df.to_csv(out_path, index=False)

    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
