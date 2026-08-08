"""Full worst-5 false-positive / worst-5 false-negative failure table (brief
requirement) -- completes what report/TECHNICAL_REPORT.md Section 12 previously
covered with only one example each.

"Worst" is defined by how confidently wrong the model was, not just which side of
the threshold a row landed on:
  - Worst FN: true attack rows with the LOWEST raw_score -- the model was most
    confident these were benign, despite being attack.
  - Worst FP: benign rows with the HIGHEST raw_score. At the current operating
    threshold (v2 model), there are ZERO false positives (see
    results/metrics/stateless_model_report.json:combined -- fp=0). Rather than
    invent examples, this reports that honestly and shows the highest-scoring
    benign rows as "closest calls" (near-misses that never crossed the alert
    threshold), clearly labelled as not actual false positives.
"""
import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.data.load import load_split

REPORT_FEATURES = [
    "sl_fqdn_count", "sl_subdomain_length", "sl_upper", "sl_lower", "sl_numeric",
    "sl_entropy", "sl_special", "sl_labels", "sl_len", "sl_labels_max",
    "sl_labels_average", "sl_subdomain_flag", "sl_longest_word_len", "sl_numeric_ratio",
]


def run_failure_analysis(top_k: int = 5) -> dict:
    bundle = joblib.load("models/stateless_lgbm_v1.pkl")
    model, features = bundle["model"], bundle["features"]
    threshold = json.load(open("models/stateless_threshold_v1.json"))["threshold"]

    test = load_split("test")
    test = test.copy()
    test["raw_score"] = model.predict_proba(test[features])[:, 1]
    test["prediction"] = (test["raw_score"] >= threshold).astype(int)

    def row_record(row) -> dict:
        return {
            "unit_id": row["unit_id"],
            "session_id": row["session_id"],
            "attack_category": row["attack_category"],
            "raw_score": float(row["raw_score"]),
            "threshold": float(threshold),
            "features": {f: float(row[f]) for f in REPORT_FEATURES},
        }

    fn_mask = (test["label"] == 1) & (test["prediction"] == 0)
    n_fn = int(fn_mask.sum())
    worst_fn = test[fn_mask].nsmallest(top_k, "raw_score")
    worst_fn_records = [row_record(r) for _, r in worst_fn.iterrows()]

    fp_mask = (test["label"] == 0) & (test["prediction"] == 1)
    n_fp = int(fp_mask.sum())

    if n_fp > 0:
        worst_fp = test[fp_mask].nlargest(top_k, "raw_score")
        fp_records = [row_record(r) for _, r in worst_fp.iterrows()]
        fp_note = f"{n_fp} actual false positives at threshold {threshold:.4f} -- shown below."
    else:
        near_miss = test[test["label"] == 0].nlargest(top_k, "raw_score")
        fp_records = [row_record(r) for _, r in near_miss.iterrows()]
        fp_note = (
            f"0 false positives at the operating threshold ({threshold:.4f}) -- precision is "
            "100% on this test set. The rows below are the highest-scoring BENIGN rows in test "
            "(closest calls), none of which crossed the alert threshold. They are near-misses, "
            "not actual false positives, and are labelled as such rather than presented as failures."
        )

    result = {
        "threshold": float(threshold),
        "n_false_negatives": n_fn,
        "n_false_positives": n_fp,
        "worst_false_negatives": worst_fn_records,
        "false_positive_note": fp_note,
        "false_positive_examples_or_near_misses": fp_records,
    }

    out_dir = Path("results/metrics")
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "failure_analysis.json", "w") as f:
        json.dump(result, f, indent=2)

    print(json.dumps({k: v for k, v in result.items()
                      if k not in ("worst_false_negatives", "false_positive_examples_or_near_misses")},
                     indent=2))
    print(f"n_false_negatives={n_fn}  n_false_positives={n_fp}")
    return result


if __name__ == "__main__":
    run_failure_analysis()
