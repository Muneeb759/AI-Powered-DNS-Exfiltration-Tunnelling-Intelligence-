import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd

project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.data.load import load_split
from src.eval.metrics import pick_threshold_at_fpr, point_metrics, ranking_metrics

RULE_FEATURES = ["sl_len", "sl_entropy"]
OPERATING_FPR = 0.001


def fit_rule_score(train: pd.DataFrame) -> dict:
    """Mandatory non-ML baseline: query length + Shannon entropy, combined as a
    z-score sum. Fit (mean, std) on train only, per the brief's 'fit on training
    data only' requirement -- the rule itself is fixed a priori, but its
    normalization constants must not see val/test."""
    stats = {}
    for feat in RULE_FEATURES:
        stats[feat] = {"mean": float(train[feat].mean()), "std": float(train[feat].std() or 1.0)}
    return stats


def score_rule(df: pd.DataFrame, stats: dict) -> np.ndarray:
    z_sum = np.zeros(len(df), dtype=np.float64)
    for feat in RULE_FEATURES:
        z_sum += (df[feat].values - stats[feat]["mean"]) / stats[feat]["std"]
    return z_sum


def run_rule_baseline() -> dict:
    train = load_split("train")
    val_thr = load_split("val_thr")
    test = load_split("test")

    stats = fit_rule_score(train)

    val_thr = val_thr.copy()
    val_thr["score"] = score_rule(val_thr, stats)
    threshold = pick_threshold_at_fpr(val_thr["label"].values, val_thr["score"].values, target_fpr=OPERATING_FPR)

    test = test.copy()
    test["score"] = score_rule(test, stats)

    report = {"rule": "z(sl_len) + z(sl_entropy), normalization fit on train only",
              "rule_stats": stats,
              "operating_fpr_target": OPERATING_FPR,
              "threshold": threshold,
              "combined": _slice_report(test, threshold),
              "light": _slice_report(test[test["attack_category"] == "light"].copy(), threshold, all_labels_ref=test),
              "heavy": _slice_report(test[test["attack_category"] == "heavy"].copy(), threshold, all_labels_ref=test)}

    out_dir = Path("results/metrics")
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "rule_baseline_report.json", "w") as f:
        json.dump(report, f, indent=2)

    pred_dir = Path("results/predictions")
    pred_dir.mkdir(parents=True, exist_ok=True)
    test["prediction"] = (test["score"] >= threshold).astype(int)
    test[["session_id", "score", "prediction"]].rename(columns={"score": "raw_score"}).assign(
        calibrated_score=lambda d: d["raw_score"], model_version="rule_baseline_v1"
    )[["session_id", "raw_score", "calibrated_score", "prediction", "model_version"]].to_csv(
        pred_dir / "test_predictions_rule.csv", index=False
    )

    print(json.dumps({k: v for k, v in report.items() if k != "rule_stats"}, indent=2))
    return report


def _slice_report(df_slice: pd.DataFrame, threshold: float, all_labels_ref: pd.DataFrame = None) -> dict:
    """For light/heavy slices, benign rows must come from the FULL test set (a light/heavy
    slice by attack_category has no benign rows of its own -- FPR is only meaningful against
    the whole benign population)."""
    if all_labels_ref is not None:
        benign_rows = all_labels_ref[all_labels_ref["label"] == 0]
        combined = pd.concat([benign_rows, df_slice], ignore_index=True)
    else:
        combined = df_slice

    y_true = combined["label"].values
    scores = combined["score"].values
    rm = ranking_metrics(y_true, scores)
    pm = point_metrics(y_true, scores, threshold=threshold)
    return {"pr_auc": rm["pr_auc"], "roc_auc_supplementary": rm["roc_auc"], **pm}


if __name__ == "__main__":
    run_rule_baseline()
