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
    signed z-score sum. Fit (mean, std, sign) on train only, per the brief's 'fit
    on training data only' requirement.

    BUG FOUND AND FIXED: naively summing z(sl_len) + z(sl_entropy) drove ROC-AUC to
    ~0.499 (exactly chance) on this dataset. Diagnosis: sl_len alone has real signal
    on train (ROC-AUC 0.63, attack queries are longer) but sl_entropy alone points
    the OPPOSITE direction on this dataset (ROC-AUC 0.449 -- attack entropy is
    slightly LOWER than benign here, contrary to the textbook "exfil payloads are
    high-entropy" assumption). Adding the two raw z-scores cancels the length
    signal against the (backwards, for this data) entropy signal, producing an
    apparently-broken baseline that was actually two real signals fighting each
    other. Fix: determine each feature's sign from TRAIN data only (whichever
    direction correlates with the attack class there), so both terms point the
    same way. This is a standard non-ML baseline convention (same sign-direction
    logic already used in src/models/cascade.py's discriminative feature ranking),
    not new modelling -- the rule and its two input features are unchanged."""
    stats = {}
    for feat in RULE_FEATURES:
        benign_mean = float(train.loc[train["label"] == 0, feat].mean())
        attack_mean = float(train.loc[train["label"] == 1, feat].mean())
        sign = 1.0 if attack_mean > benign_mean else -1.0
        stats[feat] = {"mean": float(train[feat].mean()), "std": float(train[feat].std() or 1.0),
                       "sign": sign, "train_benign_mean": benign_mean, "train_attack_mean": attack_mean}
    return stats


def score_rule(df: pd.DataFrame, stats: dict) -> np.ndarray:
    z_sum = np.zeros(len(df), dtype=np.float64)
    for feat in RULE_FEATURES:
        z_sum += stats[feat]["sign"] * (df[feat].values - stats[feat]["mean"]) / stats[feat]["std"]
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
