"""Leave-one-day-out robustness check (challenge brief, mandatory evaluation protocol #3:
'Test across collection days. Report the organizer's held-out-day or leave-one-day-out
robustness result in addition to any development split.').

This is deliberately separate from the Phase 1 dev split (train/val_cal/val_thr/test),
which is NOT day-disjoint by design (see docs/SESSION_CONSTRUCTION.md) and cannot answer
"how much does the model degrade on a calendar day it has never seen at all."

Only 18 capture sessions and 5 eligible days (2020-11-25 has zero benign rows and is
excluded, per docs/PHASE1_AUDIT.md C7) exist in the whole dataset. That scarcity means each
fold's remaining training data can't also afford a clean nested val_cal/val_thr split
without shrinking to near-nothing, so the operating threshold for each fold is picked
in-sample on that fold's own training data. This is an explicit, documented simplification
for a secondary robustness check -- it is NOT a substitute for the headline stateless model
protocol, which does keep threshold selection strictly out-of-sample.
"""
import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
import lightgbm as lgb

project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.data.load import load_raw, get_schema_lock
from src.eval.metrics import pick_threshold_at_fpr, point_metrics, ranking_metrics

SEED = 20260808
OPERATING_FPR = 0.001


def run_leave_one_day_out() -> dict:
    schema = get_schema_lock()
    features = schema["stateless_features"]

    full_df = load_raw()
    eligible_days = sorted(full_df["collection_day"].unique())

    folds = []
    for day in eligible_days:
        rest = full_df[full_df["collection_day"] != day]
        held = full_df[full_df["collection_day"] == day]

        model = lgb.LGBMClassifier(
            objective="binary", n_estimators=300, num_leaves=31, learning_rate=0.05,
            random_state=SEED, deterministic=True, force_row_wise=True, verbosity=-1,
        )
        model.fit(rest[features], rest["label"])

        rest_scores = model.predict_proba(rest[features])[:, 1]
        threshold = pick_threshold_at_fpr(rest["label"].values, rest_scores, target_fpr=OPERATING_FPR)

        held_scores = model.predict_proba(held[features])[:, 1]
        held_labels = held["label"].values
        has_both_classes = len(np.unique(held_labels)) > 1

        fold_result = {
            "held_out_day": day,
            "held_out_rows": int(len(held)),
            "held_out_benign_rows": int((held_labels == 0).sum()),
            "held_out_positive_rows": int((held_labels == 1).sum()),
            "held_out_attack_categories_present": sorted(held["attack_category"].unique().tolist()),
            "threshold_source": "in-sample on training fold (see module docstring)",
            "threshold": threshold,
            "point_metrics_on_held_out_day": point_metrics(held_labels, held_scores, threshold=threshold),
        }
        if has_both_classes:
            fold_result["pr_auc_on_held_out_day"] = ranking_metrics(held_labels, held_scores)["pr_auc"]
        else:
            fold_result["pr_auc_on_held_out_day"] = None
            fold_result["pr_auc_note"] = "undefined -- held-out day has only one class present"

        folds.append(fold_result)

    pr_aucs = [f["pr_auc_on_held_out_day"] for f in folds if f["pr_auc_on_held_out_day"] is not None]
    recalls = [f["point_metrics_on_held_out_day"]["recall"] for f in folds if f["held_out_positive_rows"] > 0]

    report = {
        "protocol": "leave-one-day-out (5 eligible days; 2020-11-25 excluded, zero benign rows)",
        "operating_fpr_target": OPERATING_FPR,
        "folds": folds,
        "fold_to_fold_variation": {
            "pr_auc_folds_used": len(pr_aucs),
            "pr_auc_mean": float(np.mean(pr_aucs)) if pr_aucs else None,
            "pr_auc_std": float(np.std(pr_aucs)) if pr_aucs else None,
            "pr_auc_min": float(np.min(pr_aucs)) if pr_aucs else None,
            "pr_auc_max": float(np.max(pr_aucs)) if pr_aucs else None,
            "recall_folds_used": len(recalls),
            "recall_mean": float(np.mean(recalls)) if recalls else None,
            "recall_std": float(np.std(recalls)) if recalls else None,
        },
    }

    out_dir = Path("results/metrics")
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "leave_one_day_out_report.json", "w") as f:
        json.dump(report, f, indent=2)

    print(json.dumps({"fold_to_fold_variation": report["fold_to_fold_variation"]}, indent=2))
    for fr in folds:
        print(f"  day={fr['held_out_day']}  rows={fr['held_out_rows']:6d}  "
              f"pos={fr['held_out_positive_rows']:6d}  cats={fr['held_out_attack_categories_present']}  "
              f"pr_auc={fr['pr_auc_on_held_out_day']}  recall={fr['point_metrics_on_held_out_day']['recall']:.4f}")
    return report


if __name__ == "__main__":
    run_leave_one_day_out()
