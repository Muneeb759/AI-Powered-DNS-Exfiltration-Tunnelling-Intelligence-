import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
import lightgbm as lgb
from sklearn.isotonic import IsotonicRegression

project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.data.load import load_split, get_schema_lock
from src.eval.metrics import pick_threshold_at_fpr, point_metrics, ranking_metrics, bootstrap_ci, recall_at_fpr_curve
from src.features.stateless_engineered import build_engineered_features, ENGINEERED_FEATURES

OPERATING_FPR = 0.001
SEED = 20260808
MODEL_VERSION = "stateless_lgbm_v2"


def train_stateless_model():
    schema = get_schema_lock()
    base_features = schema["stateless_features"]
    features = base_features + ENGINEERED_FEATURES

    train = build_engineered_features(load_split("train"))
    val_cal = build_engineered_features(load_split("val_cal"))
    val_thr = build_engineered_features(load_split("val_thr"))
    test = build_engineered_features(load_split("test"))

    # scale_pos_weight corrects for class imbalance: penalises missing an attack row
    # proportionally to how rare attacks are in the training set.
    n_neg = int((train["label"] == 0).sum())
    n_pos = int((train["label"] == 1).sum())
    spw = n_neg / n_pos

    model = lgb.LGBMClassifier(
        objective="binary",
        n_estimators=300,
        num_leaves=31,
        learning_rate=0.05,
        scale_pos_weight=spw,
        random_state=SEED,
        deterministic=True,
        force_row_wise=True,
        verbosity=-1,
    )
    model.fit(train[features], train["label"])

    # Isotonic calibration fit on val_cal ONLY (not train, not test).
    # NOTE: isotonic regression is a monotonic *step function* -- at this dataset's scale it
    # collapses the benign score distribution into a handful of large plateaus (e.g. 26k of
    # 71k val_thr benign rows tied at one calibrated value). That's far too coarse to hit an
    # extreme operating point like FPR=0.1%: the closest achievable threshold either admits an
    # entire tied plateau (FPR overshoots by 40x) or excludes it entirely (recall collapses to
    # ~0). The raw LightGBM score has ~4.5k distinct values in the same range (finest tie = 4
    # rows), so the operating threshold is selected on the RAW score, which the brief's FPR
    # requirement actually needs the resolution for. The calibrated score is still produced and
    # reported (isotonic-calibrated probability, useful for human-facing confidence display) but
    # is not what the alert/no-alert decision is based on.
    raw_val_cal = model.predict_proba(val_cal[features])[:, 1]
    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(raw_val_cal, val_cal["label"].values)

    # Threshold selection on val_thr ONLY, using the raw (uncalibrated) score
    raw_val_thr = model.predict_proba(val_thr[features])[:, 1]
    threshold = pick_threshold_at_fpr(val_thr["label"].values, raw_val_thr, target_fpr=OPERATING_FPR)

    # Score test (held out, untouched until now)
    raw_test = model.predict_proba(test[features])[:, 1]
    cal_test = calibrator.predict(raw_test)
    test = test.copy()
    test["raw_score"] = raw_test
    test["calibrated_score"] = cal_test
    test["prediction"] = (raw_test >= threshold).astype(int)

    report = build_report(test, threshold)
    report["feature_set"] = {"base": base_features, "engineered": ENGINEERED_FEATURES,
                             "total": len(features)}
    report["scale_pos_weight"] = round(spw, 4)

    _save_artifacts(model, calibrator, threshold, test, report, features)
    print(json.dumps({k: v for k, v in report.items() if k != "light_recall_bootstrap_ci"}, indent=2))
    print("light_recall_bootstrap_ci:", report["light_recall_bootstrap_ci"])
    return report


def build_report(test: pd.DataFrame, threshold: float) -> dict:
    # Ranking metrics and the operating decision are evaluated on raw_score -- see the note in
    # train_stateless_model() on why the isotonic-calibrated score is too coarse (large tied
    # plateaus) to resolve an extreme operating point like FPR=0.1%.
    y_true = test["label"].values
    scores = test["raw_score"].values
    rm = ranking_metrics(y_true, scores)
    pm = point_metrics(y_true, scores, threshold=threshold)

    benign = test[test["label"] == 0]

    def slice_metrics(category: str) -> dict:
        pos = test[test["attack_category"] == category]
        combined = pd.concat([benign, pos], ignore_index=True)
        y = combined["label"].values
        s = combined["raw_score"].values
        rm_s = ranking_metrics(y, s)
        pm_s = point_metrics(y, s, threshold=threshold)
        return {"pr_auc": rm_s["pr_auc"], "roc_auc_supplementary": rm_s["roc_auc"], **pm_s}

    def _recall_or_raise(y, s):
        if not (y == 1).any():
            raise ValueError("no positives in bootstrap sample")
        return point_metrics(y, s, threshold=threshold)["recall"]

    light_only = test[test["attack_category"] == "light"].copy()
    light_recall_ci = bootstrap_ci(
        light_only.rename(columns={"raw_score": "score"}),
        metric_fn=_recall_or_raise,
        group_col="session_id",
        n_bootstraps=1000,
        seed=SEED,
    )

    curve = recall_at_fpr_curve(y_true, scores, target_fprs=[0.01, 0.005, 0.002, 0.001])

    return {
        "model_version": MODEL_VERSION,
        "operating_fpr_target": OPERATING_FPR,
        "headline_pr_auc": rm["pr_auc"],
        "supplementary_roc_auc": rm["roc_auc"],
        "threshold": threshold,
        "combined": pm,
        "light": slice_metrics("light"),
        "heavy": slice_metrics("heavy"),
        "light_recall_bootstrap_ci": light_recall_ci,
        "light_recall_bootstrap_ci_note": (
            "Zero-width CI is structural, not a precise estimate: all test-set light-attack "
            "positives (3,479 rows) belong to a single capture trace, so session-grouped "
            "bootstrap resampling has only one cluster to draw from. See summary.md's own "
            "flag on this and its recommended leave-one-day-out backstop (docs/PHASE2_PLAN.md item 1)."
        ),
        "recall_vs_fpr_curve": curve.to_dict(orient="records"),
    }


def _save_artifacts(model, calibrator, threshold: float, test: pd.DataFrame, report: dict,
                    features: list):
    models_dir = Path("models")
    models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "calibrator": calibrator, "model_version": MODEL_VERSION,
                 "features": features},
                models_dir / "stateless_lgbm.pkl")
    with open(models_dir / "stateless_threshold.json", "w") as f:
        json.dump({"threshold": threshold, "operating_fpr_target": OPERATING_FPR,
                   "model_version": MODEL_VERSION}, f, indent=2)

    pred_dir = Path("results/predictions")
    pred_dir.mkdir(parents=True, exist_ok=True)
    test[["session_id", "raw_score", "calibrated_score", "prediction"]].assign(
        model_version=MODEL_VERSION
    ).to_csv(pred_dir / "test_predictions_stateless.csv", index=False)

    metrics_dir = Path("results/metrics")
    metrics_dir.mkdir(parents=True, exist_ok=True)
    with open(metrics_dir / "stateless_model_report.json", "w") as f:
        json.dump(report, f, indent=2)


if __name__ == "__main__":
    train_stateless_model()
