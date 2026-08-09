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

# Headline operating point target. NOT 0.1%: this model's benign score distribution is heavily
# quantized (71,012 val_thr benign rows -> 4,289 distinct scores), leaving a 15-point FPR band
# unreachable at any threshold (report section 7, results/metrics/operating_points_v1.json).
# 0.045 resolves to threshold 0.713607..., the most recall available BELOW that cliff
# (val_thr FPR 4.398%, test FPR 4.12%, recall 10.78%). The former 0.001 target resolved to
# 0.726207 / 0.229% recall -- retained in the threshold artifact as an alternate operating
# point, but it is not the reported headline.
OPERATING_FPR = 0.045

# Alternate operating points reported alongside the headline (report section 7). Recorded in the
# threshold artifact so the product and the report cannot silently diverge on which point is live.
ALTERNATE_FPR_TARGETS = {
    "tight_fpr_0.1pct": 0.001,   # -> 0.726207, 0.229% recall; abandoned as headline
    # 0.1942, not 0.20: targets above ~0.1943 walk PAST the first post-cliff point to lower
    # thresholds still inside budget, landing on 0.709489 instead of the 0.712416 the report
    # and results bundle use. Verified: 0.1942/0.1943 -> 0.712416; 0.195 -> 0.712261.
    "post_cliff": 0.1942,        # -> 0.712416, 47.06% recall at 17.95% test FPR
}
SEED = 20260808


def train_stateless_model(use_engineered: bool = True):
    """use_engineered=False trains v1 (base 14 gated features only) -- the surviving
    headline detector after sl2_session_repeat_count/ratio (v2) failed the mixed-session
    stress test (see docs/PHASE2_PLAN.md item 8 and the rejected-approach section of the
    report). use_engineered=True keeps producing the v2 artifacts, retained ONLY as
    evidence for that rejected-approach writeup, never as a headline number again."""
    model_version = "stateless_lgbm_v2" if use_engineered else "stateless_lgbm_v1"
    suffix = "" if use_engineered else "_v1"

    schema = get_schema_lock()
    base_features = schema["stateless_features"]
    features = base_features + ENGINEERED_FEATURES if use_engineered else list(base_features)

    loader = build_engineered_features if use_engineered else (lambda df: df)
    train = loader(load_split("train"))
    val_cal = loader(load_split("val_cal"))
    val_thr = loader(load_split("val_thr"))
    test = loader(load_split("test"))

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
    y_val_thr = val_thr["label"].values
    threshold = pick_threshold_at_fpr(y_val_thr, raw_val_thr, target_fpr=OPERATING_FPR)

    # Achieved val_thr FPR at the selected threshold, and the alternate points, recorded so the
    # threshold artifact is auditable rather than a bare number.
    neg_val_thr = raw_val_thr[y_val_thr == 0]
    achieved_val_fpr = float((neg_val_thr >= threshold).mean())
    alternates = {}
    for name, target in ALTERNATE_FPR_TARGETS.items():
        alt_thr = pick_threshold_at_fpr(y_val_thr, raw_val_thr, target_fpr=target)
        alternates[name] = {
            "threshold": float(alt_thr),
            "fpr_target": target,
            "achieved_val_thr_fpr": float((neg_val_thr >= alt_thr).mean()),
        }

    # Score test (held out, untouched until now)
    raw_test = model.predict_proba(test[features])[:, 1]
    cal_test = calibrator.predict(raw_test)
    test = test.copy()
    test["raw_score"] = raw_test
    test["calibrated_score"] = cal_test
    test["prediction"] = (raw_test >= threshold).astype(int)

    report = build_report(test, threshold, model_version)
    report["feature_set"] = {"base": base_features,
                             "engineered": ENGINEERED_FEATURES if use_engineered else [],
                             "total": len(features)}
    report["scale_pos_weight"] = round(spw, 4)

    _save_artifacts(model, calibrator, threshold, test, report, features, model_version, suffix,
                    achieved_val_fpr=achieved_val_fpr, alternates=alternates)
    print(json.dumps({k: v for k, v in report.items() if k != "light_recall_bootstrap_ci"}, indent=2))
    print("light_recall_bootstrap_ci:", report["light_recall_bootstrap_ci"])
    return report


def build_report(test: pd.DataFrame, threshold: float, model_version: str = "stateless_lgbm_v1") -> dict:
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
        "model_version": model_version,
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
                    features: list, model_version: str, suffix: str = "",
                    achieved_val_fpr: float = None, alternates: dict = None):
    models_dir = Path("models")
    models_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "calibrator": calibrator, "model_version": model_version,
                 "features": features},
                models_dir / f"stateless_lgbm{suffix}.pkl")
    with open(models_dir / f"stateless_threshold{suffix}.json", "w") as f:
        json.dump({
            "threshold": threshold,
            "operating_fpr_target": OPERATING_FPR,
            "achieved_val_thr_fpr": achieved_val_fpr,
            "selected_on": "val_thr",
            "selection_rule": (
                "Most recall available below the 15-point score-quantization cliff: thresholds "
                "are enumerated from distinct val_thr benign score values, and this is the lowest "
                "(most recall-friendly) one whose val_thr FPR stays within the target. See "
                "report/TECHNICAL_REPORT.md section 7 and "
                "results/metrics/operating_points_v1.json:unreachable_fpr_bands."
            ),
            "model_version": model_version,
            "alternate_operating_points": alternates or {},
        }, f, indent=2)

    pred_dir = Path("results/predictions")
    pred_dir.mkdir(parents=True, exist_ok=True)
    test[["session_id", "raw_score", "calibrated_score", "prediction"]].assign(
        model_version=model_version
    ).to_csv(pred_dir / f"test_predictions_stateless{suffix}.csv", index=False)

    metrics_dir = Path("results/metrics")
    metrics_dir.mkdir(parents=True, exist_ok=True)
    with open(metrics_dir / f"stateless_model{suffix}_report.json", "w") as f:
        json.dump(report, f, indent=2)


if __name__ == "__main__":
    import sys as _sys
    train_stateless_model(use_engineered=("--v1" not in _sys.argv))
