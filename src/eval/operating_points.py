"""Full achievable operating-point curve + calibration quality for the v1 detector.

Replaces the previous fixed-target-FPR curve, which produced duplicate rows
(three distinct points from six targets) and looked like a coarse grid search.
Root cause was NOT a coarse grid: this model's benign score distribution is
heavily quantized (71,012 val_thr benign rows -> only 4,289 distinct scores,
with large plateaus), so most target FPRs resolve to the same threshold and
whole FPR bands are simply unreachable.

This enumerates every DISTINCT achievable operating point instead -- one row per
distinct benign score value on val_thr, evaluated on test -- so the reachable
points and the gaps between them are both visible, and no operating point is
claimed that the score distribution cannot actually produce.

Also computes calibration quality (Brier score + 10-bin reliability table, raw
vs isotonic-calibrated) to close Track 1's "calibrated confidence and an explicit
operating threshold" criterion: calibrated probabilities are reported for display
and ranking, while the raw score drives the operating threshold (see
src/models/stateless_model.py for why -- isotonic plateaus are too coarse to
resolve a tight FPR budget).
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
from src.eval.metrics import point_metrics

MAX_VAL_FPR = 0.35  # stop enumerating past this -- beyond it the rule is not operationally useful


def _slice_recall(df: pd.DataFrame, category: str, threshold: float) -> float:
    sub = df[(df["label"] == 0) | (df["attack_category"] == category)]
    return point_metrics(sub["label"].values, sub["raw_score"].values, threshold=threshold)["recall"]


def build_operating_point_curve(val_thr: pd.DataFrame, test: pd.DataFrame) -> list:
    neg = val_thr.loc[val_thr["label"] == 0, "raw_score"].values
    distinct_desc = np.unique(neg)[::-1]

    rows, seen = [], set()
    for cand in distinct_desc:
        val_fpr = float((neg >= cand).mean())
        if val_fpr > MAX_VAL_FPR:
            break
        key = round(val_fpr, 5)
        if key in seen:
            continue
        seen.add(key)
        pm = point_metrics(test["label"].values, test["raw_score"].values, threshold=float(cand))
        rows.append({
            "threshold": float(cand),
            "val_thr_fpr": val_fpr,
            "test_fpr": pm["fpr"],
            "test_recall_combined": pm["recall"],
            "test_precision_combined": pm["precision"],
            "test_recall_light": _slice_recall(test, "light", float(cand)),
            "test_recall_heavy": _slice_recall(test, "heavy", float(cand)),
            "test_false_alerts_per_10k_benign": pm["false_alerts_per_10k_benign"],
            "tp": pm["tp"], "fp": pm["fp"], "fn": pm["fn"], "tn": pm["tn"],
        })
    return rows


def calibration_report(test: pd.DataFrame, n_bins: int = 10) -> dict:
    """Brier score + reliability table for raw vs isotonic-calibrated scores."""
    y = test["label"].values.astype(float)

    def _brier(p):
        return float(np.mean((p - y) ** 2))

    def _reliability(p):
        bins = np.linspace(0.0, 1.0, n_bins + 1)
        idx = np.clip(np.digitize(p, bins) - 1, 0, n_bins - 1)
        out = []
        for b in range(n_bins):
            mask = idx == b
            n = int(mask.sum())
            if n == 0:
                out.append({"bin": f"[{bins[b]:.1f},{bins[b+1]:.1f})", "n": 0,
                            "mean_predicted": None, "observed_frequency": None})
                continue
            out.append({
                "bin": f"[{bins[b]:.1f},{bins[b+1]:.1f})",
                "n": n,
                "mean_predicted": float(p[mask].mean()),
                "observed_frequency": float(y[mask].mean()),
                "gap": float(p[mask].mean() - y[mask].mean()),
            })
        return out

    raw, cal = test["raw_score"].values, test["calibrated_score"].values
    return {
        "brier_raw": _brier(raw),
        "brier_isotonic_calibrated": _brier(cal),
        "reliability_raw": _reliability(raw),
        "reliability_isotonic_calibrated": _reliability(cal),
        "note": (
            "Isotonic calibration is fit on val_cal only. Calibrated probabilities are what the "
            "analyst UI displays and what ranking uses; the RAW score drives the operating "
            "threshold, because isotonic's step-function output collapses benign scores into a "
            "few large plateaus that cannot resolve a tight FPR budget (see "
            "src/models/stateless_model.py). Both are reported here so the trade-off is visible "
            "rather than asserted."
        ),
    }


def confusion_matrices(test: pd.DataFrame, threshold: float) -> dict:
    """2x2 confusion matrix at the headline threshold, combined and per attack slice.
    Brief requirement: "a confusion matrix and false alerts per 10,000 benign decisions."
    Written to an artifact so the report table is traceable rather than typed in."""
    y = test["label"].values
    s = test["raw_score"].values

    def _cm(mask) -> dict:
        pm = point_metrics(y[mask], s[mask], threshold=threshold)
        return {
            "true_positive": pm["tp"], "false_negative": pm["fn"],
            "false_positive": pm["fp"], "true_negative": pm["tn"],
            "recall": pm["recall"], "precision": pm["precision"], "fpr": pm["fpr"],
            "false_alerts_per_10k_benign": pm["false_alerts_per_10k_benign"],
            "n_actual_attack": int((y[mask] == 1).sum()),
            "n_actual_benign": int((y[mask] == 0).sum()),
        }

    all_mask = np.ones(len(y), dtype=bool)
    cats = test["attack_category"].values
    return {
        "threshold": float(threshold),
        "combined": _cm(all_mask),
        "light": _cm((y == 0) | (cats == "light")),
        "heavy": _cm((y == 0) | (cats == "heavy")),
        "note": ("Light/heavy slices pair that attack category against the FULL benign population, "
                 "so FPR is meaningful; a category slice has no benign rows of its own."),
    }


def run_operating_points() -> dict:
    bundle = joblib.load("models/stateless_lgbm_v1.pkl")
    model, features, calibrator = bundle["model"], bundle["features"], bundle["calibrator"]

    val_thr = load_split("val_thr").copy()
    val_thr["raw_score"] = model.predict_proba(val_thr[features])[:, 1]
    test = load_split("test").copy()
    test["raw_score"] = model.predict_proba(test[features])[:, 1]
    test["calibrated_score"] = calibrator.predict(test["raw_score"].values)

    curve = build_operating_point_curve(val_thr, test)

    # Largest unreachable FPR gap -- the structural finding that explains the old duplicate rows.
    gaps = []
    for a, b in zip(curve, curve[1:]):
        jump = b["val_thr_fpr"] - a["val_thr_fpr"]
        if jump > 0.01:
            gaps.append({"from_val_fpr": a["val_thr_fpr"], "to_val_fpr": b["val_thr_fpr"],
                         "unreachable_band_width": jump,
                         "from_threshold": a["threshold"], "to_threshold": b["threshold"]})

    report = {
        "model_version": bundle["model_version"],
        "n_val_thr_benign_rows": int((val_thr["label"] == 0).sum()),
        "n_distinct_val_thr_benign_scores": int(len(np.unique(val_thr.loc[val_thr["label"] == 0, "raw_score"]))),
        "quantization_note": (
            "Score quantization, not a coarse threshold grid, is why fixed target-FPR tables for "
            "this model contain duplicate rows: LightGBM on 14 coarse, largely integer-valued "
            "features produces heavily tied scores, so entire FPR bands are unreachable at any "
            "threshold. The unreachable_fpr_bands field below quantifies this directly."
        ),
        "achievable_operating_points": curve,
        "unreachable_fpr_bands": gaps,
        "confusion_matrix_at_headline_threshold": confusion_matrices(
            test, json.load(open("models/stateless_threshold_v1.json"))["threshold"]),
        "calibration": calibration_report(test),
    }

    out_dir = Path("results/metrics")
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "operating_points_v1.json", "w") as f:
        json.dump(report, f, indent=2)

    print(json.dumps({k: v for k, v in report.items()
                      if k not in ("achievable_operating_points", "calibration")}, indent=2))
    print(f"\n{len(curve)} achievable operating points; "
          f"Brier raw={report['calibration']['brier_raw']:.5f} "
          f"isotonic={report['calibration']['brier_isotonic_calibrated']:.5f}")
    return report


if __name__ == "__main__":
    run_operating_points()
