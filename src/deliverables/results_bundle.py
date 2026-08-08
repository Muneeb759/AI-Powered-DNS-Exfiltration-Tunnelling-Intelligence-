"""Builds the Results Bundle deliverable (challenge brief, required deliverable #4:
"machine-readable predictions for the organizer test IDs, scoring output, threshold
selection record, and model/configuration version").

Scripted, not handcrafted, so it regenerates deterministically from the saved model
artifacts and can be re-run after any model change.

FIX APPLIED HERE: the per-model prediction CSVs written by the individual training
scripts carried only `session_id`, which has just 3 distinct values across the whole
test partition (it identifies a capture+day, not a row). Predictions keyed that way
cannot be joined back to the rows they describe, so they did not satisfy the brief's
"predictions for the organizer test IDs" requirement. Every predictions file in this
bundle is keyed on `unit_id` -- the decision-unit primary key declared in
configs/schema.lock.yaml -- with session_id retained as a grouping column.

Note on "organizer test IDs": Sofstica confirmed on the hackathon Discord (2026-08-08)
that no separate organizer snapshot or split manifest exists beyond the public dataset,
so the IDs of record are this project's own `unit_id`s over the capture-file-grouped
test partition (docs/SESSION_CONSTRUCTION.md). This is stated in the manifest rather
than left implicit.
"""
import sys
import json
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import joblib

project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.data.load import load_split
from src.eval.metrics import point_metrics, ranking_metrics
from src.eval.window_alerting import build_windows, score_windows_at_t, window_point_metrics
from src.models.rule_baseline import fit_rule_score, score_rule, RULE_FEATURES
from src.latency.utils import get_hardware_software_info

BUNDLE_DIR = Path("results/bundle")

# Operating points of record, as reported in report/TECHNICAL_REPORT.md.
HEADLINE_THRESHOLD = 0.713607050540105      # v1 row-level, ~4.12% test FPR (report section 7)
POST_CLIFF_THRESHOLD = 0.7124161589841883   # v1 row-level, ~17.95% test FPR (report section 7)
WINDOW_N = 200
WINDOW_T = 0.7142572903774336
WINDOW_KN = 0.10


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_csv(df: pd.DataFrame, name: str) -> Path:
    path = BUNDLE_DIR / name
    df.to_csv(path, index=False)
    return path


def build_row_level_predictions(test: pd.DataFrame) -> pd.DataFrame:
    """v1 stateless model -- the headline detector. Keyed on unit_id."""
    bundle = joblib.load("models/stateless_lgbm_v1.pkl")
    model, features, calibrator = bundle["model"], bundle["features"], bundle["calibrator"]

    raw = model.predict_proba(test[features])[:, 1]
    cal = calibrator.predict(raw)
    return pd.DataFrame({
        "unit_id": test["unit_id"].values,
        "session_id": test["session_id"].values,
        "collection_day": test["collection_day"].values,
        "raw_score": raw,
        "calibrated_score": cal,
        "prediction_at_headline_threshold": (raw >= HEADLINE_THRESHOLD).astype(int),
        "prediction_at_post_cliff_threshold": (raw >= POST_CLIFF_THRESHOLD).astype(int),
        "model_version": bundle["model_version"],
    })


def build_baseline_predictions(train: pd.DataFrame, test: pd.DataFrame) -> pd.DataFrame:
    """Mandatory non-ML rule baseline. Normalization + sign fit on train only."""
    stats = fit_rule_score(train)
    threshold = json.load(open("results/metrics/rule_baseline_report.json"))["threshold"]
    scores = score_rule(test, stats)
    return pd.DataFrame({
        "unit_id": test["unit_id"].values,
        "session_id": test["session_id"].values,
        "rule_score": scores,
        "prediction": (scores >= threshold).astype(int),
        "model_version": "rule_baseline_v1",
    }), stats, threshold


def build_cascade_predictions(row_preds: pd.DataFrame) -> pd.DataFrame:
    """Track 2 cascade row-level verdicts, re-keyed on unit_id from the saved report's
    session verdicts (single source of truth -- no re-derivation of the escalation logic)."""
    report = json.load(open("results/metrics/cascade_report.json"))
    verdicts = report["session_verdicts"]

    def token(session_id: str) -> str:
        from src.models.cascade import _token_from_capture_file
        return _token_from_capture_file(session_id.split("::")[0])

    toks = row_preds["session_id"].map(token)
    stage1 = (row_preds["raw_score"].values >= report["stage1_threshold"]).astype(int)

    escalated, verdict_label, cascade_pred = [], [], []
    for t, s1 in zip(toks, stage1):
        v = verdicts.get(t, {})
        esc = bool(v.get("escalated", False))
        if not esc:
            label, pred = "not_escalated", int(s1)
        elif not v.get("stateful_context_available", True):
            label, pred = "escalated_context_unavailable_fallback_stage1", int(s1)
        elif v.get("confirmed"):
            label, pred = "escalated_confirmed_alert", 1
        else:
            label, pred = "escalated_unconfirmed_human_review", int(s1)
        escalated.append(esc); verdict_label.append(label); cascade_pred.append(pred)

    return pd.DataFrame({
        "unit_id": row_preds["unit_id"].values,
        "session_id": row_preds["session_id"].values,
        "raw_score": row_preds["raw_score"].values,
        "stage1_prediction": stage1,
        "escalated_to_stage2": escalated,
        "session_verdict": verdict_label,
        "cascade_prediction": cascade_pred,
        "model_version": f"cascade_over_{report['stage1_model_version']}",
    })


def build_window_predictions(test_scored: pd.DataFrame) -> pd.DataFrame:
    """Window-level alerting (reported experiment, not the headline unit)."""
    df = test_scored.rename(columns={"raw_score": "raw_score"}).copy()
    windows = build_windows(df, WINDOW_N)
    scored = score_windows_at_t(windows, WINDOW_T)
    return pd.DataFrame({
        "window_id": [f"W{i:05d}" for i in range(len(scored))],
        "session_id": scored["session_id"].values,
        "window_start_unit_id": scored["window_start_unit_id"].values,
        "window_size": scored["window_size"].values,
        "n_rows_exceeding_t": scored["n_rows_exceeding"].values,
        "fraction_exceeding": scored["fraction_exceeding"].values,
        "prediction": (scored["fraction_exceeding"].values >= WINDOW_KN).astype(int),
        "label": scored["label"].values,
        "config_version": f"window_N{WINDOW_N}_t{WINDOW_T:.6f}_kn{WINDOW_KN}",
    })


def build_scoring_output(test: pd.DataFrame, row_preds, base_preds, cascade_preds, window_preds) -> dict:
    y = test["label"].values

    def _slice(scores, threshold, category=None):
        if category:
            mask = (y == 0) | (test["attack_category"].values == category)
            return point_metrics(y[mask], scores[mask], threshold=threshold)
        return point_metrics(y, scores, threshold=threshold)

    raw = row_preds["raw_score"].values
    rank = ranking_metrics(y, raw)

    out = {
        "primary_system": {
            "name": "v1 stateless row-level detector",
            "model_version": row_preds["model_version"].iloc[0],
            "decision_unit": "row (individual DNS query)",
            "headline_operating_point": {
                "threshold": HEADLINE_THRESHOLD,
                "combined": _slice(raw, HEADLINE_THRESHOLD),
                "light": _slice(raw, HEADLINE_THRESHOLD, "light"),
                "heavy": _slice(raw, HEADLINE_THRESHOLD, "heavy"),
            },
            "post_cliff_operating_point": {
                "threshold": POST_CLIFF_THRESHOLD,
                "note": "higher-recall alternative; no operating point exists between these two "
                        "(15-point unreachable FPR band from score quantization -- see report section 7)",
                "combined": _slice(raw, POST_CLIFF_THRESHOLD),
                "light": _slice(raw, POST_CLIFF_THRESHOLD, "light"),
                "heavy": _slice(raw, POST_CLIFF_THRESHOLD, "heavy"),
            },
            "ranking_metrics": {"pr_auc_headline": rank["pr_auc"], "roc_auc_supplementary": rank["roc_auc"]},
        },
        "mandatory_non_ml_baseline": {
            "name": "signed z(sl_len) + z(sl_entropy) rule",
            "model_version": "rule_baseline_v1",
            "metrics": json.load(open("results/metrics/rule_baseline_report.json")),
        },
        "supporting_track2_cascade": {
            "model_version": cascade_preds["model_version"].iloc[0],
            "decision_unit": "row-level Stage 1, session-level Stage 2 confirmation",
            "metrics": {k: v for k, v in json.load(open("results/metrics/cascade_report.json")).items()
                        if k in ("one_stage_stage1_only", "two_stage_cascade", "light_only", "heavy_only",
                                  "escalation_margins", "pct_test_rows_escalated_to_stage2")},
        },
        "reported_experiment_window_alerting": {
            "config_version": window_preds["config_version"].iloc[0],
            "decision_unit": f"non-overlapping {WINDOW_N}-row window within one session_id",
            "metrics_on_pure_captures": window_point_metrics(
                pd.DataFrame({"fraction_exceeding": window_preds["fraction_exceeding"],
                              "label": window_preds["label"],
                              "window_size": window_preds["window_size"]}), WINDOW_KN),
            "mixed_traffic_stress_test": json.load(open("results/metrics/window_mixed_stream_stress_test.json"))["result_by_k_over_N"],
            "note": "NOT adopted as the alerting unit -- fails under diluted/interleaved attack "
                    "traffic. See report section 8 and section 9.",
        },
    }
    return out


def build_threshold_selection_record(rule_stats, rule_threshold) -> dict:
    op = json.load(open("results/metrics/operating_points_v1.json"))
    return {
        "principle": "Every threshold is selected on a validation partition only. test is scored "
                     "once, at the already-chosen operating point, and never used for selection.",
        "v1_row_level_headline": {
            "threshold": HEADLINE_THRESHOLD,
            "selected_on": "val_thr",
            "selection_rule": "highest-recall achievable operating point below the unreachable "
                              "FPR band (val_thr FPR 4.40%); thresholds enumerated from distinct "
                              "val_thr benign score values, not a fixed target-FPR grid",
            "val_thr_fpr": 0.04397848250999831,
            "test_fpr_achieved": 0.04117,
            "artifact": "results/metrics/operating_points_v1.json",
        },
        "v1_row_level_post_cliff": {
            "threshold": POST_CLIFF_THRESHOLD,
            "selected_on": "val_thr",
            "selection_rule": "first achievable point above the 15-point unreachable FPR band",
            "val_thr_fpr": 0.1941784487128936,
            "test_fpr_achieved": 0.17948,
        },
        "rule_baseline": {
            "threshold": rule_threshold,
            "selected_on": "val_thr",
            "selection_rule": "FPR=0.1% target via pick_threshold_at_fpr",
            "normalization_and_sign_fit_on": "train only",
            "fitted_stats": rule_stats,
        },
        "cascade_escalation_delta": {
            "delta": json.load(open("results/metrics/cascade_report.json"))["chosen_escalation_delta"],
            "selected_on": "val_thr (3 sessions -- explicitly small-N)",
            "selection_rule": "smallest delta that escalates the positive session(s) without "
                              "escalating every session",
            "suspicious_score_cut": 0.4,
        },
        "window_alerting": {
            "N": WINDOW_N, "t": WINDOW_T, "k_over_N": WINDOW_KN,
            "selected_on": "val_thr",
            "selection_rule": "grid search over N x t x k/N on val_thr; chosen for val->test "
                              "stability, not peak val recall",
            "artifact": "results/metrics/window_alerting_report.json",
        },
        "unreachable_fpr_bands": op["unreachable_fpr_bands"],
    }


def run_build_bundle() -> dict:
    BUNDLE_DIR.mkdir(parents=True, exist_ok=True)

    train = load_split("train")
    test = load_split("test")

    row_preds = build_row_level_predictions(test)
    base_preds, rule_stats, rule_threshold = build_baseline_predictions(train, test)
    cascade_preds = build_cascade_predictions(row_preds)

    test_scored = test.copy()
    test_scored["raw_score"] = row_preds["raw_score"].values
    window_preds = build_window_predictions(test_scored)

    files = {
        "predictions_primary_row_level.csv": _write_csv(row_preds, "predictions_primary_row_level.csv"),
        "predictions_baseline_rule.csv": _write_csv(base_preds, "predictions_baseline_rule.csv"),
        "predictions_cascade_row_level.csv": _write_csv(cascade_preds, "predictions_cascade_row_level.csv"),
        "predictions_window_level.csv": _write_csv(window_preds, "predictions_window_level.csv"),
    }

    scoring = build_scoring_output(test, row_preds, base_preds, cascade_preds, window_preds)
    with open(BUNDLE_DIR / "scoring_output.json", "w") as f:
        json.dump(scoring, f, indent=2)
    files["scoring_output.json"] = BUNDLE_DIR / "scoring_output.json"

    thresholds = build_threshold_selection_record(rule_stats, rule_threshold)
    with open(BUNDLE_DIR / "threshold_selection_record.json", "w") as f:
        json.dump(thresholds, f, indent=2)
    files["threshold_selection_record.json"] = BUNDLE_DIR / "threshold_selection_record.json"

    manifest = {
        "bundle_name": "CIC-Bell-DNS-EXF-2021 detection results bundle",
        "team": "Muneeb + Serena",
        "declared_primary_track": "Track 1 -- Detection & Alert Prioritization",
        "supporting_tracks": ["Track 2 cascade (interpretability layer)", "Track 3 analyst case view"],
        "test_id_column": "unit_id",
        "test_id_note": (
            "unit_id is pipeline-assigned, not dataset-native. CIC-Bell-DNS-EXF-2021 CSVs contain "
            "no key column, and Sofstica confirmed on the hackathon Discord (2026-08-08) that no "
            "organizer split manifest exists, so these are the IDs of record for this submission. "
            "Construction: {capture_file_relative_path}_{row_index}, where the index is the row's "
            "position in the source CSV. Assignment happens before any row filtering, so "
            "line-traceability holds by construction; tests/test_unit_id_traceability.py verifies "
            "the mapping. IDs are deterministic across reruns but not stable across changes to the "
            "exclusion policy or a re-export with different row ordering."
        ),
        "n_test_rows": int(len(test)),
        "n_test_windows": int(len(window_preds)),
        "model_and_config_versions": {
            "primary_detector": row_preds["model_version"].iloc[0],
            "baseline": "rule_baseline_v1",
            "cascade": cascade_preds["model_version"].iloc[0],
            "window_alerting": window_preds["config_version"].iloc[0],
            "seed": 20260808,
        },
        "rejected_not_submitted": {
            "stateless_lgbm_v2": (
                "Session-repeat feature model (PR-AUC 0.99999) -- REJECTED, not part of this "
                "bundle. Failed a mixed-session stress test (PR-AUC collapsed to 0.082). "
                "Artifacts retained in results/metrics/*_v2_rejected.json as evidence for "
                "report section 9. Not a submitted result."
            )
        },
        "environment": get_hardware_software_info(concurrency=1),
        "dataset_provenance": "see results/metrics/provenance.json (per-file SHA-256) and "
                              "deliverables/DATA_AND_MODEL_STATEMENT.md",
        "file_checksums_sha256": {name: _sha256(path) for name, path in files.items()},
    }
    with open(BUNDLE_DIR / "MANIFEST.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(json.dumps({k: v for k, v in manifest.items() if k != "environment"}, indent=2))
    print(f"\nBundle written to {BUNDLE_DIR}/ ({len(files) + 1} files)")
    return manifest


if __name__ == "__main__":
    run_build_bundle()
