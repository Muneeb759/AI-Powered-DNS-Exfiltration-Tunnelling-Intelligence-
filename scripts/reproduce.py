"""Single command that reproduces the reported headline score (challenge brief,
deliverable #2: "a command that reproduces the reported score").

    python scripts/reproduce.py        # verify committed artifacts reproduce the report
    python scripts/reproduce.py --retrain   # retrain from raw data first, then verify

This does not merely run the pipeline -- it ASSERTS that the numbers produced match
the values published in report/TECHNICAL_REPORT.md, and exits non-zero if they do
not. A reproduction script that prints numbers without checking them cannot fail,
and a check that cannot fail is not a check (the same principle that retired the
v2 model, see TECHNICAL_REPORT.md section 9).
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import joblib
import numpy as np

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Published values in report/TECHNICAL_REPORT.md section 7 (v1, headline operating point).
HEADLINE_THRESHOLD = 0.713607050540105
EXPECTED = {
    "pr_auc": 0.6269,
    "roc_auc": 0.8097,
    "test_fpr": 0.0412,
    "recall_combined": 0.1078,
    "precision_combined": 0.626,
    "recall_light": 0.1064,
    "recall_heavy": 0.1080,
}
TOL = 0.0015  # absolute; generous enough for platform float variation, tight enough to catch drift


def _fail(msg: str) -> None:
    print(f"\n  FAIL: {msg}")
    sys.exit(1)


def ensure_inputs(retrain: bool) -> None:
    raw = project_root / "data" / "raw"
    if not raw.exists() or not any(raw.rglob("stateless_features*.csv")):
        _fail(
            "data/raw is missing. This repository does not redistribute the dataset.\n"
            "  Obtain CIC-Bell-DNS-EXF-2021 from the official UNB page and extract it to\n"
            "  data/raw/ so that data/raw/Benign/ and data/raw/Attack_*/ exist.\n"
            "  See deliverables/DATA_AND_MODEL_STATEMENT.md for provenance and checksums."
        )

    splits = project_root / "data" / "splits" / "test.parquet"
    if not splits.exists() or retrain:
        print("  building splits (data/splits/*.parquet) ...")
        r = subprocess.run([sys.executable, "-m", "src.data.build_splits"],
                           cwd=project_root, capture_output=True, text=True)
        if r.returncode != 0:
            _fail(f"split construction failed:\n{r.stdout[-2000:]}\n{r.stderr[-2000:]}")

    if retrain:
        print("  retraining v1 model (this overwrites models/stateless_lgbm_v1.pkl) ...")
        code = ("import sys; sys.path.insert(0,'.');"
                "from src.models.stateless_model import train_stateless_model;"
                "train_stateless_model(use_engineered=False)")
        r = subprocess.run([sys.executable, "-c", code],
                           cwd=project_root, capture_output=True, text=True)
        if r.returncode != 0:
            _fail(f"retraining failed:\n{r.stdout[-2000:]}\n{r.stderr[-2000:]}")


def score_and_check() -> None:
    from src.data.load import load_split
    from src.eval.metrics import point_metrics, ranking_metrics

    model_path = project_root / "models" / "stateless_lgbm_v1.pkl"
    if not model_path.exists():
        _fail("models/stateless_lgbm_v1.pkl missing. Re-run with --retrain.")

    bundle = joblib.load(model_path)
    if bundle["model_version"] != "stateless_lgbm_v1":
        _fail(f"expected stateless_lgbm_v1, found {bundle['model_version']}")

    # The threshold artifact the PRODUCT uses must be the one the REPORT documents. These
    # drifted apart once (the app ran at FPR=0.1%/0.229% recall while the report claimed
    # 4.12%/10.78%), so verify agreement rather than trusting either in isolation.
    thr_path = project_root / "models" / "stateless_threshold_v1.json"
    if not thr_path.exists():
        _fail("models/stateless_threshold_v1.json missing.")
    thr_record = json.load(open(thr_path))
    if abs(thr_record["threshold"] - HEADLINE_THRESHOLD) > 1e-12:
        _fail(
            f"threshold artifact disagrees with the reported headline operating point:\n"
            f"    models/stateless_threshold_v1.json -> {thr_record['threshold']}\n"
            f"    report/TECHNICAL_REPORT.md section 7 -> {HEADLINE_THRESHOLD}\n"
            "  The product would run at a different operating point than the report claims."
        )

    test = load_split("test")
    scores = bundle["model"].predict_proba(test[bundle["features"]])[:, 1]
    y = test["label"].values

    rank = ranking_metrics(y, scores)
    pm = point_metrics(y, scores, threshold=HEADLINE_THRESHOLD)

    def slice_recall(cat: str) -> float:
        mask = (y == 0) | (test["attack_category"].values == cat)
        return point_metrics(y[mask], scores[mask], threshold=HEADLINE_THRESHOLD)["recall"]

    actual = {
        "pr_auc": rank["pr_auc"],
        "roc_auc": rank["roc_auc"],
        "test_fpr": pm["fpr"],
        "recall_combined": pm["recall"],
        "precision_combined": pm["precision"],
        "recall_light": slice_recall("light"),
        "recall_heavy": slice_recall("heavy"),
    }

    print(f"\n  {'metric':<22} {'reported':>10} {'reproduced':>12}   status")
    print(f"  {'-'*22} {'-'*10} {'-'*12}   ------")
    failures = []
    for k, want in EXPECTED.items():
        got = actual[k]
        ok = abs(got - want) <= TOL
        print(f"  {k:<22} {want:>10.4f} {got:>12.4f}   {'ok' if ok else 'MISMATCH'}")
        if not ok:
            failures.append(f"{k}: reported {want:.4f}, reproduced {got:.4f} (tol {TOL})")

    print(f"\n  model_version : {bundle['model_version']}")
    print(f"  n_features    : {len(bundle['features'])} (base gated stateless only)")
    print(f"  threshold     : {HEADLINE_THRESHOLD}")
    print(f"  test rows     : {len(test):,}")

    if failures:
        _fail("reproduced numbers do not match report/TECHNICAL_REPORT.md section 7:\n    "
              + "\n    ".join(failures))
    print("\n  PASS: reproduced values match report/TECHNICAL_REPORT.md section 7.")


def main() -> None:
    ap = argparse.ArgumentParser(description="Reproduce the reported headline score.")
    ap.add_argument("--retrain", action="store_true",
                    help="rebuild splits and retrain from data/raw before verifying")
    args = ap.parse_args()

    print("Reproducing headline score: v1 stateless detector, Track 1 primary.")
    ensure_inputs(args.retrain)
    score_and_check()


if __name__ == "__main__":
    main()
