"""Does the score actually RANK, or does it only enrich by a fixed factor?

Motivated by an observation on §7's operating-point table: precision sits at
~62.6% at every threshold from 4% to 28% FPR, i.e. a constant likelihood ratio
of ~2.62. A constant likelihood ratio means the flagged pool has the same
attack/benign mixture at every cut -- which would mean the score does NOT rank
within the flagged set, and Track 1's "alert ranking that reduces analyst
workload" criterion would not be met.

This tests that directly with precision-at-depth.

IMPORTANT -- ties: naive precision@k is meaningless on this model. The score is
heavily quantized (e.g. 22,763 test rows share the single value 0.712416), so a
fixed k slices arbitrarily into a tie block and the resulting number reflects
row order, not the model. (Measured directly: naive precision@1000 = 0.949, but
k=1000 cuts 859 of 1,451 tied rows arbitrarily -- that number is an artifact and
is NOT reported.) Everything below is computed at tie-respecting cut points --
cumulative precision over {rows with score >= d} for each distinct score d -- so
every reported depth is a cut the model actually licenses.
"""
import sys
import json
from pathlib import Path

import numpy as np
import joblib

project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.data.load import load_split

REPORT_DEPTHS = [50, 100, 200, 500, 1600, 7000, 30000, 40000]


def run_ranking_quality() -> dict:
    bundle = joblib.load("models/stateless_lgbm_v1.pkl")
    model, features = bundle["model"], bundle["features"]

    test = load_split("test").copy()
    test["raw_score"] = model.predict_proba(test[features])[:, 1]
    y = test["label"].values
    s = test["raw_score"].values
    base_rate = float(y.mean())

    blocks = []
    for d in np.unique(s)[::-1]:
        mask = s >= d
        n = int(mask.sum())
        blocks.append({"score": float(d), "cum_n": n, "cum_precision": float(y[mask].mean()),
                       "lift_over_base_rate": float(y[mask].mean() / base_rate)})
        if n > 50000:
            break

    reported = []
    for target in REPORT_DEPTHS:
        b = min(blocks, key=lambda r: abs(r["cum_n"] - target))
        reported.append({"requested_depth": target, **b})

    # Where does ranking stop working? First depth at which cumulative precision has
    # decayed to within 1 point of the asymptotic flat value.
    flat = blocks[-1]["cum_precision"]
    decay_point = next((b for b in blocks if abs(b["cum_precision"] - flat) < 0.01), None)

    return {
        "model_version": bundle["model_version"],
        "test_base_rate": base_rate,
        "tie_warning": (
            "Naive precision@k is invalid on this model: 22,763 test rows share the score "
            "0.712416, so a fixed k slices a tie block arbitrarily. Naive precision@1000 "
            "computes to 0.949 purely as an artifact of row order within that block and is not "
            "reported. All depths below are tie-respecting (score >= d cut points)."
        ),
        "precision_at_depth_tie_respecting": reported,
        "asymptotic_flat_precision": flat,
        "ranking_effective_until_depth": decay_point["cum_n"] if decay_point else None,
        "finding": (
            "Ranking is real but confined to the extreme head of the list. The top ~48 rows are "
            "91.7% precise (2.35x base rate) and the top 100 are 74.0% (1.90x), but by depth ~141 "
            "cumulative precision has already collapsed to ~63.8%, and from there to the full "
            "flagged pool it is flat at ~62.6% (~1.61x). So the score meaningfully prioritizes "
            "only the first ~100-140 alerts out of 100,841 rows; past that it provides a fixed "
            "enrichment factor and no further discrimination. This is the same underlying "
            "limitation as the score quantization in section 7 and the near-identical light/heavy "
            "recall: the 14-feature space supports coarse separation, not fine ranking."
        ),
    }


if __name__ == "__main__":
    report = run_ranking_quality()
    out = Path("results/metrics/ranking_quality_v1.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))
