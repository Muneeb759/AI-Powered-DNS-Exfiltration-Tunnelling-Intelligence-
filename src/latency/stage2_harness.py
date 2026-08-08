"""Stage 2 (stateful escalation) latency measurement -- completes the Track 2 brief's
"measurements of feature-extraction and inference cost" requirement, which
src/latency/harness.py only answered for Stage 1.

Stage 2's cost has two genuinely different components, measured separately because
they amortize differently:
  1. One-time stateful context build (src/features/stateful_context.py): parses all
     18 capture files' stateful CSVs once. Paid once per enrichment refresh, not per
     query or even per session -- the pooled table is reused for every escalation
     decision until the underlying data changes.
  2. Marginal per-escalated-session cost: dict lookup into the already-built pooled
     context + the majority-vote decision over 3 features (src/models/cascade.py's
     _stage2_vote). This is the cost actually paid at escalation time, on top of
     Stage 1's per-query cost.

Reporting both, rather than only the marginal cost, is deliberate: a design that
looks "nearly free" per escalation would be misleading if the enrichment build cost
were hidden.
"""
import sys
import time
import json
from pathlib import Path

import numpy as np

project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.features.stateful_context import build_stateful_context, STATEFUL_CONTEXT_FEATURES
from src.models.cascade import (
    _token_from_capture_file, _build_token_indexed_context, _stage2_vote,
)
from src.data.load import load_split
from src.features.stateless_engineered import build_engineered_features
from src.latency.utils import get_hardware_software_info, summarize_latency_distribution

N_TRIALS_BUILD = 3
N_TRIALS_LOOKUP_VOTE = 2000
SEED = 20260808


def run_stage2_latency_benchmark() -> dict:
    # 1. One-time context build cost (averaged over a few repeats -- disk cache warms
    # up after the first run, so we report both cold and warm). n=3 is too small for a
    # meaningful percentile, so this reports min/mean/max explicitly rather than a
    # median/p95 that would be a false precision claim over 3 samples.
    build_times = []
    for i in range(N_TRIALS_BUILD):
        t0 = time.perf_counter()
        ctx = build_stateful_context()
        t1 = time.perf_counter()
        build_times.append(t1 - t0)
    context_by_token = ctx.reset_index()
    context_by_token["token"] = context_by_token["capture_file"].apply(_token_from_capture_file)
    context_by_token = context_by_token.set_index("token")

    # Discriminative features fixed from the actual cascade run (train-set only) --
    # reuse the same 3 features so this measures the real decision path, not a stand-in.
    train = build_engineered_features(load_split("train"))
    train["base_trace"] = train["session_id"].apply(lambda s: _token_from_capture_file(s.split("::")[0]))
    train_tokens_by_label = {
        "benign": sorted(train.loc[train["label"] == 0, "base_trace"].unique()),
        "attack": sorted(train.loc[train["label"] == 1, "base_trace"].unique()),
    }
    benign_rows = context_by_token.loc[train_tokens_by_label["benign"]]
    attack_rows = context_by_token.loc[train_tokens_by_label["attack"]]
    gaps = []
    for feat in STATEFUL_CONTEXT_FEATURES:
        b_mean, a_mean = benign_rows[feat].mean(), attack_rows[feat].mean()
        gaps.append({"feature": feat, "gap": abs(a_mean - b_mean),
                     "attack_side": "higher" if a_mean > b_mean else "lower",
                     "benign_mean": float(b_mean), "attack_mean": float(a_mean)})
    gaps.sort(key=lambda d: -d["gap"])
    discriminative_features = gaps[:3]

    tokens = list(context_by_token.index)
    rng = np.random.default_rng(SEED)

    # Warmup
    for _ in range(10):
        tok = tokens[rng.integers(0, len(tokens))]
        _ = _stage2_vote(context_by_token.loc[tok], discriminative_features)

    # 2. Marginal per-escalated-session cost: dict lookup + vote, timed in isolation,
    # per-call (not a total-wall-time average) so median/p95 are real distribution
    # statistics, not derived from a single aggregate number.
    marginal_latencies_us = []
    for _ in range(N_TRIALS_LOOKUP_VOTE):
        tok = tokens[rng.integers(0, len(tokens))]
        t0 = time.perf_counter()
        session_context = context_by_token.loc[tok]  # lookup
        _ = _stage2_vote(session_context, discriminative_features)  # vote
        t1 = time.perf_counter()
        marginal_latencies_us.append((t1 - t0) * 1e6)
    marginal_dist = summarize_latency_distribution(marginal_latencies_us)
    marginal_latency_us = marginal_dist["mean_us"]
    marginal_wall_time = sum(marginal_latencies_us) / 1e6

    # Cross-reference against Stage 1's own latency numbers, if available, to build
    # the one-stage vs two-stage comparison the brief asks for.
    stage1_path = Path("results/metrics/latency_baseline.json")
    stage1 = json.load(open(stage1_path)) if stage1_path.exists() else None

    result = {
        "measurement_mode": "EMPIRICAL_BENCHMARK",
        "environment": get_hardware_software_info(concurrency=1),
        "one_time_context_build": {
            "description": (
                "Parses all 18 capture files' stateful CSVs into the pooled per-capture "
                "context table (src/features/stateful_context.py). Paid once per "
                "enrichment refresh, not per query or per session. n=3 repeats is too small "
                "for a meaningful percentile -- reporting min/mean/max, not median/p95, to "
                "avoid a false-precision claim over so few samples."
            ),
            "n_capture_files": 18,
            "wall_time_seconds_per_run": [round(t, 3) for t in build_times],
            "wall_time_seconds_mean": round(float(np.mean(build_times)), 3),
            "wall_time_seconds_min": round(float(np.min(build_times)), 3),
            "wall_time_seconds_max": round(float(np.max(build_times)), 3),
        },
        "marginal_per_escalated_session": {
            "description": (
                "Dict lookup into the already-built pooled context + majority-vote "
                "decision over the top-3 discriminative stateful features "
                "(src/models/cascade.py:_stage2_vote). This is the cost actually paid "
                "at escalation time, on top of Stage 1's per-query cost."
            ),
            "n_trials": N_TRIALS_LOOKUP_VOTE,
            "latency_us_per_session": round(marginal_latency_us, 4),
            "latency_distribution_us": marginal_dist,
            "wall_time_seconds": round(marginal_wall_time, 4),
        },
        "discriminative_features_used": discriminative_features,
    }

    if stage1 is not None:
        stage1_single_us = stage1["single_query_mode"]["total_pipeline_latency_us_per_query"]
        stage1_single_median_us = stage1["single_query_mode"].get("total_pipeline_median_us_per_query")
        stage1_single_p95_us = stage1["single_query_mode"].get("total_pipeline_p95_us_per_query")
        result["one_stage_vs_two_stage_comparison"] = {
            "stage1_only_mean_us_per_query": stage1_single_us,
            "stage1_only_median_us_per_query": stage1_single_median_us,
            "stage1_only_p95_us_per_query": stage1_single_p95_us,
            "stage2_marginal_mean_us_per_escalated_session": round(marginal_latency_us, 4),
            "stage2_marginal_median_us_per_escalated_session": marginal_dist["median_us"],
            "stage2_marginal_p95_us_per_escalated_session": marginal_dist["p95_us"],
            "note": (
                "Stage 2's marginal cost is paid once per SESSION (18 sessions total in this "
                "dataset), not once per query -- effectively negligible per-query overhead even "
                "for the 38.9% of test rows that belong to an escalated session (see "
                "results/metrics/cascade_report.json:pct_test_rows_escalated_to_stage2). The "
                f"one-time context build ({round(float(np.mean(build_times)), 1)}s for 18 files) "
                "is the real cost driver, and it is a batch/offline operation, not part of the "
                "online query path."
            ),
        }

    out_dir = Path("results/metrics")
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "stage2_latency_report.json", "w") as f:
        json.dump(result, f, indent=2)

    print(json.dumps(result, indent=2))
    return result


if __name__ == "__main__":
    run_stage2_latency_benchmark()
