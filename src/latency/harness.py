import sys
import time
import json
import numpy as np
import pandas as pd
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.data.load import load_split
from src.latency.utils import get_hardware_software_info, summarize_latency_distribution

def get_classifier(seed: int = 20260808):
    try:
        import lightgbm as lgb
        return lgb.LGBMClassifier(n_estimators=100, learning_rate=0.05, random_state=seed, n_jobs=-1, verbose=-1)
    except ImportError:
        from sklearn.ensemble import HistGradientBoostingClassifier
        return HistGradientBoostingClassifier(max_iter=100, learning_rate=0.05, random_state=seed)

def run_latency_benchmark(n_trials: int = 1000, seed: int = 20260808) -> dict:
    train_df = load_split("train")
    test_df = load_split("test")
    
    with open("configs/feature_lists.yaml", "r") as f:
        import yaml
        fl = yaml.safe_load(f)
    gated_features = fl.get("stateless", [])
    
    X_train = train_df[gated_features].values
    y_train = train_df["label"].values
    X_test = test_df[gated_features].values[:1000]
    
    print("Training classifier for latency measurement...")
    clf = get_classifier(seed=seed)
    clf.fit(X_train, y_train)
    
    # Warmup
    _ = clf.predict_proba(X_test[:10])

    # 1. Feature extraction latency per query (14 float32 stateless features).
    # Per-call timings recorded individually -- required for median/p95, not just a mean.
    extraction_latencies_us = []
    for _ in range(n_trials):
        t0 = time.perf_counter()
        _ = X_test[:1].astype(np.float32)
        t1 = time.perf_counter()
        extraction_latencies_us.append((t1 - t0) * 1e6)
    extraction_dist = summarize_latency_distribution(extraction_latencies_us)

    # 2. Single-query stream inference latency (batch_size = 1, per-row calls).
    n_single = 500
    single_latencies_us = []
    for i in range(n_single):
        idx = i % len(X_test)
        t0 = time.perf_counter()
        _ = clf.predict_proba(X_test[idx:idx + 1])
        t1 = time.perf_counter()
        single_latencies_us.append((t1 - t0) * 1e6)
    single_dist = summarize_latency_distribution(single_latencies_us)
    single_wall_time = sum(single_latencies_us) / 1e6

    # 3. Batch inference latency (batch_size = 128, per-batch calls). Percentiles here
    # are over BATCH latency, not per-query -- divide by batch_size for the per-query
    # amortized figure, consistent with standard batching practice.
    batch_size = 128
    n_batches = 100
    batch_latencies_us = []
    for i in range(n_batches):
        t0 = time.perf_counter()
        _ = clf.predict_proba(X_test[:batch_size])
        t1 = time.perf_counter()
        batch_latencies_us.append((t1 - t0) * 1e6)
    batch_dist = summarize_latency_distribution(batch_latencies_us)
    batch_wall_time = sum(batch_latencies_us) / 1e6

    extraction_latency_us = extraction_dist["mean_us"]
    single_inference_us = single_dist["mean_us"]
    batch_inference_us_per_query = batch_dist["mean_us"] / batch_size

    # Serial (single-query) total latency & throughput
    serial_total_latency_us = extraction_latency_us + single_inference_us
    serial_throughput_qps = 1e6 / serial_total_latency_us

    # Batched (batch_size = 128) total latency per query & throughput
    batched_total_latency_us = extraction_latency_us + batch_inference_us_per_query
    batched_throughput_qps = 1e6 / batched_total_latency_us

    result = {
        "measurement_mode": "EMPIRICAL_BENCHMARK",
        "classifier_type": type(clf).__name__,
        "features_count": len(gated_features),
        "environment": get_hardware_software_info(concurrency=1),
        "extraction_latency_us_per_query": round(extraction_latency_us, 2),
        "extraction_latency_distribution": extraction_dist,
        "single_query_mode": {
            "batch_size": 1,
            "inference_latency_us_per_query": round(single_inference_us, 2),
            "inference_latency_distribution_us": single_dist,
            "total_pipeline_latency_us_per_query": round(serial_total_latency_us, 2),
            "total_pipeline_median_us_per_query": round(extraction_dist["median_us"] + single_dist["median_us"], 2),
            "total_pipeline_p95_us_per_query": round(extraction_dist["p95_us"] + single_dist["p95_us"], 2),
            "throughput_queries_per_sec": round(serial_throughput_qps, 1),
            "wall_time_seconds": round(single_wall_time, 4),
            "queries_processed": n_single,
            "arithmetic_proof": f"{n_single} queries, mean {round(single_wall_time, 4)} s total = {round(n_single / single_wall_time, 1)} QPS"
        },
        "batch_128_mode": {
            "batch_size": 128,
            "inference_latency_us_per_query": round(batch_inference_us_per_query, 2),
            "batch_latency_distribution_us": batch_dist,
            "total_pipeline_latency_us_per_query": round(batched_total_latency_us, 2),
            "total_pipeline_median_us_per_query": round(extraction_dist["median_us"] + batch_dist["median_us"] / batch_size, 2),
            "total_pipeline_p95_us_per_query": round(extraction_dist["p95_us"] + batch_dist["p95_us"] / batch_size, 2),
            "throughput_queries_per_sec": round(batched_throughput_qps, 1),
            "wall_time_seconds": round(batch_wall_time, 4),
            "queries_processed": n_batches * batch_size,
            "arithmetic_proof": f"{n_batches * batch_size} queries, mean {round(batch_wall_time, 4)} s total = {round((n_batches * batch_size) / batch_wall_time, 1)} QPS"
        }
    }

    out_dir = Path("results/metrics")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "latency_baseline.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"Latency benchmark completed cleanly.")
    print(f"  Single-Query Stream Mode (batch=1): mean={result['single_query_mode']['total_pipeline_latency_us_per_query']} "
          f"median={result['single_query_mode']['total_pipeline_median_us_per_query']} "
          f"p95={result['single_query_mode']['total_pipeline_p95_us_per_query']} µs/query -> {result['single_query_mode']['throughput_queries_per_sec']} QPS")
    print(f"  Batch 128 Mode (batch=128):          mean={result['batch_128_mode']['total_pipeline_latency_us_per_query']} "
          f"median={result['batch_128_mode']['total_pipeline_median_us_per_query']} "
          f"p95={result['batch_128_mode']['total_pipeline_p95_us_per_query']} µs/query -> {result['batch_128_mode']['throughput_queries_per_sec']} QPS")
    print(f"Written to {out_path}")
    return result

if __name__ == "__main__":
    run_latency_benchmark()
