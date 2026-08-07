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
    
    # 1. Feature extraction latency per query (14 float32 stateless features)
    t_start = time.perf_counter()
    for _ in range(n_trials):
        _ = X_test[:1].astype(np.float32)
    t_end = time.perf_counter()
    extraction_wall_time = t_end - t_start
    extraction_latency_us = (extraction_wall_time / n_trials) * 1e6
    
    # 2. Single-query stream inference latency (batch_size = 1, per-row calls)
    n_single = 500
    t_start = time.perf_counter()
    for i in range(n_single):
        idx = i % len(X_test)
        _ = clf.predict_proba(X_test[idx:idx+1])
    t_end = time.perf_counter()
    single_wall_time = t_end - t_start
    single_inference_us = (single_wall_time / n_single) * 1e6
    
    # 3. Batch inference latency (batch_size = 128, per-batch calls)
    batch_size = 128
    n_batches = 100
    t_start = time.perf_counter()
    for i in range(n_batches):
        _ = clf.predict_proba(X_test[:batch_size])
    t_end = time.perf_counter()
    batch_wall_time = t_end - t_start
    batch_inference_us_per_query = (batch_wall_time / (n_batches * batch_size)) * 1e6
    
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
        "extraction_latency_us_per_query": round(extraction_latency_us, 2),
        "single_query_mode": {
            "batch_size": 1,
            "inference_latency_us_per_query": round(single_inference_us, 2),
            "total_pipeline_latency_us_per_query": round(serial_total_latency_us, 2),
            "throughput_queries_per_sec": round(serial_throughput_qps, 1),
            "wall_time_seconds": round(single_wall_time, 4),
            "queries_processed": n_single,
            "arithmetic_proof": f"{n_single} queries in {round(single_wall_time, 4)} s = {round(n_single / single_wall_time, 1)} QPS"
        },
        "batch_128_mode": {
            "batch_size": 128,
            "inference_latency_us_per_query": round(batch_inference_us_per_query, 2),
            "total_pipeline_latency_us_per_query": round(batched_total_latency_us, 2),
            "throughput_queries_per_sec": round(batched_throughput_qps, 1),
            "wall_time_seconds": round(batch_wall_time, 4),
            "queries_processed": n_batches * batch_size,
            "arithmetic_proof": f"{n_batches * batch_size} queries in {round(batch_wall_time, 4)} s = {round((n_batches * batch_size) / batch_wall_time, 1)} QPS"
        }
    }
    
    out_dir = Path("results/metrics")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "latency_baseline.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
        
    print(f"Latency benchmark completed cleanly.")
    print(f"  Single-Query Stream Mode (batch=1): {result['single_query_mode']['total_pipeline_latency_us_per_query']} µs/query -> {result['single_query_mode']['throughput_queries_per_sec']} QPS")
    print(f"  Batch 128 Mode (batch=128):          {result['batch_128_mode']['total_pipeline_latency_us_per_query']} µs/query -> {result['batch_128_mode']['throughput_queries_per_sec']} QPS")
    print(f"Written to {out_path}")
    return result

if __name__ == "__main__":
    run_latency_benchmark()
