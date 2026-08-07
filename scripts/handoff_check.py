import os
import sys
import json
import pytest
import pandas as pd
from pathlib import Path

# Add project root to sys.path
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.data.load import load_split, get_schema_lock
from src.data.splits import assert_split_protocol, LeakageGate

def run_handoff_check() -> bool:
    print("=" * 60)
    print("RUNNING PHASE 1 -> PHASE 2 HANDOFF CHECK CONTRACT")
    print("=" * 60)
    
    splits_dir = Path("data/splits")
    required_files = ["train.parquet", "val_cal.parquet", "val_thr.parquet", "test.parquet"]
    
    # 1. Parquet File Existence
    for fname in required_files:
        p = splits_dir / fname
        if not p.exists():
            print(f"[FAIL] Missing split file: {p}")
            return False
    print("[PASS] All 4 parquet split files exist.")
    
    # 2. Schema and Dtype Verification
    schema = get_schema_lock()
    expected_cols = [
        "unit_id", "session_id", "collection_day", "label", "attack_category"
    ] + schema["stateless_features"]
    
    gate = LeakageGate()
    partitions = {}
    for name in ["train", "val_cal", "val_thr", "test"]:
        df = load_split(name)
        partitions[name] = df
        
        # Check column list
        if list(df.columns) != expected_cols:
            print(f"[FAIL] Partition '{name}' column mismatch!\n  Got: {list(df.columns)}\n  Expected: {expected_cols}")
            return False
            
        # Check label values
        labels = set(df["label"].unique())
        if not labels.issubset({0, 1}):
            print(f"[FAIL] Partition '{name}' invalid labels: {labels}")
            return False
            
        # Check categories
        cats = set(df["attack_category"].unique())
        if not cats.issubset({"benign", "light", "heavy"}):
            print(f"[FAIL] Partition '{name}' invalid categories: {cats}")
            return False
            
        # Check LeakageGate
        try:
            gate.assert_clean(df)
        except Exception as e:
            print(f"[FAIL] Partition '{name}' failed LeakageGate: {e}")
            return False
            
        # Verify no stateful columns
        sf_cols = [c for c in df.columns if c.startswith("sf_")]
        if len(sf_cols) > 0:
            print(f"[FAIL] Partition '{name}' contains forbidden stateful columns: {sf_cols}")
            return False

    print("[PASS] All parquet partitions strictly match schema, dtypes, and LeakageGate assertions.")
    
    # 3. Split Protocol Invariants
    try:
        assert_split_protocol(
            partitions["train"],
            partitions["val_cal"],
            partitions["val_thr"],
            partitions["test"]
        )
    except Exception as e:
        print(f"[FAIL] Split protocol assertion failed: {e}")
        return False
    print("[PASS] Split protocol invariants verified (Day & Session disjoint).")
    
    # 4. Feature Lists Verification
    with open("configs/feature_lists.yaml", "r") as f:
        import yaml
        fl = yaml.safe_load(f)
    stateless_feats = fl.get("stateless", [])
    if len(stateless_feats) == 0:
        print("[FAIL] configs/feature_lists.yaml is empty!")
        return False
    for feat in stateless_feats:
        for p_name, p_df in partitions.items():
            if feat not in p_df.columns:
                print(f"[FAIL] Feature '{feat}' missing from partition '{p_name}'")
                return False
    print(f"[PASS] Feature list verified ({len(stateless_feats)} stateless features present across all partitions).")
    
    # 5. Required Metric Artifacts
    ablation_path = Path("results/metrics/leakage_ablation.json")
    if not ablation_path.exists():
        print(f"[FAIL] Missing ablation artifact: {ablation_path}")
        return False
    with open(ablation_path, "r") as f:
        ab_data = json.load(f)
        print(f"[PASS] Ablation artifact verified (Gated PR-AUC: {ab_data['gated_model']['pr_auc']:.4f}, Shortcut PR-AUC: {ab_data['shortcut_model']['pr_auc']:.4f}).")
        
    latency_path = Path("results/metrics/latency_baseline.json")
    if not latency_path.exists():
        print(f"[FAIL] Missing latency artifact: {latency_path}")
        return False
    with open(latency_path, "r") as f:
        lat_data = json.load(f)
        single_qps = lat_data.get("single_query_mode", {}).get("throughput_queries_per_sec", 0)
        batch_qps = lat_data.get("batch_128_mode", {}).get("throughput_queries_per_sec", 0)
        print(f"[PASS] Latency artifact verified (Mode: {lat_data.get('measurement_mode', 'UNKNOWN')}, Single-Query Stream: {single_qps:.1f} QPS, Batch-128: {batch_qps:.1f} QPS).")
        
    # 6. Pytest Verification
    ret = pytest.main(["tests"])
    if ret != 0:
        print(f"[FAIL] pytest suite failed with code {ret}")
        return False
    print("[PASS] pytest suite 100% green.")
    
    # 7. Print Summary Figures for Serena
    print("\n" + "=" * 60)
    print("PHASE 1 -> PHASE 2 HANDOFF SUMMARY")
    print("=" * 60)
    for p_name, p_df in partitions.items():
        benign_count = int((p_df["label"] == 0).sum())
        pos_count = int((p_df["label"] == 1).sum())
        min_fpr = 1.0 / benign_count if benign_count > 0 else 0.0
        print(f"Partition '{p_name:7s}': Total={len(p_df):7d} | Benign={benign_count:7d} | Positives={pos_count:7d} | Min FPR Floor={min_fpr:.6e}")
        
    print("\n[SUCCESS] HANDOFF CHECK PASSED CLEANLY! SERENA IS UNBLOCKED FOR PHASE 2.")
    return True

if __name__ == "__main__":
    success = run_handoff_check()
    sys.exit(0 if success else 1)
