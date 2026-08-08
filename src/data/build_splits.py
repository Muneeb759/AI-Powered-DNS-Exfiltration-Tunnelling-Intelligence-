import os
import sys
import json
import pandas as pd
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.data.load import load_raw
from src.data.splits import (
    capture_file_grouped_split,
    assert_split_protocol,
    report_day_overlap,
    LeakageGate
)

def build_splits(raw_dir: str = "data/raw", splits_dir: str = "data/splits", seed: int = 20260808) -> dict:
    splits_path = Path(splits_dir)
    splits_path.mkdir(parents=True, exist_ok=True)
    
    print("Loading raw dataset...")
    full_df = load_raw(raw_dir=raw_dir)
    
    # Capture-file grouped split across traces
    train_df, val_cal_df, val_thr_df, test_df = capture_file_grouped_split(full_df)
    
    # Assert split protocol invariants
    assert_split_protocol(train_df, val_cal_df, val_thr_df, test_df)
    
    # Verify LeakageGate on X feature columns
    gate = LeakageGate()
    for df_name, df_part in [("train", train_df), ("val_cal", val_cal_df), ("val_thr", val_thr_df), ("test", test_df)]:
        gate.assert_clean(df_part)
        
    # Save Parquet files
    train_df.to_parquet(splits_path / "train.parquet", index=False)
    val_cal_df.to_parquet(splits_path / "val_cal.parquet", index=False)
    val_thr_df.to_parquet(splits_path / "val_thr.parquet", index=False)
    test_df.to_parquet(splits_path / "test.parquet", index=False)
    
    # Summary metrics
    def partition_stats(df_part: pd.DataFrame) -> dict:
        benign_cnt = int((df_part["label"] == 0).sum())
        return {
            "total_rows": len(df_part),
            "collection_days": sorted(list(df_part["collection_day"].unique())),
            "capture_files": sorted(list(df_part["session_id"].unique())),
            "benign_rows": benign_cnt,
            "positive_rows": int((df_part["label"] == 1).sum()),
            "light_positive_rows": int((df_part["attack_category"] == "light").sum()),
            "heavy_positive_rows": int((df_part["attack_category"] == "heavy").sum()),
            "positive_rate": float((df_part["label"] == 1).mean()),
            "min_fpr_floor": float(1.0 / benign_cnt) if benign_cnt > 0 else 0.0
        }
        
    summary = {
        "split_type": "capture_file_grouped_split",
        "split_status": "PROVISIONAL",
        "release_type": "public_unb_release",
        "seed": seed,
        "disjointness_guarantee": "session_id (capture_file) only -- collection_day is NOT guaranteed disjoint, see day_overlap below",
        "day_overlap": report_day_overlap(train_df, val_cal_df, val_thr_df, test_df),
        "partitions": {
            "train": partition_stats(train_df),
            "val_cal": partition_stats(val_cal_df),
            "val_thr": partition_stats(val_thr_df),
            "test": partition_stats(test_df)
        }
    }
    
    out_dir = Path("results/metrics")
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_path = out_dir / "split_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
        
    # Export leakage audit table
    audit_table = gate.audit_table()
    audit_table.to_csv(out_dir / "leakage_audit.csv", index=False)
    
    print(f"Splits successfully built and saved to {splits_dir}.")
    print(f"Split summary written to {summary_path}")
    return summary

if __name__ == "__main__":
    build_splits()
