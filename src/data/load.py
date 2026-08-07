import os
import yaml
import numpy as np
import pandas as pd
from pathlib import Path

def get_schema_lock(schema_path: str = "configs/schema.lock.yaml") -> dict:
    with open(schema_path, "r") as f:
        return yaml.safe_load(f)

def load_raw(raw_dir: str = "data/raw", schema_path: str = "configs/schema.lock.yaml") -> pd.DataFrame:
    raw_path = Path(raw_dir)
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw directory not found: {raw_dir}")
        
    schema = get_schema_lock(schema_path)
    stateless_files = sorted(list(raw_path.rglob("stateless_features*.csv")))
    
    dfs = []
    for f in stateless_files:
        rel_path = str(f.relative_to(raw_path)).replace("\\", "/")
        df = pd.read_csv(f)
        
        # Derive label and category from directory path
        if "Attacks" in rel_path:
            cat = "heavy" if "heavy" in rel_path.lower() else "light"
            lbl = 1
        else:
            cat = "benign"
            lbl = 0
            
        df["label"] = np.int8(lbl)
        df["attack_category"] = pd.Categorical([cat] * len(df), categories=["benign", "light", "heavy"])
        
        # Parse timestamp to collection_day
        ts_parsed = pd.to_datetime(df["timestamp"], format="mixed")
        df["collection_day"] = ts_parsed.dt.date.astype(str)
        
        # Exclude single-class day 2020-11-25 per S5
        df = df[df["collection_day"] != "2020-11-25"].copy()
        if len(df) == 0:
            continue
            
        # Grouping session key (capture file + collection day segment)
        df["session_id"] = [f"{rel_path}::{day}" for day in df["collection_day"]]
        
        # Build 14 stateless features
        df["sl_fqdn_count"] = df["FQDN_count"].astype(np.float32)
        df["sl_subdomain_length"] = df["subdomain_length"].astype(np.float32)
        df["sl_upper"] = df["upper"].astype(np.float32)
        df["sl_lower"] = df["lower"].astype(np.float32)
        df["sl_numeric"] = df["numeric"].astype(np.float32)
        df["sl_entropy"] = df["entropy"].astype(np.float32)
        df["sl_special"] = df["special"].astype(np.float32)
        df["sl_labels"] = df["labels"].astype(np.float32)
        df["sl_labels_max"] = df["labels_max"].astype(np.float32)
        df["sl_labels_average"] = df["labels_average"].astype(np.float32)
        df["sl_len"] = df["len"].astype(np.float32)
        df["sl_subdomain_flag"] = df["subdomain"].astype(np.float32)
        
        # Derived safe features
        lw_len = df["longest_word"].astype(str).apply(lambda x: len(x) if x != "nan" else 0)
        df["sl_longest_word_len"] = lw_len.astype(np.float32)
        
        num_ratio = df["numeric"] / np.maximum(df["FQDN_count"].astype(np.float32), 1.0)
        df["sl_numeric_ratio"] = num_ratio.astype(np.float32)
        
        # Create unique unit_id
        df["unit_id"] = [f"{rel_path}_{i}" for i in range(len(df))]
        
        # Select canonical columns + stateless features
        cols = [
            "unit_id", "session_id", "collection_day", "label", "attack_category"
        ] + schema["stateless_features"]
        
        dfs.append(df[cols])
        
    full_df = pd.concat(dfs, ignore_index=True)
    
    # Assertions on load
    unique_labels = set(full_df["label"].unique())
    assert unique_labels == {0, 1}, f"Unexpected labels: {unique_labels}"
    
    unique_cats = set(full_df["attack_category"].unique())
    assert unique_cats == {"benign", "light", "heavy"}, f"Unexpected categories: {unique_cats}"
    
    # Sort deterministically
    full_df = full_df.sort_values(by=["collection_day", "session_id", "unit_id"]).reset_index(drop=True)
    return full_df

def load_split(split_name: str, splits_dir: str = "data/splits") -> pd.DataFrame:
    path = Path(splits_dir) / f"{split_name}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Split file not found: {path}")
    return pd.read_parquet(path)

if __name__ == "__main__":
    df = load_raw()
    print(f"Loaded raw dataset cleanly: {len(df)} rows across {df['collection_day'].nunique()} days.")
