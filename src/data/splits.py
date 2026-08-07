import yaml
import numpy as np
import pandas as pd
from typing import List, Tuple, Set

class LeakageGate:
    def __init__(self, exclusions_path: str = "configs/leakage_exclusions.yaml"):
        with open(exclusions_path, "r") as f:
            self.config = yaml.safe_load(f)
        self.allowlist_prefixes = self.config.get("allowlist_prefixes", ["sl_", "sf_"])
        self.explicit_exclusions = self.config.get("explicit_exclusions", {})

    def is_feature(self, col_name: str) -> bool:
        if col_name in self.explicit_exclusions:
            return False
        return any(col_name.startswith(prefix) for prefix in self.allowlist_prefixes)

    def assert_clean(self, df: pd.DataFrame) -> None:
        unclassified = []
        for col in df.columns:
            if not self.is_feature(col) and col not in self.explicit_exclusions:
                unclassified.append(col)
        if unclassified:
            raise ValueError(f"LeakageGate violation! Unclassified columns present: {unclassified}")

    @property
    def feature_columns(self) -> List[str]:
        with open("configs/feature_lists.yaml", "r") as f:
            fl = yaml.safe_load(f)
        return fl.get("stateless", [])

    def audit_table(self) -> pd.DataFrame:
        rows = []
        for col, reason in self.explicit_exclusions.items():
            rows.append({"column": col, "status": "EXCLUDED", "reason": reason})
        for feat in self.feature_columns:
            rows.append({"column": feat, "status": "ALLOWLISTED", "reason": "Safe stateless feature"})
        return pd.DataFrame(rows)

def assert_group_disjoint(df_list: List[pd.DataFrame], group_col: str) -> None:
    group_sets = [set(df[group_col].dropna().unique()) for df in df_list]
    for i in range(len(group_sets)):
        for j in range(i + 1, len(group_sets)):
            overlap = group_sets[i].intersection(group_sets[j])
            assert len(overlap) == 0, f"Group overlap found in '{group_col}' between partition {i} and {j}: {overlap}"

def assert_split_protocol(train: pd.DataFrame, val_cal: pd.DataFrame, val_thr: pd.DataFrame, test: pd.DataFrame) -> None:
    # 1. Group / capture file disjointness
    assert_group_disjoint([train, val_cal, val_thr, test], "session_id")
    
    # 2. Dual-class and dual-category composition checks
    for name, partition in [("train", train), ("val_cal", val_cal), ("val_thr", val_thr), ("test", test)]:
        n_ben = (partition["label"] == 0).sum()
        n_light = (partition["attack_category"] == "light").sum()
        n_heavy = (partition["attack_category"] == "heavy").sum()
        assert n_ben > 0, f"Partition '{name}' must contain benign samples, got 0"
        assert n_light > 0, f"Partition '{name}' must contain light attack samples, got 0"
        assert n_heavy > 0, f"Partition '{name}' must contain heavy attack samples, got 0"
        
    # 3. Decision unit uniqueness
    all_units = list(train["unit_id"]) + list(val_cal["unit_id"]) + list(val_thr["unit_id"]) + list(test["unit_id"])
    assert len(all_units) == len(set(all_units)), "Duplicate unit_id found across partitions!"
    
    print("ALL CAPTURE-FILE GROUPED SPLIT INVARIANT ASSERTIONS PASSED SUCCESSFULLY!")

def capture_file_grouped_split(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # Extract base trace file name from session_id
    df = df.copy()
    all_files = set(df["session_id"].apply(lambda x: x.split("::")[0]).unique())
    
    # Trace assignments guaranteed to give dual-class & dual-category to all partitions
    test_files = {f for f in all_files if any(t in f for t in ["light_text", "heavy_audio", "benign_heavy_1"])}
    cal_files = {f for f in all_files if any(t in f for t in ["light_compressed", "heavy_compressed", "benign_heavy_2"])}
    thr_files = {f for f in all_files if any(t in f for t in ["light_audio", "heavy_image", "benign_heavy_3"])}
    train_files = all_files - test_files - cal_files - thr_files
    
    df["base_trace"] = df["session_id"].apply(lambda x: x.split("::")[0])
    
    train_df = df[df["base_trace"].isin(train_files)].drop(columns=["base_trace"]).reset_index(drop=True)
    val_cal_df = df[df["base_trace"].isin(cal_files)].drop(columns=["base_trace"]).reset_index(drop=True)
    val_thr_df = df[df["base_trace"].isin(thr_files)].drop(columns=["base_trace"]).reset_index(drop=True)
    test_df = df[df["base_trace"].isin(test_files)].drop(columns=["base_trace"]).reset_index(drop=True)
    
    return train_df, val_cal_df, val_thr_df, test_df
