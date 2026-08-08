"""Pooled per-capture stateful context for the Track 2 cascade's Stage 2.

docs/PHASE1_AUDIT.md B1(d) established there is no row-level join between the stateless
and stateful tables -- the only valid join key is the capture file itself (18 files total).
This module builds one pooled stateful feature vector per capture file (mean over that
file's stateful rows) rather than inventing a row-level or sub-file pairing that the data
doesn't support (docs/PHASE2_PLAN.md's ordering check confirmed no reliable positional
alignment exists either). It applies the same C1-C5 corrections docs/PHASE1_AUDIT.md
already specified for the stateful table but that were never implemented:
  - C2/C3: parse the string-encoded set/list columns into safe numeric counts
  - C4: reverse_dns -> boolean "known" flag, raw string dropped
  - C5: drop the 10 constant columns already flagged in configs/leakage_exclusions.yaml
  - rr_type: one-hot encoded (small fixed cardinality, non-identity-revealing)
"""
import ast
import sys
from pathlib import Path

import numpy as np
import pandas as pd

project_root = Path(__file__).resolve().parent.parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

SAFE_NUMERIC_STATEFUL = [
    "rr", "A_frequency", "PTR_frequency", "TXT_frequency", "AAAA_frequency",
    "rr_count", "rr_name_entropy", "rr_name_length", "distinct_ns", "ttl_mean", "ttl_variance",
]
RR_TYPE_CATEGORIES = ["A", "AAAA", "CNAME", "MX", "NS", "PTR", "SOA", "TXT", "NONE"]


def _parse_set_len(x) -> int:
    try:
        val = ast.literal_eval(x) if isinstance(x, str) else x
        return len(val) if val is not None else 0
    except (ValueError, SyntaxError):
        return 0


def _parse_rr_type(x) -> str:
    try:
        val = ast.literal_eval(x) if isinstance(x, str) else x
        if not val:
            return "NONE"
        return sorted(val)[0]  # single dominant type per row in this dataset
    except (ValueError, SyntaxError):
        return "NONE"


def build_stateful_context(raw_dir: str = "data/raw") -> pd.DataFrame:
    """Returns one row per capture_file (base trace, no day suffix) with a pooled
    (mean-aggregated) stateful context vector. ~18 rows for this dataset."""
    raw_path = Path(raw_dir)
    stateful_files = sorted(raw_path.rglob("stateful_features*.csv"))

    rows = []
    for f in stateful_files:
        rel_path = str(f.relative_to(raw_path)).replace("\\", "/")
        df = pd.read_csv(f)

        derived = pd.DataFrame(index=df.index)
        for col in SAFE_NUMERIC_STATEFUL:
            derived[f"sf_{col}"] = df[col].astype(np.float64)

        derived["sf_n_countries"] = df["unique_country"].apply(_parse_set_len).astype(np.float64)
        derived["sf_n_asns"] = df["unique_asn"].apply(_parse_set_len).astype(np.float64)
        derived["sf_n_distinct_domains"] = df["distinct_domains"].apply(_parse_set_len).astype(np.float64)
        derived["sf_ttl_count"] = df["unique_ttl"].apply(_parse_set_len).astype(np.float64)
        derived["sf_reverse_dns_known"] = (df["reverse_dns"] != "unknown").astype(np.float64)

        rr_type_cat = df["rr_type"].apply(_parse_rr_type)
        for cat in RR_TYPE_CATEGORIES:
            derived[f"sf_rr_type_{cat}"] = (rr_type_cat == cat).astype(np.float64)

        pooled = derived.mean(numeric_only=True)
        pooled["capture_file"] = rel_path
        pooled["stateful_row_count"] = len(df)
        rows.append(pooled)

    context_df = pd.DataFrame(rows).set_index("capture_file")
    return context_df


STATEFUL_CONTEXT_FEATURES = (
    [f"sf_{c}" for c in SAFE_NUMERIC_STATEFUL]
    + ["sf_n_countries", "sf_n_asns", "sf_n_distinct_domains", "sf_ttl_count", "sf_reverse_dns_known"]
    + [f"sf_rr_type_{c}" for c in RR_TYPE_CATEGORIES]
)

if __name__ == "__main__":
    ctx = build_stateful_context()
    print(ctx[STATEFUL_CONTEXT_FEATURES + ["stateful_row_count"]])
