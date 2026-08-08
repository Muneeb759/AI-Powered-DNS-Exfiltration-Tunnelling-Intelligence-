"""unit_id must resolve to an exact line of the source CSV.

unit_id is pipeline-assigned, not dataset-native: it is
`{capture_file_relative_path}_{row_index}` where the index is the row's position in
the SOURCE CSV (src/data/load.py). Every prediction in the submitted results bundle
is keyed on it, so if that mapping ever breaks, the deliverable silently starts
pointing at the wrong source rows.

The mapping is fragile in one specific way: it holds only while unit_id is assigned
BEFORE any row filtering. It was previously assigned after the 2020-11-25 exclusion,
which re-indexes the kept rows -- harmless in this snapshot purely because every
excluded row sits contiguously at the end of the one affected file (heavy_exe), so
pre- and post-filter indices coincided. Interleaved exclusions would have shifted
every subsequent id onto the wrong line with nothing to catch it.

This test is that catch. It fails if anyone reorders the pipeline, changes the
exclusion policy, or re-exports the CSVs with different row ordering.
"""
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data.load import load_split

N_SAMPLES = 20
SEED = 20260808

# (feature column in the split, source column in the raw CSV)
CHECKED_FEATURES = [
    ("sl_len", "len"),
    ("sl_fqdn_count", "FQDN_count"),
    ("sl_entropy", "entropy"),
    ("sl_subdomain_length", "subdomain_length"),
    ("sl_upper", "upper"),
    ("sl_lower", "lower"),
    ("sl_numeric", "numeric"),
    ("sl_special", "special"),
    ("sl_labels", "labels"),
]


def _resolve(unit_id: str, raw_dir: Path) -> pd.Series:
    """unit_id -> the exact source CSV row it names."""
    rel_path, idx = unit_id.rsplit("_", 1)
    csv_path = raw_dir / rel_path
    assert csv_path.exists(), f"unit_id names a capture file that does not exist: {csv_path}"
    source = pd.read_csv(csv_path)
    idx = int(idx)
    assert 0 <= idx < len(source), (
        f"unit_id index {idx} is out of range for {rel_path} ({len(source)} rows) -- "
        "unit_id no longer indexes the source CSV"
    )
    return source.iloc[idx]


def test_unit_id_resolves_to_source_csv_line():
    raw_dir = Path("data/raw")
    if not raw_dir.exists():
        pytest.skip("data/raw not present -- traceability check needs the raw snapshot")

    test = load_split("test")
    rng = np.random.default_rng(SEED)
    sampled = test.iloc[rng.choice(len(test), N_SAMPLES, replace=False)]

    for _, row in sampled.iterrows():
        source = _resolve(row["unit_id"], raw_dir)
        for feat_col, raw_col in CHECKED_FEATURES:
            expected, actual = float(source[raw_col]), float(row[feat_col])
            assert math.isclose(expected, actual, rel_tol=1e-5, abs_tol=1e-5), (
                f"unit_id {row['unit_id']} does not resolve to its source line: "
                f"{feat_col}={actual} but source CSV {raw_col}={expected}. "
                "unit_id no longer maps to the original CSV row -- check that it is still "
                "assigned before any row filtering in src/data/load.py."
            )


def test_unit_ids_are_unique_across_all_partitions():
    """A duplicate unit_id would make bundle predictions ambiguous."""
    ids = pd.concat([load_split(p)["unit_id"] for p in ("train", "val_cal", "val_thr", "test")])
    assert ids.is_unique, f"unit_id is not unique: {int(len(ids) - ids.nunique())} duplicates"
