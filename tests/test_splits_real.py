import pytest
from src.data.load import load_split
from src.data.splits import assert_split_protocol, assert_group_disjoint

def test_split_invariants_real_partitions():
    train = load_split("train")
    val_cal = load_split("val_cal")
    val_thr = load_split("val_thr")
    test = load_split("test")
    
    # Assert main protocol
    assert_split_protocol(train, val_cal, val_thr, test)
    
    # Assert session disjointness
    assert_group_disjoint([train, val_cal, test], "session_id")
    assert_group_disjoint([train, val_thr, test], "session_id")
    assert_group_disjoint([val_cal, val_thr], "session_id")
    
    # Assert no sf_* columns exist
    for df in [train, val_cal, val_thr, test]:
        sf_cols = [c for c in df.columns if c.startswith("sf_")]
        assert len(sf_cols) == 0, f"Found stateful sf_* columns in partition: {sf_cols}"
