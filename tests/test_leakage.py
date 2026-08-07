import pytest
from src.data.splits import LeakageGate
from src.data.load import load_split

def test_leakage_gate_on_split_partitions():
    gate = LeakageGate()
    for part in ["train", "val_cal", "val_thr", "test"]:
        df = load_split(part)
        gate.assert_clean(df)
