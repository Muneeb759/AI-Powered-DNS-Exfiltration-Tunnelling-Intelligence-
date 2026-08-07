import pytest
import pandas as pd
from src.data.load import load_raw, get_schema_lock

def test_schema_lock_resolution():
    schema = get_schema_lock()
    assert "canonical_columns" in schema
    assert "label_encoding" in schema
    assert schema["label_encoding"][0] == "benign"
    assert schema["label_encoding"][1] == "exfiltration"
    assert len(schema["stateless_features"]) == 14

def test_raw_loader_assertions():
    df = load_raw()
    assert set(df["label"].unique()) == {0, 1}
    assert set(df["attack_category"].unique()) == {"benign", "light", "heavy"}
    
    # Check dtypes
    schema = get_schema_lock()
    for feat in schema["stateless_features"]:
        assert df[feat].dtype == "float32", f"Feature {feat} is not float32"
