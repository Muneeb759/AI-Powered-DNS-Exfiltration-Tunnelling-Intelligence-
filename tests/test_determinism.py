import pytest
import hashlib
from pathlib import Path
from src.data.build_splits import build_splits

def get_file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()

def test_split_determinism(tmp_path):
    # Run 1
    dir1 = tmp_path / "splits1"
    build_splits(splits_dir=str(dir1), seed=20260808)
    
    # Run 2
    dir2 = tmp_path / "splits2"
    build_splits(splits_dir=str(dir2), seed=20260808)
    
    for f_name in ["train.parquet", "val_cal.parquet", "val_thr.parquet", "test.parquet"]:
        h1 = get_file_hash(dir1 / f_name)
        h2 = get_file_hash(dir2 / f_name)
        assert h1 == h2, f"Determinism failure for {f_name}: hashes do not match"
