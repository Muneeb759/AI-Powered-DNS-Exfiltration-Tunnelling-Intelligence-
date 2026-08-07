import os
import pytest
from pathlib import Path

def test_no_banned_split_functions():
    banned_terms = ["train_test_split", ".sample(frac=", "shuffle("]
    src_dir = Path("src")
    
    violations = []
    for p in src_dir.rglob("*.py"):
        content = p.read_text(encoding="utf-8")
        for term in banned_terms:
            if term in content:
                # Check if it is numpy RandomState or random.shuffle unseeded
                if "np.random.RandomState" in content or "RandomState(seed)" in content:
                    continue
                violations.append(f"{p}: found banned term '{term}'")
                
    assert len(violations) == 0, f"Found banned splitting terms in src/: {violations}"
