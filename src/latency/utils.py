"""Shared latency-measurement helpers (brief requirement: "median and p95 processing
latency... declared hardware, software versions, batch size, and concurrency").

Previously src/latency/harness.py and src/latency/stage2_harness.py only reported a
mean over repeated trials (total wall time / n_trials), which cannot produce a median
or p95 -- both require the individual per-call latency distribution, not just its
average. This module standardizes recording that distribution and the environment
declaration the brief asks for.
"""
import os
import platform
import sys
from importlib.metadata import version, PackageNotFoundError

import numpy as np


def get_hardware_software_info(concurrency: int = 1) -> dict:
    def pkg_version(name: str) -> str:
        try:
            return version(name)
        except PackageNotFoundError:
            return "not installed"

    return {
        "os": platform.platform(),
        "processor": platform.processor(),
        "cpu_count_logical": os.cpu_count(),
        "python_version": sys.version.split()[0],
        "lightgbm_version": pkg_version("lightgbm"),
        "scikit_learn_version": pkg_version("scikit-learn"),
        "numpy_version": pkg_version("numpy"),
        "concurrency": concurrency,
        "concurrency_note": "single-threaded, sequential calls -- no concurrent/async execution measured",
    }


def summarize_latency_distribution(latencies_us: list) -> dict:
    """latencies_us: per-call latencies in microseconds, one entry per individual
    timed operation (not a total-wall-time average) -- required to compute percentiles."""
    arr = np.asarray(latencies_us, dtype=np.float64)
    return {
        "n_calls": int(len(arr)),
        "mean_us": round(float(arr.mean()), 3),
        "median_us": round(float(np.median(arr)), 3),
        "p95_us": round(float(np.percentile(arr, 95)), 3),
        "p99_us": round(float(np.percentile(arr, 99)), 3),
        "min_us": round(float(arr.min()), 3),
        "max_us": round(float(arr.max()), 3),
    }
