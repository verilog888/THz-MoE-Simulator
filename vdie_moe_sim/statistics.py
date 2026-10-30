"""Seed-level descriptive statistics and explicitly exploratory bootstrap means."""

import numpy as np

from .config import stable_digest
from .sweep_plan import StatisticsConfig


def describe(values, config: StatisticsConfig, *, key=""):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or not values.size or not np.isfinite(values).all():
        raise ValueError("statistics require a nonempty finite vector")
    n = len(values)
    row = {"n": n, "mean": float(np.mean(values)), "std": float(np.std(values, ddof=1)) if n > 1 else None,
           "min": float(np.min(values)), "max": float(np.max(values)), "ci_low": None, "ci_high": None,
           "ci_method": "not_estimated_single_seed" if n == 1 else "disabled",
           "confidence": config.confidence, "bootstrap_resamples": config.bootstrap_resamples}
    if n > 1 and config.bootstrap_resamples:
        entropy = [config.seed, int(stable_digest(key)[:16], 16)]
        rng = np.random.default_rng(np.random.SeedSequence(entropy))
        means = np.empty(config.bootstrap_resamples)
        # Bounded intermediate allocation, independent of chunk size and row order.
        for start in range(0, len(means), 256):
            count = min(256, len(means)-start)
            indices = rng.integers(0, n, size=(count, n))
            means[start:start+count] = values[indices].mean(axis=1)
        alpha = (1-config.confidence)/2
        row["ci_low"], row["ci_high"] = map(float, np.quantile(means, [alpha, 1-alpha], method="linear"))
        row["ci_method"] = "exploratory_percentile_bootstrap_mean"
    return row


def paired_observations(rows, series, reference="C"):
    """Pair within point AND routing seed; never divide two unrelated means."""
    if reference not in series:
        raise ValueError("reference series is missing")
    groups = {}
    for row in rows:
        key = (row["point_id"], row["routing_seed"])
        group = groups.setdefault(key, {})
        if row["series"] in group:
            raise ValueError("duplicate point/seed/series result")
        group[row["series"]] = row
    paired = []
    for (point, seed), group in sorted(groups.items()):
        if set(group) != set(series):
            raise ValueError("incomplete matched seed group")
        ref = group[reference]
        for name in series:
            row = group[name]
            if row["workload_sha256"] != ref["workload_sha256"] or row["controlled_config_sha256"] != ref["controlled_config_sha256"]:
                raise ValueError("paired results changed workload or controlled configuration")
            latency, base = row["layer_makespan_s"], ref["layer_makespan_s"]
            if not np.isfinite(latency) or not np.isfinite(base) or min(latency, base) <= 0:
                raise ValueError("paired latency must be positive and finite")
            paired.append({"point_id": point, "routing_seed": seed, "series": name, "reference": reference,
                           "speedup_vs_C": base/latency, "latency_difference_vs_C_s": latency-base,
                           "workload_sha256": row["workload_sha256"]})
    return paired
