"""Transparent, label-first simulation; profiles are assumptions, not clinical facts."""
import numpy as np
import pandas as pd
from .settings import FEATURES, LABELS, SEED

# Purely illustrative profiles, not estimated five-disease distributions.
PROFILES = {"PD": [0.78, 0.74, 0.74], "ET": [0.76, 0.32, 0.68],
            "CA": [0.50, 0.75, 0.40], "AP": [0.68, 0.65, 0.38], "DLB": [0.38, 0.48, 0.62]}

def generate(n=1000, seed=SEED, proportions=None, noise=0.85, prefix="SYN"):
    if n < 10 or noise <= 0: raise ValueError("At least 10 records and positive noise are required")
    proportions = np.asarray(proportions if proportions is not None else [0.2]*5, dtype=float)
    if len(proportions) != 5 or np.any(proportions < 0) or not np.isclose(proportions.sum(), 1):
        raise ValueError("Provide five nonnegative class proportions summing to 1")
    rng = np.random.default_rng(seed)
    counts = np.floor(n*proportions).astype(int)
    for i in np.argsort(-(n*proportions-counts))[:n-counts.sum()]: counts[i] += 1
    rows = []
    for label, count in zip(LABELS, counts):
        center = np.asarray(PROFILES[label]); latent = np.log(center/(1-center))
        covariance = noise**2 * np.array([[1, .25, .30], [.25, 1, .20], [.30, .20, 1]])
        values = rng.multivariate_normal(latent, covariance, int(count))
        scores = np.clip(1/(1+np.exp(-values)) + rng.normal(0, .025, values.shape), .001, .999)
        for score in scores: rows.append(dict(zip(FEATURES, score), label=label))
    frame = pd.DataFrame(rows).iloc[rng.permutation(n)].reset_index(drop=True)
    frame.insert(0, "synthetic_id", [f"{prefix}-{i+1:04d}" for i in range(n)])
    frame[FEATURES] = frame[FEATURES].round(6)
    frame["provenance"] = "synthetic_assumption_based"
    return frame
