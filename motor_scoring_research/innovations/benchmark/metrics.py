"""Parent-level benchmark outcomes and preregistered cluster uncertainty."""
from __future__ import annotations

import math
import random
from collections import defaultdict


def mps(scores: tuple[int, int, int, int] | None) -> float | None:
    if scores is None:
        return None
    if len(scores) != 4 or any(type(x) is not int or x not in range(5) for x in scores):
        raise ValueError('Demo scores must be four integers in 0..4')
    return sum(scores)/16


def paired_effect(initial: float | None, after: float | None, reference: float | None) -> float | None:
    if initial is None or after is None or reference is None:
        return None
    if not all(math.isfinite(x) and 0 <= x <= 1 for x in (initial, after, reference)):
        raise ValueError('Invalid normalized demo MPS')
    return abs(initial-reference) - abs(after-reference)


def parent_mean(rows: list[dict], key: str) -> dict[str, float]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        value = row.get(key)
        if value is not None:
            if not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError('Nonfinite benchmark metric')
            grouped[row['parent_id']].append(float(value))
    return {parent: sum(values)/len(values) for parent, values in grouped.items()}


def cluster_interval(rows: list[dict], profiles: dict[str, str], key: str,
                     *, seed: int = 9022026, replicates: int = 2000) -> dict:
    """Resample parents, retaining all their slot descendants; never replicas."""
    if replicates != 2000:
        raise ValueError('Preregistered bootstrap uses exactly 2000 replicates')
    values = parent_mean(rows, key)
    groups = {p: sorted(parent for parent in profiles if profiles[parent] == p)
              for p in ('healthy_like', 'symptom_like')}
    if not all(groups.values()) or not values:
        return {'status': 'indeterminate', 'supporting_parent_clusters': len(values)}
    rng = random.Random(seed)
    draws = []
    for _ in range(replicates):
        sampled = [rng.choice(groups[p]) for p in ('healthy_like', 'symptom_like')
                   for _ in range(len(groups[p]))]
        available = [values[x] for x in sampled if x in values]
        if available:
            draws.append(sum(available)/len(available))
    if not draws:
        return {'status': 'indeterminate', 'supporting_parent_clusters': len(values)}
    draws.sort()
    point = sum(values.values())/len(values)
    n = len(draws)
    return {'status': 'estimated', 'point': point,
            'percentile_95': [draws[int(.025*(n-1))], draws[int(.975*(n-1))]],
            'one_sided_97_5_lower': draws[int(.025*(n-1))],
            'supporting_parent_clusters': len(values), 'replicates': replicates}


def success_checks(false_rejection: dict, detection: dict,
                   trust_enrichment: dict, recap_vs_random: dict) -> dict:
    def point(metric):
        return metric.get('point') if metric.get('status') == 'estimated' else None
    a, b, c, d = map(point, (false_rejection, detection, trust_enrichment, recap_vs_random))
    checks = {
        'technical_false_rejection_at_most_0_05': None if a is None else a <= .05,
        'observable_fault_detection_at_least_0_95': None if b is None else b >= .95,
        'TRUST_repeat_flip_enrichment_at_least_0_10_and_lower_above_zero':
            None if c is None else c >= .10 and trust_enrichment['one_sided_97_5_lower'] > 0,
        'RECAP_vs_random_MPS_deviation_positive_and_lower_above_zero':
            None if d is None else d > 0 and recap_vs_random['one_sided_97_5_lower'] > 0,
    }
    return {'checks': checks, 'overall': 'pass' if all(v is True for v in checks.values()) else
            'indeterminate' if any(v is None for v in checks.values()) else 'fail'}
