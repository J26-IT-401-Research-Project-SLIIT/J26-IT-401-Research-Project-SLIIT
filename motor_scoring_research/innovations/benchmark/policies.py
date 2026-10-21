"""Preregistered decisions; no potential-repeat bank enters these functions."""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Mapping

from ..contracts import (AssessmentMode, AssessmentResult, DOMAIN_ORDER,
                         DURATION_SECONDS, Domain, QualityStatus)
from ..recap import RecapConfig, recommend_reassessment
from .scenarios import derived_seed

POLICIES = ('A_no_retest', 'B_repeat_all', 'C_random_eligible',
            'D_quality_only', 'E_trust_instability_only', 'F_full_trust_recap')


@dataclass(frozen=True)
class Decision:
    policy: str
    tasks: tuple[Domain, ...]
    additional_seconds: int
    reason: str
    feasible: bool = True


def _decision(policy: str, tasks: tuple[Domain, ...], reason: str,
              budget: int, feasible: bool = True) -> Decision:
    seconds = sum(DURATION_SECONDS[d] for d in tasks)
    if feasible and seconds > budget:
        raise ValueError('Policy exceeded recording-time budget')
    return Decision(policy, tasks, seconds, reason, feasible)


def decide(policy: str, assessment: AssessmentResult, trust: dict | None,
           remediation: Mapping[Domain, str], budget: int, *, parent_id: str,
           slot: str, random_seed: int, recap_config: RecapConfig) -> Decision:
    if policy not in POLICIES or budget not in (10, 20, 180, 210):
        raise ValueError('Unregistered policy or budget')
    if policy == POLICIES[0]:
        return _decision(policy, (), 'no_retest', budget)
    if policy == POLICIES[1]:
        return _decision(policy, DOMAIN_ORDER if budget == 210 else (),
                         'repeat_all' if budget == 210 else 'infeasible_at_budget',
                         budget, budget == 210)
    if tuple(q.domain for q in assessment.quality) != DOMAIN_ORDER:
        raise ValueError('Incomplete initial quality report')
    invalid = [q.domain for q in assessment.quality if q.status is not QualityStatus.VALID]
    corrected = [d for d in DOMAIN_ORDER if d in invalid and remediation.get(d) == 'corrected']
    blocked = [d for d in invalid if d not in corrected]
    if policy == POLICIES[5]:
        advice, sidecar = recommend_reassessment(assessment, trust, config=recap_config,
                    remediation=remediation, recording_time_budget_seconds=budget)
        tasks = tuple(advice.ranked_domains[:1])
        return _decision(policy, tasks, sidecar['stopping_reason'], budget)
    if blocked:
        return _decision(policy, (), 'clarify_or_repair_acquisition', budget)
    if policy == POLICIES[3]:
        choices = [d for d in corrected if DURATION_SECONDS[d] <= budget]
        return _decision(policy, tuple(choices[:1]),
                         'confirmed_corrected_fault' if choices else 'no_eligible_fault_or_budget_block', budget)
    if policy == POLICIES[2]:
        choices = corrected or ([d for d in DOMAIN_ORDER if DURATION_SECONDS[d] <= budget]
                                if not invalid else [])
        choices = [d for d in choices if DURATION_SECONDS[d] <= budget]
        if not choices:
            return _decision(policy, (), 'no_eligible_task_or_budget_block', budget)
        rng = random.Random(derived_seed(random_seed, parent_id, slot, 'random_policy', Domain.TS))
        return _decision(policy, (rng.choice(choices),), 'seeded_uniform_eligible', budget)
    # E has no acquisition-repair pathway: only complete valid demo sessions.
    if invalid or assessment.mode is not AssessmentMode.SYNTHETIC_DEMO:
        return _decision(policy, (), 'invalid_or_real_device', budget)
    _, sidecar = recommend_reassessment(assessment, trust, config=recap_config,
                    recording_time_budget_seconds=budget)
    if sidecar['decision_status'] == 'insufficient_evidence':
        return _decision(policy, (), 'missing_or_invalid_trust_evidence', budget)
    ranked = sidecar['modeled_instability_ranking']
    choices = [Domain(row['domain']) for row in ranked if DURATION_SECONDS[Domain(row['domain'])] <= budget]
    return _decision(policy, tuple(choices[:1]),
                     'trust_instability_priority' if choices else 'no_observed_instability_or_budget_block', budget)


def reveal_selected(initial: tuple, bank: Mapping[Domain, tuple], decision: Decision) -> tuple:
    """Evaluator-only function; a policy never receives `bank`."""
    if not decision.feasible:
        return initial
    output = list(initial)
    for d in decision.tasks:
        if d not in bank or len(bank[d]) != 4:
            raise ValueError('Potential-repeat bank lacks selected complete session')
        index = DOMAIN_ORDER.index(d)
        output[index] = bank[d][index]
    return tuple(output)
