"""Colab-CPU-only, locked innovation benchmark runner.

No benchmark work occurs at import. A run writes only into a fresh directory.
Audit claims are persistent and precede any audit case access.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import tempfile
import time
from dataclasses import asdict
from pathlib import Path

from ..contracts import (AssessmentMode, DOMAIN_ORDER, Domain, RAW_COLUMNS,
                         SensorSeries, TapSeries, QualityStatus)
from ..assessment import assemble_assessment
from ..frozen_adapter import load_verified_models
from ..integration.feature_bridge import CompleteSessionFeatureBridge, load_reference_engine
from ..quality import validate_session
from ..recap import load_recap_config
from ..trust import analyze_trust, load_config
from .locks import (claim_audit, eligible_ids, read_json, sha256,
                    verify_design, verify_frozen_models)
from .metrics import cluster_interval, mps, paired_effect, success_checks
from .policies import POLICIES, decide
from .scenarios import SLOTS, Observation, scenario_pair


def _colab_cpu() -> None:
    if not Path('/content').is_dir() or not os.environ.get('COLAB_RELEASE_TAG'):
        raise RuntimeError('Actual benchmark execution requires Google Colab CPU')
    if Path('/dev/nvidia0').exists() or os.environ.get('COLAB_TPU_ADDR'):
        raise RuntimeError('Select Colab CPU runtime')


def _write_json_new(path: Path, value: dict) -> None:
    with path.open('x', encoding='utf-8') as stream:
        json.dump(value, stream, sort_keys=True, indent=2, allow_nan=False)


def _archive_preflight(dataset: Path, parents: dict[str, dict], spec: dict) -> dict:
    """Read control metadata only; never enumerate original validation/test raw."""
    complete = read_json(dataset / 'COMPLETE.json')
    integrity = read_json(dataset / 'integrity_report.json')
    if complete.get('integrity_passed') is not True or complete.get('sessions') != 240:
        raise ValueError('V3 dataset is not complete')
    if integrity.get('status') != 'passed' or integrity.get('session_count') != 240:
        raise ValueError('V3 integrity report does not positively attest PASS')
    manifest = read_json(dataset / 'dataset_manifest.json')
    if manifest.get('sessions') != 240 or manifest.get('stress_only') is not False:
        raise ValueError('Unexpected V3 dataset type')
    if sha256(dataset / 'splits.csv') != manifest['splits_sha256']:
        raise ValueError('Archived V3 split checksum changed')
    with (dataset / 'splits.csv').open(newline='', encoding='utf-8') as stream:
        archived = {r['session_id']: r for r in csv.DictReader(stream)}
    if len(archived) != 240 or any(archived[sid]['split'] != 'train' for sid in parents):
        raise ValueError('Archived V3 splits differ from frozen training parents')
    return manifest


def _archive_parent(dataset: Path, sid: str, split: str, parents: dict[str, dict],
                    records: dict[str, dict], config: dict):
    if sid not in parents or parents[sid]['innovation_split'] != split:
        raise PermissionError('A parent cannot cross innovation splits')
    record = records.get(sid)
    if not record or record.get('split') != 'train' or record.get('session_id') != sid:
        raise ValueError('Archived parent manifest missing or mismatched')
    paths = {d: dataset / 'cases' / sid / f'{d.value}_{config["durations_s"][d.value]}s.csv'
             for d in DOMAIN_ORDER}
    for d, path in paths.items():
        if sha256(path) != record['raw_sha256'][d.value]:
            raise ValueError(f'Parent raw checksum mismatch: {sid}/{d.value}')
    validated = validate_session(paths)
    if any(v.input is None for v in validated):
        raise ValueError(f'Original V3 parent has invalid acquisition: {sid}')
    return tuple(v.input for v in validated), record


def _write_observation(folder: Path, obs: Observation):
    paths = {}
    for d, raw in zip(DOMAIN_ORDER, obs.raw):
        path = folder / f'{d.value}.csv'
        with path.open('x', newline='', encoding='utf-8') as stream:
            writer = csv.writer(stream)
            writer.writerow(RAW_COLUMNS[d])
            if isinstance(raw, TapSeries):
                writer.writerows((t,) for t in raw.tap_time_s)
            elif isinstance(raw, SensorSeries):
                writer.writerows((t, *xyz) for t, xyz in zip(raw.time_s, raw.xyz))
            else:
                raise ValueError('Unknown raw input type')
        paths[d] = path
    contexts = dict(zip(DOMAIN_ORDER, obs.contexts))
    return validate_session(paths, contexts)


def _evaluate_observation(obs: Observation, folder: Path, bridge, predictor, trust_config, seed):
    folder.mkdir(parents=True, exist_ok=False)
    validations = _write_observation(folder, obs)
    assessment, sidecar = analyze_trust(validations, AssessmentMode.SYNTHETIC_DEMO,
                                        bridge, predictor, config=trust_config, seed=seed)
    return assessment, sidecar


def _evaluate_repeat(obs: Observation, folder: Path, bridge, predictor):
    folder.mkdir(parents=True, exist_ok=False)
    validations = _write_observation(folder, obs)
    if any(v.input is None or v.quality.status is not QualityStatus.VALID for v in validations):
        return assemble_assessment(AssessmentMode.SYNTHETIC_DEMO, validations)
    features = tuple(bridge(tuple(v.input for v in validations)))
    predictions = tuple(predictor.predict(f) for f in features)
    return assemble_assessment(AssessmentMode.SYNTHETIC_DEMO, validations, features, predictions)


def _reference_mps(engine, raw: tuple) -> float:
    ts, bs, rs, gi = raw
    result = engine.assess_motor(ts.xyz, bs.tap_time_s, rs.xyz, gi.xyz, fs=100, mode='demo')
    if set(result['scores']) != {d.value for d in DOMAIN_ORDER}:
        raise ValueError('Original rule engine score contract changed')
    reference = mps(tuple(int(result['scores'][d.value]) for d in DOMAIN_ORDER))
    if reference != result['mps']:
        raise ValueError('Rule reference MPS arithmetic mismatch')
    return reference


def _score_tuple(assessment) -> tuple[int, ...] | None:
    return None if assessment.predictions is None else tuple(p.score for p in assessment.predictions)


def _sensitivity_flag(trust: dict, domain: Domain) -> bool | None:
    families = trust.get('domain_diagnostics', {}).get(domain.value)
    if not isinstance(families, dict):
        return None
    valid = sum(row['valid'] for row in families.values())
    if valid == 0:
        return None
    return any(row['score_changes'] > 0 for row in families.values())


def _summarize(rows: list[dict], profiles: dict[str, str], spec: dict) -> dict:
    parent_prospective = {}
    for sid in profiles:
        own = [r for r in rows if r['parent_id'] == sid]
        fn = sum(r.get('prospective_flagged_n', 0) for r in own)
        un = sum(r.get('prospective_unflagged_n', 0) for r in own)
        if fn and un:
            parent_prospective[sid] = (sum(r.get('prospective_flagged_flip_sum', 0) for r in own)/fn
                                       - sum(r.get('prospective_unflagged_flip_sum', 0) for r in own)/un)
    for row in rows:
        if row['parent_id'] in parent_prospective:
            row['independent_repeat_flip_enrichment'] = parent_prospective[row['parent_id']]
    # Each row is an exposure; cluster_interval first averages within parent.
    keys = ('technical_false_rejection', 'observable_fault_detected',
            'independent_repeat_flip_enrichment', 'F_minus_C_error_180')
    intervals = {key: cluster_interval(rows, profiles, key,
                 seed=spec['bootstrap_seed']) for key in keys}
    policy_table = {}
    observed = [r for r in rows if r.get('status') != 'excluded']
    for budget in spec['budgets_additional_recording_seconds']:
        for policy in POLICIES:
            key = f'{policy}@{budget}'
            outcomes = [(r, r['policy_outcomes'][key]) for r in observed]
            feasible = [(r, o) for r, o in outcomes if o['feasible']]
            effects = [o['paired_benefit'] for _, o in feasible if o['paired_benefit'] is not None]
            policy_table[key] = {
                'n_observations': len(outcomes), 'n_feasible': len(feasible),
                'accepted_complete_session_coverage': (sum(o['after_mps'] is not None for _, o in feasible)/len(feasible)
                                                       if feasible else None),
                'mean_additional_recording_seconds': (sum(o['additional_seconds'] for _, o in feasible)/len(feasible)
                                                      if feasible else None),
                'n_paired_benefit': len(effects),
                'positive_zero_negative_repeat_effect_counts': {
                    'positive': sum(x > 0 for x in effects),
                    'zero': sum(x == 0 for x in effects),
                    'negative': sum(x < 0 for x in effects)},
                'coverage_recovery_count': sum(r['initial_mps'] is None and o['after_mps'] is not None
                                               for r, o in feasible),
                'coverage_loss_count': sum(r['initial_mps'] is not None and o['after_mps'] is None
                                           for r, o in feasible),
                'unnecessary_S0_retest_count': sum(r['slot'] == 'S0' and bool(o['tasks']) for r, o in feasible),
                'unresolved_S4_retest_count': sum(r['slot'] == 'S4' and bool(o['tasks']) for r, o in feasible),
                'known_corrected_fault_target_count': sum(
                    r['slot'] == 'S3' and r.get('visible_fault_domain') in o['tasks']
                    for r, o in feasible),
            }
    exclusions = {}
    for row in rows:
        if row.get('status') == 'excluded':
            why = row['reason']
            exclusions[why] = exclusions.get(why, 0) + 1
    descriptive_keys = ('accepted_complete_initial', 'initial_mps_absolute_demo_reference_deviation',
                        'modeled_joint_mps_width', 'score_flip_fraction', 'mean_absolute_score_drift')
    descriptive = {key: cluster_interval(rows, profiles, key, seed=spec['bootstrap_seed'])
                   for key in descriptive_keys}
    by_family = {}
    for family in sorted({r.get('scenario_family') for r in observed if r.get('scenario_family') is not None}):
        family_rows = [r for r in observed if r.get('scenario_family') == family]
        by_family[family] = {'n_observations': len(family_rows),
            'n_parent_clusters': len({r['parent_id'] for r in family_rows}),
            'hard_rejected': sum(any(q == 'invalid' for q in r['quality_statuses']) for r in family_rows),
            'unknown_or_suspect': sum(any(q in ('unknown','suspect') for q in r['quality_statuses']) for r in family_rows)}
    return {'n_parent_clusters': len(profiles), 'n_scenario_rows': len(rows),
            'n_evaluable_scenario_rows': len(observed), 'excluded_by_reason': exclusions,
            'policy_by_budget': policy_table,
            'descriptive_parent_cluster_intervals': descriptive,
            'scenario_family_quality_counts': by_family,
            'primary_intervals': intervals,
            'preregistered_success': (success_checks(*[intervals[k] for k in keys])
                                      if rows and rows[0].get('split') == 'audit' else
                                      'not_applicable_to_development'),
            'limitations': ['Within-generator synthetic demo-rule evaluation only.',
                            'The original four classifiers were fitted using these V3 training parents.',
                            'Modeled sensitivity is not clinical confidence or device-calibrated tolerance.']}


def run(*, workspace: Path, dataset: Path, frozen_run: Path, output: Path,
        selected_models_sha256: str, approved_development: bool = False,
        split: str = 'development', approved_audit: bool = False,
        registry: Path | None = None, development_lock_sha256: str | None = None) -> Path:
    """Explicitly gated Colab entry point. Never called by notebook setup cells."""
    _colab_cpu()
    if split == 'development':
        if not approved_development or approved_audit:
            raise PermissionError('Development requires its own explicit approval')
    elif split == 'audit':
        if not approved_audit or registry is None or development_lock_sha256 is None:
            raise PermissionError('Locked audit requires separate approval and persistent registry')
    else:
        raise ValueError('Unknown innovation split')
    spec, parents = verify_design(workspace)
    models = verify_frozen_models(frozen_run, selected_models_sha256)
    generator_path = workspace / 'motor_scoring_research/v3/pp1_v3.py'
    if sha256(generator_path) != read_json(workspace / 'motor_scoring_research/v3/manifests/preparation_lock.json')['files']['pp1_v3.py']:
        raise ValueError('Original V3 synthesizer hash differs from preparation lock')
    _archive_preflight(dataset, parents, spec)
    ids = eligible_ids(parents, split, audit_enabled=approved_audit)
    if len(ids) != 84:
        raise ValueError('Expected exactly 84 parents in selected innovation split')
    if output.exists():
        raise FileExistsError('Every benchmark output must be a new versioned directory')
    allowed_output_parent = Path('/content/drive/MyDrive/PP1_V3_innovation_benchmark_runs')
    if output.parent.resolve() != allowed_output_parent:
        raise ValueError('Benchmark output must be a fresh child of the dedicated Drive runs folder')
    if split == 'audit':
        frozen_decision = registry / 'DEVELOPMENT_LOCK.json'
        if sha256(frozen_decision) != development_lock_sha256:
            raise ValueError('Reviewed development decision lock hash mismatch')
        decision = read_json(frozen_decision)
        if (decision.get('design_lock_sha256') != sha256(workspace / 'motor_scoring_research/innovations/benchmark/PHASE_3F_A_DESIGN_LOCK.json')
                or decision.get('selected_models_sha256') != models.selected_models_sha256
                or decision.get('source_code_sha256') != {
                    path.name: sha256(path) for path in sorted((workspace / 'motor_scoring_research/innovations/benchmark').glob('*.py'))}):
            raise ValueError('Development decision lock does not match audit software')
        # Claim the permanent marker before any audit raw case, score or result
        # is read or generated. A failure after this point still consumes audit.
        claim_audit(registry, {'split_manifest_sha256': spec['parent_split_manifest_sha256'],
                   'design_lock_sha256': sha256(workspace / 'motor_scoring_research/innovations/benchmark/PHASE_3F_A_DESIGN_LOCK.json'),
                   'selected_models_sha256': models.selected_models_sha256},
                   approved=True, frozen_development_sha256=development_lock_sha256)
    output.mkdir(parents=True, exist_ok=False)
    _write_json_new(output / 'RUN_STARTED.json', {'split': split,
        'design_lock_sha256': sha256(workspace / 'motor_scoring_research/innovations/benchmark/PHASE_3F_A_DESIGN_LOCK.json'),
        'selected_models_sha256': models.selected_models_sha256,
        'checkpoint_sha256': {m.domain.value: m.checkpoint_sha256 for m in models.models},
        'dataset_manifest_sha256': sha256(dataset / 'dataset_manifest.json'),
        'parent_count': len(ids), 'benchmark_generation_approved_for_this_call': True})
    adapter = load_verified_models(models, enable_inference=True)
    engine_path = workspace / 'motor_scoring_research/reference_engine/pp1_motor_engine.py'
    engine = load_reference_engine(engine_path, enable_extraction=True)
    bridge = CompleteSessionFeatureBridge(engine)
    trust_config, recap_config = load_config(), load_recap_config()
    config = read_json(dataset / 'config.json')
    # Use only the pinned original code copy for S6; it still requires Colab.
    import importlib.util
    module_spec = importlib.util.spec_from_file_location('pp1_v3_benchmark_pinned', generator_path)
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    # The JSONL control file is read only after split preflight; its audit
    # entries are indexed by ID but raw audit cases are never accessed in a
    # development run. No scores or parameters enter policy inputs.
    records = {}
    with (dataset / 'sessions.jsonl').open(encoding='utf-8') as stream:
        for line in stream:
            entry = json.loads(line)
            if entry['session_id'] in ids:
                records[entry['session_id']] = entry
    if set(records) != set(ids):
        raise ValueError('Selected parent session manifest is incomplete')
    profiles = {sid: parents[sid]['profile'] for sid in ids}
    rows = []
    for rank, sid in enumerate(ids):
        start = time.monotonic()
        anchor, record = _archive_parent(dataset, sid, split, parents, records, config)
        nominal_reference = _reference_mps(engine, anchor)
        with tempfile.TemporaryDirectory(dir='/content') as baseline_scratch:
            nominal = _evaluate_repeat(Observation(sid, split, 'S0', 'nominal', anchor,
                                       tuple()), Path(baseline_scratch)/'nominal', bridge, adapter)
        nominal_scores = _score_tuple(nominal)
        if nominal_scores is None:
            raise ValueError('Original V3 parent cannot produce frozen demo scores')
        for slot in SLOTS:
            initial, bank = scenario_pair(sid, split, rank, slot, anchor, spec,
                   parameters=record['parameters'], v3_config=config, synthesize=module._synthesize)
            lineage = {'parent_id': sid, 'split': split, 'slot': slot,
                       'rank': rank,
                       'severity_index': (rank//4 % 2 if slot in ('S1','S6') else
                                          rank%2 if slot == 'S2' else rank//6%2 if slot in ('S3','S4') else None),
                       'initial_raw_sha256': hashlib.sha256(repr(initial.raw).encode()).hexdigest(),
                       'potential_repeat_raw_sha256': {
                           d.value: hashlib.sha256(repr(bank[d].raw).encode()).hexdigest()
                           for d in DOMAIN_ORDER},
                       'repeat_bank_generated_before_policy': True}
            if not initial.eligible:
                rows.append({'parent_id': sid, 'split': split, 'slot': slot, 'status': 'excluded',
                             'reason': initial.exclusion_reason, 'lineage': lineage})
                continue
            with tempfile.TemporaryDirectory(dir='/content') as scratch:
                scratch = Path(scratch)
                initial_assessment, trust = _evaluate_observation(initial, scratch/'initial', bridge,
                       adapter, trust_config, spec['master_seed'] + rank*7 + SLOTS.index(slot))
                reference = (_reference_mps(engine, initial.raw) if slot == 'S6'
                             else nominal_reference)
                initial_scores = _score_tuple(initial_assessment)
                initial_mps = mps(initial_scores)
                score_drift = (tuple(x-y for x, y in zip(initial_scores, nominal_scores))
                               if initial_scores is not None else None)
                row = {'parent_id': sid, 'split': split, 'profile': profiles[sid], 'slot': slot,
                       'lineage': lineage,
                       'scenario_family': initial.fault_truth, 'initial_scores': initial_scores,
                       'nominal_frozen_scores': nominal_scores,
                       'signed_ordinal_score_drift': score_drift,
                       'initial_mps': initial_mps, 'simulator_rule_demo_mps': reference,
                       'quality_statuses': [q.status.value for q in initial_assessment.quality],
                       'trust_status': trust['status'], 'joint_modeled_sensitivity':
                       trust.get('joint_mps_sensitivity'), 'policy_outcomes': {}}
                quality_bad = any(q.status.value != 'valid' for q in initial_assessment.quality)
                row['accepted_complete_initial'] = float(initial_mps is not None)
                if score_drift is not None:
                    row['score_flip_fraction'] = sum(x != 0 for x in score_drift)/4
                    row['mean_absolute_score_drift'] = sum(abs(x) for x in score_drift)/4
                row['initial_mps_absolute_demo_reference_deviation'] = (
                    abs(initial_mps-reference) if initial_mps is not None else None)
                if trust.get('joint_mps_sensitivity'):
                    row['modeled_joint_mps_width'] = trust['joint_mps_sensitivity']['observed_width']
                if slot in ('S0', 'S1', 'S2', 'S6'):
                    row['technical_false_rejection'] = float(any(q.status is QualityStatus.INVALID
                                                                 for q in initial_assessment.quality))
                if slot in ('S3', 'S4') and rank%6 < 4:
                    row['observable_fault_detected'] = float(quality_bad)
                # Only externally confirmed *observable* S3 repair is shown to
                # policies. The latent tap-event correction remains hidden.
                remediation = {}
                if slot in ('S3', 'S4'):
                    fault_domain = Domain.BS if rank%6 >= 4 else (Domain.TS, Domain.RS, Domain.GI)[rank//6%3]
                    remediation[fault_domain] = ('corrected' if slot == 'S3' and rank%6 < 4 else 'unresolved')
                    if slot == 'S3' and rank%6 < 4:
                        row['visible_fault_domain'] = fault_domain.value
                # A repeat bank is evaluator-only. Cache each distinct selected
                # task set across policies/budgets, avoiding duplicate inference.
                evaluated_repeats = {}
                for budget in spec['budgets_additional_recording_seconds']:
                    for policy in POLICIES:
                        decision = decide(policy, initial_assessment, trust, remediation, budget,
                            parent_id=sid, slot=slot, random_seed=spec['random_policy_seed'],
                            recap_config=recap_config)
                        key = f'{policy}@{budget}'
                        outcome = {'tasks': [d.value for d in decision.tasks],
                                   'additional_seconds': decision.additional_seconds,
                                   'reason': decision.reason, 'feasible': decision.feasible,
                                   'after_scores': initial_scores, 'after_mps': initial_mps}
                        # Evaluate only the selected potential repeats; unchosen
                        # outcomes are never revealed to decision functions.
                        if decision.tasks and decision.tasks in evaluated_repeats:
                            previous = evaluated_repeats[decision.tasks]
                            outcome['after_scores'], outcome['after_mps'], outcome['repeat_failure'] = previous
                        elif decision.tasks:
                            candidate_raw = list(initial.raw)
                            candidate_context = list(initial.contexts)
                            unavailable = []
                            for d in decision.tasks:
                                potential = bank[d]
                                if not potential.eligible:
                                    unavailable.append(potential.exclusion_reason)
                                index = DOMAIN_ORDER.index(d)
                                candidate_raw[index] = potential.raw[index]
                                candidate_context[index] = potential.contexts[index]
                            if unavailable:
                                outcome['repeat_failure'] = unavailable
                                outcome['after_scores'] = None
                                outcome['after_mps'] = None
                            else:
                                combined = Observation(sid, split, slot, 'selected_repeat',
                                           tuple(candidate_raw), tuple(candidate_context))
                                after = _evaluate_repeat(combined, scratch/key.replace('@','_'),
                                           bridge, adapter)
                                outcome['after_scores'] = _score_tuple(after)
                                outcome['after_mps'] = mps(outcome['after_scores'])
                            evaluated_repeats[decision.tasks] = (outcome['after_scores'],
                                outcome['after_mps'], outcome.get('repeat_failure'))
                        outcome['paired_benefit'] = (paired_effect(initial_mps, outcome['after_mps'], reference)
                                                     if slot != 'S6' else None)
                        row['policy_outcomes'][key] = outcome
                c = row['policy_outcomes']['C_random_eligible@180']['after_mps']
                f = row['policy_outcomes']['F_full_trust_recap@180']['after_mps']
                if slot != 'S6' and c is not None and f is not None:
                    row['F_minus_C_error_180'] = abs(c-reference)-abs(f-reference)
                if slot in ('S0', 'S1', 'S2') and initial_scores is not None:
                    # Prospective same-state event is observed from the *same*
                    # policy-blind potential bank, irrespective of policy choice.
                    flags, flips = [], []
                    for d in DOMAIN_ORDER:
                        flag = _sensitivity_flag(trust, d)
                        if flag is None or not bank[d].eligible:
                            continue
                        taskset = (d,)
                        if taskset not in evaluated_repeats:
                            candidate_raw = list(initial.raw)
                            candidate_context = list(initial.contexts)
                            index = DOMAIN_ORDER.index(d)
                            candidate_raw[index] = bank[d].raw[index]
                            candidate_context[index] = bank[d].contexts[index]
                            combined = Observation(sid, split, slot, 'prospective_repeat',
                                                   tuple(candidate_raw), tuple(candidate_context))
                            after = _evaluate_repeat(combined, scratch/f'prospective_{d.value}', bridge, adapter)
                            evaluated_repeats[taskset] = (_score_tuple(after), after.mps, None)
                        next_scores = evaluated_repeats[taskset][0]
                        if next_scores is not None:
                            flags.append(flag)
                            flips.append(float(next_scores[DOMAIN_ORDER.index(d)] != initial_scores[DOMAIN_ORDER.index(d)]))
                    row['prospective_flagged_flip_sum'] = sum(flip for flag, flip in zip(flags, flips) if flag)
                    row['prospective_flagged_n'] = sum(flags)
                    row['prospective_unflagged_flip_sum'] = sum(flip for flag, flip in zip(flags, flips) if not flag)
                    row['prospective_unflagged_n'] = len(flags)-sum(flags)
                rows.append(row)
        # A completed parent chunk is immutable and independently checkable.
        parent_path = output / f'parent_{rank:03d}.json'
        _write_json_new(parent_path, {'parent_id': sid, 'split': split,
                                      'elapsed_colab_cpu_seconds': time.monotonic()-start,
                                      'rows': [r for r in rows if r['parent_id'] == sid]})
    summary = _summarize(rows, profiles, spec)
    _write_json_new(output / 'summary.json', summary)
    _write_json_new(output / 'COMPLETE.json', {'split': split, 'parent_count': len(ids),
                    'summary_sha256': sha256(output / 'summary.json'),
                    'parent_file_sha256': {p.name: sha256(p) for p in sorted(output.glob('parent_*.json'))}})
    if split == 'development':
        _write_json_new(output / 'DEVELOPMENT_FREEZE_CANDIDATE.json', {
            'design_lock_sha256': sha256(workspace / 'motor_scoring_research/innovations/benchmark/PHASE_3F_A_DESIGN_LOCK.json'),
            'selected_models_sha256': models.selected_models_sha256,
            'development_summary_sha256': sha256(output / 'summary.json'),
            'source_code_sha256': {
                path.name: sha256(path) for path in sorted((workspace / 'motor_scoring_research/innovations/benchmark').glob('*.py'))},
            'review_required_before_audit': True})
    return output
