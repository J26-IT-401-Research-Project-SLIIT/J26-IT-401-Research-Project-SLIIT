"""Colab-only reproduction of V2 stored features/targets; no ML evaluation."""
import argparse
import csv
from pathlib import Path

from pp1_v3 import (DEFAULT_ENGINE, DOMAINS, assess, load_engine, read_config,
                    read_raw, require_colab_cpu, sha256, write_json)

MAPPING = {
    'ts_tremor_rms_ms2': ('TS', 'tremor_rms_ms2'),
    'bs_tap_count': ('BS', 'tap_count'),
    'bs_tap_rate_per_min': ('BS', 'tap_rate_per_min'),
    'bs_tap_interval_std_s': ('BS', 'tap_interval_std_s'),
    'rs_mean_gyro_peak_rad_s': ('RS', 'mean_gyro_peak_rad_s'),
    'rs_movement_peaks': ('RS', 'movement_peaks'),
    'gi_step_candidates': ('GI', 'step_candidates'),
    'gi_cadence_steps_min': ('GI', 'cadence_steps_min'),
    'gi_step_interval_std_s': ('GI', 'step_interval_std_s'),
}
TARGETS = dict(TS='TS_demo', BS='BS_demo', RS='RS_active_mobility_proxy_demo', GI='GI_demo')


def verify(root, engine_path):
    require_colab_cpu()
    c = read_config(); engine = load_engine(engine_path, c)
    def table(name):
        with (root/name).open(encoding='utf-8-sig', newline='') as stream:
            rows = list(csv.DictReader(stream))
        if len({r['session_id'] for r in rows}) != len(rows):
            raise ValueError(f'Duplicate IDs: {name}')
        return {r['session_id']: r for r in rows}
    stored = table('features_extracted_by_existing_PP1_engine.csv')
    labels = table('demo_rule_scores_for_fusion.csv')
    provenance = table('sessions_and_provenance.csv')
    if not (set(stored) == set(labels) == set(provenance) == {p.name for p in (root/'cases').iterdir()}):
        raise ValueError('V2 session membership mismatch')
    feature_mismatches, label_mismatches = [], []
    max_error = {key: 0.0 for key in MAPPING}
    # V2 writes signals to six decimals but scores BEFORE serialization.
    # 2e-6 allows raw rounding plus feature-table rounding, not changed peaks.
    tolerance = 2e-6
    integer_features = {'bs_tap_count', 'rs_movement_peaks', 'gi_step_candidates'}
    for sid in sorted(stored):
        result = assess(engine, read_raw(root/'cases'/sid, c))
        for key, (domain, field) in MAPPING.items():
            actual = result['features'][domain][field]; saved = stored[sid][key]
            if actual is None or saved == '':
                if not (actual is None and saved == ''):
                    feature_mismatches.append([sid, key, saved, actual])
                continue
            error = abs(actual-float(saved)); max_error[key] = max(max_error[key], error)
            if error > (0 if key in integer_features else tolerance):
                feature_mismatches.append([sid, key, saved, actual])
        for d in DOMAINS:
            if result['scores'][d] != int(labels[sid][TARGETS[d]]):
                label_mismatches.append([sid, d, labels[sid][TARGETS[d]], result['scores'][d]])
        if result['mps'] is None or result['mps'] != float(labels[sid]['demo_MPS_0_1']):
            label_mismatches.append([sid, 'MPS', labels[sid]['demo_MPS_0_1'], result['mps']])
    return {'sessions': len(stored), 'engine_sha256': sha256(engine_path),
            'feature_absolute_tolerance': tolerance, 'max_feature_absolute_error': max_error,
            'feature_mismatches': feature_mismatches, 'label_mismatches': label_mismatches,
            'passed': not feature_mismatches and not label_mismatches,
            'interpretation': 'Raw-to-stored reproduction, not clinical or ML accuracy'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--v2', type=Path, required=True)
    parser.add_argument('--engine', type=Path, default=DEFAULT_ENGINE)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    if args.report.exists():
        raise FileExistsError(args.report)
    report = verify(args.v2, args.engine)
    write_json(args.report, report)
    print('V2 reproduction:', 'PASS' if report['passed'] else 'MISMATCH: inspect report')
    if not report['passed']:
        raise SystemExit(1)
