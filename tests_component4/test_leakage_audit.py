import json
import pandas as pd
from multimodal_fusion.settings import ARTIFACTS,DATA,LABELS,ACTIVE_DATASET

def test_demo_presets_use_fixed_validation_rule():
    cases=json.loads((ARTIFACTS/'demo_cases.json').read_text())
    frame=pd.read_csv(DATA/ACTIVE_DATASET)
    for label,case in zip(LABELS,cases[:5]):
        expected=frame[(frame.split=='validation')&(frame.label==label)].sort_values('synthetic_id').iloc[0]
        assert case['synthetic_id']==expected.synthetic_id
        assert case['split']=='validation'
        assert 'not prediction-selected' in case['selection']

def test_leakage_audit_invariants_and_live_preservation():
    audit=json.loads((ARTIFACTS/'leakage_and_simulation_audit.json').read_text())
    checks=audit['direct_checks']
    assert audit['active_scenario']=='moderate_overlap'
    assert audit['dataset_filename']==ACTIVE_DATASET
    assert not any(checks['participant_overlap'].values())
    assert not any(checks['exact_feature_overlap'].values())
    assert checks['training_background_matches_training_only']
    assert not checks['metadata_or_labels_used_as_features']
    assert audit['live_dataset_unchanged'] and audit['live_model_unchanged']
    assert len(audit['simulator_sensitivity'])==3
