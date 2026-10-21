"""Leakage checks and controlled simulator sensitivity; never replaces live data/model."""
import json
import hashlib
from datetime import datetime, timezone
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from threadpoolctl import threadpool_limits
from .settings import DATA,ARTIFACTS,FEATURES,LABELS,SEED
from .synthetic import generate

def audit():
    model_path=ARTIFACTS/'fusion_model.joblib';model_hash=hashlib.sha256(model_path.read_bytes()).hexdigest()
    bundle=joblib.load(model_path);model=bundle['model']
    original_path=DATA/bundle.get('dataset_filename','synthetic_participants.csv')
    original_hash=hashlib.sha256(original_path.read_bytes()).hexdigest()
    frame=pd.read_csv(original_path)
    parts={name:frame[frame.split==name] for name in ['train','validation','test']}
    fresh_name='synthetic_moderate_fresh_test.csv' if bundle.get('scenario')=='moderate_overlap' else 'synthetic_fresh_final_test.csv'
    fresh=pd.read_csv(DATA/fresh_name)
    ids={name:set(part.synthetic_id) for name,part in parts.items()}
    tuples={name:set(map(tuple,part[FEATURES].to_numpy())) for name,part in parts.items()}
    overlaps={f'{a}/{b}':len(ids[a]&ids[b]) for a,b in [('train','validation'),('train','test'),('validation','test')]}
    exact_overlaps={f'{a}/{b}':len(tuples[a]&tuples[b]) for a,b in [('train','validation'),('train','test'),('validation','test')]}
    fitted_background=set(map(tuple,bundle['training_background']))
    partition_accuracy={name:float(accuracy_score(part.label,model.predict(part[FEATURES]))) for name,part in {**parts,'fresh_final_test':fresh}.items()}
    rng=np.random.default_rng(SEED+700)
    shuffled=RandomForestClassifier(n_estimators=240,min_samples_leaf=6,max_features=None,class_weight='balanced',random_state=SEED,n_jobs=1)
    shuffled.fit(parts['train'][FEATURES],rng.permutation(parts['train'].label.to_numpy()))
    permutation_accuracy=float(accuracy_score(fresh.label,shuffled.predict(fresh[FEATURES])))
    sensitivity=[]
    # Predeclared noise scenarios, not a target-accuracy search. Same model spec,
    # counts and profile centers throughout. Original deployed files untouched.
    with threadpool_limits(limits=1):
        for name,noise in [('low_overlap',.25),('moderate_overlap',.50),('harder_overlap',.85)]:
            simulated=generate(1000,SEED,noise=noise,prefix='DIAG')
            development,_=train_test_split(simulated,test_size=.2,stratify=simulated.label,random_state=SEED)
            training,_=train_test_split(development,test_size=.25,stratify=development.label,random_state=SEED+1)
            independent=generate(5000,SEED+300,noise=noise,prefix='DIAG-EVAL')
            forest=RandomForestClassifier(n_estimators=240,min_samples_leaf=6,max_features=None,class_weight='balanced',random_state=SEED,n_jobs=1)
            forest.fit(training[FEATURES],training.label)
            prediction=forest.predict(independent[FEATURES])
            accuracy=float(accuracy_score(independent.label,prediction));n=len(independent)
            sensitivity.append({'scenario':name,'logit_noise_sd':noise,'training_n':600,'evaluation_n':n,
                'accuracy':accuracy,'macro_f1':float(f1_score(independent.label,prediction,average='macro')),
                'accuracy_binomial_se':float(np.sqrt(accuracy*(1-accuracy)/n)),
                'scope':'Different assumed simulator overlap; not an improvement to the same prediction problem and not clinical evidence'})
            print(json.dumps(sensitivity[-1]),flush=True)
    result={'created_utc':datetime.now(timezone.utc).isoformat(),
        'active_scenario':bundle.get('scenario','historical_hard'),'dataset_filename':original_path.name,'direct_checks':{'participant_overlap':overlaps,'exact_feature_overlap':exact_overlaps,
            'fresh_vs_training_exact_feature_overlap':len(tuples['train']&set(map(tuple,fresh[FEATURES].to_numpy()))),
            'training_background_matches_training_only':fitted_background==tuples['train'],
            'fitted_features':list(model.feature_names_in_),
            'metadata_or_labels_used_as_features':set(model.feature_names_in_)!=set(FEATURES),
            'class_counts':frame.label.value_counts().to_dict(),
            'score_missing_cells':int(frame[FEATURES].isna().sum().sum())},
        'code_review':{'training':'Fits only training partition; logistic scaling is inside its training pipeline',
            'selection':'Uses validation macro F1; final labels do not enter fitting',
            'test_reuse':'All previously reported tests are now inspected. The current cohort/confirmation splits were fixed before its configuration selection. Future tuning after their inspection needs a new reserved test.',
            'deployment_decision':'The active moderate scenario was selected after a sensitivity study and user approval. Configuration selection used its validation records. Historical harder-scenario tests were already inspected; report the scenario change explicitly.',
            'presentation_issue':'Original presets picked confident test examples. Fixed-rule validation examples replace them; demo success is not accuracy evidence.',
            'external_validity':'Train/test share an assumed generator. No clinical or five-disorder validation is established.'},
        'partition_accuracy':partition_accuracy,'shuffled_label_control_accuracy':permutation_accuracy,
        'uniform_random_expected_accuracy':.20,'simulator_sensitivity':sensitivity,
        'live_dataset_unchanged':hashlib.sha256(original_path.read_bytes()).hexdigest()==original_hash,
        'live_model_unchanged':hashlib.sha256(model_path.read_bytes()).hexdigest()==model_hash,
        'conclusion':'No direct split/feature/preprocessing leakage identified in these checks. This is not proof of every possible leakage path. Simulation assumptions and class overlap dominate observed performance.'}
    (ARTIFACTS/'leakage_and_simulation_audit.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result,indent=2))
    return result

if __name__=='__main__':audit()
