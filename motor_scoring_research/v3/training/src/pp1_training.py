"""Colab-only PP1 V3 training. Importing this module never fits a model.

Synthetic rule imitation only. No generator entry points are called.
"""
from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import math
import os
import platform
import random
import shutil
import sys
from collections import Counter
from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[1]
DOMAINS = ('TS', 'BS', 'RS', 'GI')
FEATURES = {
    'TS': ['TS_tremor_rms_ms2'],
    'BS': ['BS_tap_count', 'BS_tap_rate_per_min', 'BS_tap_interval_std_s'],
    'RS': ['RS_mean_gyro_peak_rad_s', 'RS_movement_peaks'],
    'GI': ['GI_step_candidates', 'GI_cadence_steps_min', 'GI_step_interval_std_s'],
}
TARGETS = {'TS': 'TS', 'BS': 'BS', 'RS': 'RS', 'GI': 'GI'}
DISPLAY_TARGETS = {'TS': 'TS_demo', 'BS': 'BS_demo',
                   'RS': 'RS_active_mobility_proxy_demo', 'GI': 'GI_demo'}
ENGINE_HASH = 'c39620fe8738ca9217b1b7f9d75141e0f2ab9a7b49d22ebd63b6234202c437e4'
SPLIT_COUNTS = {'train': 168, 'validation': 36, 'test': 36}


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def save_json(path, obj):
    with Path(path).open('x', encoding='utf-8') as stream:
        json.dump(obj, stream, indent=2, allow_nan=False)
        stream.write('\n')


def save_csv(path, rows):
    if not rows:
        raise ValueError('Refuse an empty report')
    with Path(path).open('x', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def csv_rows(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or len(reader.fieldnames) != len(set(reader.fieldnames)):
            raise ValueError(f'Invalid/duplicate header: {path}')
        rows = list(reader)
    if any(None in row or any(v is None for v in row.values()) for row in rows):
        raise ValueError(f'Malformed CSV: {path}')
    return rows


def by_id(rows):
    result = {r['session_id']: r for r in rows}
    if len(result) != len(rows) or '' in result:
        raise ValueError('Duplicate or empty session IDs')
    return result


def require_colab():
    if not Path('/content').is_dir() or not os.environ.get('COLAB_RELEASE_TAG'):
        raise RuntimeError('Actual data processing and fitting require Colab CPU')
    import google.colab  # noqa: F401
    if Path('/dev/nvidia0').exists() or os.environ.get('COLAB_TPU_ADDR'):
        raise RuntimeError('Choose accelerator None / CPU')
    sys.dont_write_bytecode = True


def safe_relative(root, relative):
    parts = relative.split('/')
    if not relative or '\\' in relative or ':' in relative or any(p in ('', '.', '..') for p in parts):
        raise ValueError('Unsafe manifest path')
    result = Path(root).joinpath(*parts)
    if not result.resolve().is_relative_to(Path(root).resolve()):
        raise ValueError('Path escapes root')
    return result


def verify_hashes(root, files):
    for relative, expected in files.items():
        if sha(safe_relative(root, relative)) != expected:
            raise ValueError(f'Hash mismatch: {relative}')


def verify_package():
    lock = read_json(PACKAGE/'package_lock.json')
    verify_hashes(PACKAGE, lock['files'])
    return lock


def verify_environment():
    from importlib.metadata import version
    for line in (PACKAGE/'requirements-colab.txt').read_text().splitlines():
        name,expected=line.split('==')
        if version(name)!=expected:
            raise RuntimeError(f'Expected {name}=={expected}; install pinned Colab dependencies and restart kernel')


def ordinal(value):
    number = float(value)
    if not math.isfinite(number) or number != int(number) or int(number) not in range(5):
        raise ValueError(f'Invalid ordinal score: {value}')
    return int(number)


def cv_folds(y):
    counts = Counter(y)
    if len(counts) < 2:
        raise ValueError('Fewer than two training classes: the four-classifier comparison is unsupported')
    return min(3, min(counts.values())) if min(counts.values()) >= 2 else 0


def wilson(success, n):
    if n == 0:
        return None
    z = 1.959963984540054; p = success/n
    center = (p+z*z/(2*n))/(1+z*z/n)
    half = z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/(1+z*z/n)
    return [max(0., center-half), min(1., center+half)]


def mean_interval(values, seed=20261021, replicates=2000):
    if not values:
        return None
    rng = random.Random(seed); n = len(values)
    means = sorted(sum(values[rng.randrange(n)] for _ in range(n))/n for _ in range(replicates))
    return [means[int(.025*(replicates-1))], means[int(.975*(replicates-1))]]


def metrics(truth, predicted, uncertainty=False, replicates=2000):
    if len(truth) != len(predicted) or not truth:
        raise ValueError('Predictions must align with nonempty labels')
    truth = [ordinal(v) for v in truth]; predicted = [ordinal(v) for v in predicted]
    cm = [[0]*5 for _ in range(5)]
    for a, b in zip(truth, predicted): cm[a][b] += 1
    support = [sum(row) for row in cm]; n = len(truth)
    predicted_support = [sum(cm[i][j] for i in range(5)) for j in range(5)]
    f1 = [2*cm[i][i]/(support[i]+predicted_support[i]) if support[i]+predicted_support[i] else 0. for i in range(5)]
    errors = [abs(a-b) for a,b in zip(truth,predicted)]
    exact = sum(e == 0 for e in errors); within = sum(e <= 1 for e in errors)
    result = {'n': n, 'exact_accuracy': exact/n, 'within_1_accuracy': within/n,
              'macro_f1': sum(f1)/5, 'balanced_accuracy': sum(cm[i][i]/support[i] for i in range(5) if support[i])/sum(s>0 for s in support),
              'MAE': sum(errors)/n, 'confusion_matrix': cm, 'class_support': support,
              'unsupported_true_classes': [i for i,s in enumerate(support) if not s]}
    if uncertainty:
        result['uncertainty'] = {'exact_accuracy_wilson_95': wilson(exact,n),
            'within_1_accuracy_wilson_95': wilson(within,n), 'MAE_session_bootstrap_95': mean_interval(errors,replicates=replicates),
            'class_recall_wilson_95': [wilson(cm[i][i],support[i]) for i in range(5)],
            'limitation': 'Conditional on these simulator sessions and frozen models; not source/clinical uncertainty. Rare-class intervals are unstable; no multiplicity correction.'}
    return result


def cv_rank(row):
    return (row['mean_fold_MAE'], -row['mean_fold_macro_f1'], row['configuration_index'])


def validation_rank(row, order):
    return (row['MAE'], -row['macro_f1'], order.index(row['algorithm']))


def validate_tables(features, targets, splits, expected_plan):
    """Pure table checks: exercised only on tiny fixtures in local tests."""
    f, y, s, plan = map(by_id, (features, targets, splits, expected_plan))
    if not (set(f) == set(y) == set(s) == set(plan)):
        raise ValueError('Session sets differ')
    allowed = {'session_id'} | {k for values in FEATURES.values() for k in values}
    for sid in s:
        if s[sid] != plan[sid]: raise ValueError('Frozen split/provenance changed')
        if set(f[sid]) != allowed or set(y[sid]) != {'session_id','TS','BS','RS','GI','demo_MPS'}:
            raise ValueError('Unexpected feature/target schema; refuse implicit mappings')
        for key in allowed-{'session_id'}:
            if not math.isfinite(float(f[sid][key])): raise ValueError('Missing/nonfinite feature in complete dataset')
        values = [ordinal(y[sid][d]) for d in DOMAINS]
        if float(y[sid]['demo_MPS']) != sum(values)/16: raise ValueError('Reference MPS mismatch')
    return f, y, s


def load_tables(dataset):
    return validate_tables(*(csv_rows(Path(dataset)/name) for name in ('features.csv','targets.csv','splits.csv')),
                           csv_rows(PACKAGE/'configs/frozen_session_plan.csv'))


def preflight(dataset, run):
    require_colab(); verify_package(); verify_environment()
    dataset, run = Path(dataset).resolve(), Path(run).resolve()
    if run == dataset or run.is_relative_to(dataset): raise ValueError('Run must be outside read-only dataset')
    if run.is_relative_to(PACKAGE): raise ValueError('Run must be outside the frozen code package')
    if run.exists(): raise FileExistsError(run)
    complete = read_json(dataset/'COMPLETE.json'); report = read_json(dataset/'integrity_report.json')
    meta = read_json(dataset/'dataset_manifest.json')
    if complete.get('integrity_passed') is not True or complete.get('sessions') != 240:
        raise ValueError('Missing passing completion marker')
    if report.get('status') != 'passed' or report.get('session_count') != 240 or report.get('complete_score_sessions') != 240:
        raise ValueError('Expected passing 240-complete-session integrity report')
    if meta.get('sessions') != 240 or meta.get('stress_only') is not False or meta.get('engine_sha256') != ENGINE_HASH:
        raise ValueError('Unexpected dataset/engine')
    anchors = read_json(PACKAGE/'configs/input_contract.json')
    verify_hashes(dataset, anchors['required_hashes'])
    if read_json(dataset/'evaluation_protocol.json') != read_json(PACKAGE/'configs/frozen_evaluation_protocol.json'):
        raise ValueError('Dataset evaluation protocol differs from frozen protocol')
    for key, name in [('config_sha256','config.json'),('feature_schema_sha256','feature_schema.json'),
                      ('splits_sha256','splits.csv'),('generator_sha256','code/pp1_v3.py'),
                      ('source_manifest_sha256','source_manifest.json'),('evaluation_protocol_sha256','evaluation_protocol.json')]:
        if sha(dataset/name) != meta[key]: raise ValueError(f'Dataset manifest mismatch: {name}')
    schema = {d:[f'{d}_{k}' for k in fields] for d,fields in read_json(dataset/'feature_schema.json').items()}
    if schema != FEATURES: raise ValueError('Feature schema mismatch')
    f,y,s = load_tables(dataset)
    if len(s) != 240 or dict(Counter(r['split'] for r in s.values())) != SPLIT_COUNTS:
        raise ValueError('Expected 240 unique sessions with frozen 168/36/36 splits')
    records = [json.loads(line) for line in (dataset/'sessions.jsonl').read_text().splitlines()]
    if len(records) != 240 or {r['session_id'] for r in records} != set(s): raise ValueError('Session manifest mismatch')
    # Read/hash raw bytes only; do not produce predictions or tune on test signals.
    hashes = {p.relative_to(dataset).as_posix(): sha(p) for p in dataset.rglob('*') if p.is_file()}
    expected_raw={f"cases/{sid}/{d}_{180 if d=='GI' else 10}s.csv" for sid in s for d in DOMAINS}
    if {k for k in hashes if k.startswith('cases/')} != expected_raw:
        raise ValueError('Expected exactly four raw files per session')
    for record in records:
        sid = record['session_id']
        if record['split'] != s[sid]['split']: raise ValueError('Manifest split mismatch')
        if record['scores'] != {d:ordinal(y[sid][d]) for d in DOMAINS}: raise ValueError('Manifest target mismatch')
        for d in DOMAINS:
            name = f"cases/{sid}/{d}_{180 if d=='GI' else 10}s.csv"
            if hashes.get(name) != record['raw_sha256'][d]: raise ValueError(f'Raw recording missing/changed: {name}')
    counts = {sp:{d:[sum(ordinal(y[sid][d])==k for sid in s if s[sid]['split']==sp) for k in range(5)] for d in DOMAINS} for sp in SPLIT_COUNTS}
    if counts != report['class_counts_0_to_4']: raise ValueError('Stored integrity class counts disagree')
    warnings = [f'{sp}/{d}/score{k}: support={n}' for sp in counts for d in DOMAINS for k,n in enumerate(counts[sp][d]) if n<5]
    for d in DOMAINS:
        if any(sum(counts[sp][d][k] for sp in counts)==0 for k in range(5)):
            raise ValueError('Reported global class coverage is not present')
    run.mkdir(parents=True,exist_ok=False)
    shutil.copytree(PACKAGE,run/'package')
    save_json(run/'dataset_snapshot.json',{'root':str(dataset),'files':hashes})
    save_json(run/'search_spaces.json',read_json(PACKAGE/'configs/search_spaces.json'))
    save_json(run/'mapping.json',{'features':FEATURES,'actual_targets':TARGETS,'display_targets':DISPLAY_TARGETS,'MPS':'sum(TS,BS,RS,GI)/16'})
    result = {'status':'passed','dataset_root':str(dataset),'sessions':240,'split_counts':SPLIT_COUNTS,
              'class_counts':counts,'warnings':warnings,'actual_target_columns':list(next(iter(y.values()))),
              'features':FEATURES,'test_access':'Schema/labels/counts/byte hashes only; no model predictions',
              'claim':'within-generator synthetic demo-label learning; not independent-source or clinical validation'}
    save_json(run/'preflight.json',result)
    return result


def stage_data(run):
    require_colab(); verify_package(); verify_environment()
    run=Path(run)
    verify_hashes(run/'package',read_json(run/'package/package_lock.json')['files'])
    if sha(run/'package/package_lock.json') != sha(PACKAGE/'package_lock.json'):
        raise ValueError('Use the same frozen training package throughout this run')
    if read_json(run/'search_spaces.json') != read_json(PACKAGE/'configs/search_spaces.json'):
        raise ValueError('Search spaces changed')
    snapshot=read_json(run/'dataset_snapshot.json'); dataset=Path(snapshot['root'])
    verify_hashes(dataset,{k:v for k,v in snapshot['files'].items() if not k.startswith('cases/')})
    return dataset, load_tables(dataset)


def matrix(tables, domain, split):
    import numpy as np
    f,y,s=tables
    ids=sorted(sid for sid in s if s[sid]['split']==split)
    x=np.asarray([[float(f[sid][key]) for key in FEATURES[domain]] for sid in ids])
    target=np.asarray([ordinal(y[sid][domain]) for sid in ids])
    return ids,x,target


def build_pipeline(algorithm, params, seed):
    from sklearn.pipeline import Pipeline
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    if algorithm=='RF':
        from sklearn.ensemble import RandomForestClassifier
        model=RandomForestClassifier(**params,random_state=seed,n_jobs=1)
    elif algorithm=='SVM':
        from sklearn.svm import SVC
        model=SVC(**params,kernel='rbf',probability=False,random_state=seed)
    elif algorithm=='XGB':
        from xgboost import XGBClassifier
        model=XGBClassifier(**params,random_state=seed,n_jobs=1,tree_method='hist',device='cpu',verbosity=0)
    elif algorithm=='CatBoost':
        from catboost import CatBoostClassifier
        model=CatBoostClassifier(**params,random_seed=seed,thread_count=1,task_type='CPU',verbose=False,allow_writing_files=False)
    else: raise ValueError(algorithm)
    steps=[('imputer',SimpleImputer(strategy='median',keep_empty_features=True))]
    if algorithm=='SVM': steps.append(('scaler',StandardScaler()))
    return Pipeline(steps+[('model',model)])


def encode_labels(y):
    """Training-only contiguous encoding handles absent ordinal classes for XGBoost."""
    classes=sorted(set(int(v) for v in y))
    return classes,[classes.index(int(v)) for v in y]


def decode_labels(classes, codes):
    return [classes[int(code)] for code in codes]


def fit_candidate(algorithm,params,x,y,seed):
    require_colab()
    classes,encoded=encode_labels(y)
    if len(classes)<2: raise ValueError('Cannot fit comparison with one class')
    pipeline=build_pipeline(algorithm,params,seed)
    pipeline.fit(x,encoded)
    return {'pipeline':pipeline,'ordinal_classes':classes}


def predict(model,x):
    require_colab()
    import numpy as np
    codes=np.asarray(model['pipeline'].predict(x)).reshape(-1)
    if any(float(c)!=int(c) or int(c)<0 or int(c)>=len(model['ordinal_classes']) for c in codes):
        raise ValueError('Invalid encoded model prediction')
    return decode_labels(model['ordinal_classes'],codes)


def stage_seal(folder, name):
    files={p.relative_to(folder).as_posix():sha(p) for p in Path(folder).rglob('*') if p.is_file()}
    save_json(Path(folder)/name,{'files':files})


def train(run, approved=False):
    if not approved: raise PermissionError('Training approval is required')
    require_colab()
    import numpy as np
    import joblib
    from sklearn.model_selection import StratifiedKFold
    from importlib.metadata import version
    run=Path(run); _,tables=stage_data(run)
    folder=run/'training'; folder.mkdir(exist_ok=False)
    config=read_json(run/'search_spaces.json')
    save_json(folder/'environment.json',{'python':platform.python_version(),'packages':{p:version(p) for p in ['numpy','scipy','scikit-learn','xgboost','catboost','joblib']},'threads':1})
    summaries=[]
    for d in DOMAINS:
        ids,x,y=matrix(tables,d,'train'); folds=cv_folds(y.tolist())
        partitions=list(StratifiedKFold(folds,shuffle=True,random_state=config['seed']).split(x,y)) if folds else []
        domain_folder=folder/d; domain_folder.mkdir()
        save_json(domain_folder/'fold_assignments.json',{'n_splits':folds,'folds':[{'train_ids':[ids[i] for i in a], 'heldout_ids':[ids[i] for i in b]} for a,b in partitions],
                    'policy':'singleton observed class: skip CV and use configuration 0' if not folds else 'training-only stratified CV'})
        for algorithm,space in config['spaces'].items():
            destination=domain_folder/algorithm; destination.mkdir()
            trials=[]
            if folds:
                for index,params in enumerate(space):
                    fold_reports=[]; oof=[None]*len(y)
                    for fitting,heldout in partitions:
                        fitted=fit_candidate(algorithm,params,x[fitting],y[fitting],config['seed'])
                        prediction=predict(fitted,x[heldout])
                        for pos,value in zip(heldout,prediction): oof[int(pos)]=value
                        fold_reports.append(metrics(y[heldout].tolist(),prediction))
                    trial={'configuration_index':index,'params':params,'fold_metrics':fold_reports,
                           'mean_fold_MAE':sum(m['MAE'] for m in fold_reports)/folds,
                           'mean_fold_macro_f1':sum(m['macro_f1'] for m in fold_reports)/folds,
                           'pooled_oof_metrics':metrics(y.tolist(),oof)}
                    trials.append(trial)
                    save_csv(destination/f'cv_predictions_config_{index}.csv',[{'session_id':sid,'true_score':int(t),'predicted_score':p} for sid,t,p in zip(ids,y,oof)])
                best=min(trials,key=cv_rank)['configuration_index']
            else: best=config['default_configuration_index']
            save_json(destination/'cv.json',{'trials':trials,'selected_configuration':best,'fold_count':folds,
                        'limitations':'CV selected using these folds; OOF summaries are not nested/unbiased model-selection estimates'})
            model=fit_candidate(algorithm,space[best],x,y,config['seed'])
            with (destination/'pipeline.joblib').open('xb') as stream: joblib.dump(model,stream)
            with (destination/'preprocessor.joblib').open('xb') as stream: joblib.dump(model['pipeline'][:-1],stream)
            with (destination/'estimator.joblib').open('xb') as stream: joblib.dump(model['pipeline'].named_steps['model'],stream)
            metadata={'domain':d,'algorithm':algorithm,'features':FEATURES[d],'target':d,'ordinal_classes':model['ordinal_classes'],
                      'configuration_index':best,'params':space[best],'training_ids':ids,'unsupported_train_classes':sorted(set(range(5))-set(model['ordinal_classes']))}
            save_json(destination/'metadata.json',metadata)
            summaries.append({'domain':d,'algorithm':algorithm,'configuration_index':best,'folds':folds,
                              'CV_MAE':trials[best]['mean_fold_MAE'] if folds else '',
                              'CV_macro_f1':trials[best]['mean_fold_macro_f1'] if folds else ''})
            print('Fitted',d,algorithm,flush=True)
    save_csv(folder/'cv_comparison.csv',summaries)
    stage_seal(folder,'TRAINING_COMPLETE.json')
    return summaries


def verify_training(run):
    folder=Path(run)/'training'
    verify_hashes(folder,read_json(folder/'TRAINING_COMPLETE.json')['files'])


def system_report(tables, ids, predictions, replicates):
    _,targets,_=tables; rows=[]; errors=[]
    for sid in ids:
        values=[predictions.get(d,{}).get(sid) for d in DOMAINS]
        complete=all(v is not None for v in values)
        mps=sum(values)/16 if complete else None
        reference=float(targets[sid]['demo_MPS'])
        if complete: errors.append(abs(mps-reference))
        rows.append({'session_id':sid,**{f'{d}_predicted':v for d,v in zip(DOMAINS,values)},
                     **{f'{d}_reference':ordinal(targets[sid][d]) for d in DOMAINS},
                     'predicted_demo_MPS':mps,'reference_demo_MPS':reference,'complete':complete})
    return rows,{'sessions':len(ids),'complete_sessions':len(errors),'complete_session_coverage':len(errors)/len(ids),
                 'MPS_MAE':sum(errors)/len(errors) if errors else None,
                 'MPS_MAE_session_bootstrap_95':mean_interval(errors,replicates=replicates),
                 'interpretation':'Demo index agreement only; conditional simulator-session uncertainty'}


def rule_baseline(dataset,run,tables,split,destination):
    # Import exactly the dataset's pinned, unchanged engine and archived raw adapter.
    engine_path=dataset/'code/pp1_motor_engine.py'
    if sha(engine_path)!=ENGINE_HASH: raise ValueError('Baseline engine changed')
    spec=importlib.util.spec_from_file_location('pp1_baseline_engine',engine_path)
    engine=importlib.util.module_from_spec(spec); spec.loader.exec_module(engine)
    spec=importlib.util.spec_from_file_location('pp1_archived_adapter',dataset/'code/pp1_v3.py')
    adapter=importlib.util.module_from_spec(spec); spec.loader.exec_module(adapter)
    ids=sorted(sid for sid,r in tables[2].items() if r['split']==split)
    snapshot=read_json(run/'dataset_snapshot.json')['files']; predictions={d:{} for d in DOMAINS}
    for sid in ids:
        verify_hashes(dataset,{k:v for k,v in snapshot.items() if k.startswith(f'cases/{sid}/')})
        result=adapter.assess(engine,adapter.read_raw(dataset/'cases'/sid,read_json(dataset/'config.json')))
        for d in DOMAINS: predictions[d][sid]=result['scores'][d]
    details={d:metrics([ordinal(tables[1][sid][d]) for sid in ids],[predictions[d][sid] for sid in ids]) for d in DOMAINS}
    rows,system=system_report(tables,ids,predictions,2000)
    save_csv(destination/'rule_baseline_predictions.csv',rows)
    save_json(destination/'rule_baseline.json',{'label':'deterministic target-defining consistency baseline; not learned accuracy',
                                             'domains':details,'system':system})
    if any(m['exact_accuracy'] != 1 for m in details.values()): raise ValueError('Engine no longer reproduces dataset labels')


def evaluate_candidates(run,tables,split,destination):
    import joblib
    config=read_json(run/'search_spaces.json'); comparisons=[]; predictions={d:{} for d in DOMAINS}
    for d in DOMAINS:
        ids,x,y=matrix(tables,d,split)
        for algorithm in config['spaces']:
            model=joblib.load(run/'training'/d/algorithm/'pipeline.joblib')
            p=predict(model,x); report=metrics(y.tolist(),p,True,config['bootstrap_replicates'])
            output=destination/d/algorithm; output.mkdir(parents=True)
            save_json(output/'metrics.json',report)
            save_csv(output/'confusion_matrix.csv',[{'true_score':i,**{f'predicted_{j}':n for j,n in enumerate(row)}} for i,row in enumerate(report['confusion_matrix'])])
            save_csv(output/'predictions.csv',[{'session_id':sid,'reference_score':int(a),'predicted_score':b} for sid,a,b in zip(ids,y,p)])
            comparisons.append({'domain':d,'algorithm':algorithm,**{k:report[k] for k in ['n','exact_accuracy','within_1_accuracy','macro_f1','balanced_accuracy','MAE']},
                                'class_support':json.dumps(report['class_support']), 'uncertainty':json.dumps(report['uncertainty'])})
            predictions[d][algorithm]=dict(zip(ids,p))
    save_csv(destination/'comparison.csv',comparisons)
    return comparisons,predictions,ids


def validate(run,approved=False):
    if not approved: raise PermissionError('Training/validation approval is required')
    run=Path(run); dataset,tables=stage_data(run); verify_training(run)
    destination=run/'validation'; destination.mkdir(exist_ok=False)
    rows,predictions,ids=evaluate_candidates(run,tables,'validation',destination)
    order=read_json(run/'search_spaces.json')['algorithm_tie_order']
    selected={d:min([r for r in rows if r['domain']==d],key=lambda r:validation_rank(r,order))['algorithm'] for d in DOMAINS}
    system_rows,report=system_report(tables,ids,{d:predictions[d][selected[d]] for d in DOMAINS},2000)
    save_csv(destination/'selected_system_predictions.csv',system_rows)
    save_json(destination/'selected_system.json',report)
    save_json(destination/'selection.json',{'selected':selected,'rule':'validation MAE, macro F1, fixed RF/SVM/XGB/CatBoost order',
                                           'no_refit':True,'warning':'Selection-set performance is optimistic; final test remains unopened'})
    rule_baseline(dataset,run,tables,'validation',destination)
    stage_seal(destination,'VALIDATION_COMPLETE.json')
    return selected


def freeze(run):
    run=Path(run); stage_data(run); verify_training(run)
    verify_hashes(run/'validation',read_json(run/'validation/VALIDATION_COMPLETE.json')['files'])
    selected=read_json(run/'validation/selection.json')['selected']
    if len(list((run/'training').glob('*/*/pipeline.joblib'))) != 16:
        raise ValueError('All 16 fitted candidates are required before freeze')
    selected_manifest={}
    for d,algorithm in selected.items():
        relative=f'training/{d}/{algorithm}/pipeline.joblib'
        selected_manifest[d]={'algorithm':algorithm,'pipeline':relative,'pipeline_sha256':sha(run/relative),
                              'features':FEATURES[d],'target':d,
                              'metadata':read_json(run/'training'/d/algorithm/'metadata.json')}
    save_json(run/'selected_models.json',selected_manifest)
    files={p.relative_to(run).as_posix():sha(p) for p in run.rglob('*') if p.is_file()}
    save_json(run/'FROZEN.json',{'files':files,'selected':selected,'candidate_count':16,
              'test_predictions_accessed':False,'refit_after_validation':False,'claim':'synthetic demo-label imitation'})
    return selected


def claim_final_test(registry,key,run):
    target=Path(registry)/key
    target.mkdir(parents=True,exist_ok=False)
    save_json(target/'STARTED.json',{'run':str(run),'policy':'Single-use claim; failure does not permit silent retry'})
    return target


def final_test(run,registry,approved=False):
    if not approved: raise PermissionError('Separate final-test approval is required')
    run=Path(run); dataset,tables=stage_data(run)
    frozen=read_json(run/'FROZEN.json'); verify_hashes(run,frozen['files'])
    if frozen['selected'] != read_json(run/'validation/selection.json')['selected']:
        raise ValueError('Frozen selection differs from validation decision')
    registry=Path(registry).resolve()
    if not registry.is_relative_to(Path('/content/drive/MyDrive').resolve()):
        raise ValueError('Final-test registry must persist on Google Drive MyDrive')
    # Dataset identity is stable across new run IDs: accidental new-run retries are blocked too.
    key=sha(dataset/'sessions.jsonl')
    claim=claim_final_test(registry,key,run)
    destination=run/'test'; destination.mkdir(exist_ok=False)
    rows,predictions,ids=evaluate_candidates(run,tables,'test',destination)
    selected=frozen['selected']
    system_rows,report=system_report(tables,ids,{d:predictions[d][selected[d]] for d in DOMAINS},2000)
    save_csv(destination/'selected_system_predictions.csv',system_rows)
    save_json(destination/'selected_system.json',report)
    rule_baseline(dataset,run,tables,'test',destination)
    save_json(destination/'final_report.json',{'candidate_metrics':rows,'selected_algorithms':selected,'system':report,
              'claim':'synthetic demo-label accuracy only; NOT clinical severity accuracy',
              'frozen_manifest_sha256':sha(run/'FROZEN.json'),'small_sample_warning':'36 test sessions; inspect support and intervals; no test-driven tuning'})
    stage_seal(destination,'TEST_COMPLETE.json')
    save_json(claim/'COMPLETE.json',{'run':str(run),'report_sha256':sha(destination/'final_report.json')})
    return report


def backup(run,destination):
    require_colab()
    run,destination=Path(run).resolve(),Path(destination).resolve()
    if destination.exists(): raise FileExistsError(destination)
    if not destination.is_relative_to(Path('/content/drive/MyDrive').resolve()): raise ValueError('Backup must be on Drive')
    destination.parent.mkdir(parents=True,exist_ok=True)
    shutil.copytree(run,destination)
    verify_hashes(destination,{p.relative_to(run).as_posix():sha(p) for p in run.rglob('*') if p.is_file()})
    return str(destination)
