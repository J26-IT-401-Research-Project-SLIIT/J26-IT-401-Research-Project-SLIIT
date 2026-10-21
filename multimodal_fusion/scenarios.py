"""Versioned moderate PP1 demonstration, retaining the harder experiment."""
import hashlib,json
from datetime import datetime,timezone
import importlib.metadata
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold,cross_val_score,train_test_split
from sklearn.pipeline import Pipeline,make_pipeline
from sklearn.preprocessing import StandardScaler
from .settings import ARTIFACTS,DATA,FEATURES,LABELS,VERSION,ACTIVE_DATASET,SCOPE
from .synthetic import generate,PROFILES
from .preprocessing import ScorePreprocessor,inspect_dataset

DATA_SEED=1401
FRESH_SEED=1402
NOISE=.50

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def train_pp1():
    from .train import metrics,probabilities
    ARTIFACTS.mkdir(exist_ok=True);DATA.mkdir(exist_ok=True)
    hard_path=DATA/'synthetic_participants.csv'
    if not hard_path.exists():raise RuntimeError('Retained original hard benchmark is required; do not recreate it silently')
    hard_hash=sha(hard_path)
    protocol={'version':VERSION,'scope':SCOPE,'created_utc':datetime.now(timezone.utc).isoformat(),
        'purpose':'User-approved moderate-overlap synthetic PP1 demonstration; harder scenario preserved',
        'noise_sd':NOISE,'data_seed':DATA_SEED,'fresh_seed':FRESH_SEED,'cohort_n':1000,
        'split_n':{'train':600,'validation':200,'test':200},'class_distribution':dict.fromkeys(LABELS,200),
        'profiles':PROFILES,'selection':'Highest validation macro F1 among four Random Forest candidates; first candidate breaks exact ties',
        'scenario_selection_disclosure':'Moderate overlap was chosen after a prior simulator sensitivity study, not established from clinical data',
        'retained_hard_sha256':hard_hash,'test_used_for_configuration_selection':False}
    (ARTIFACTS/'moderate_training_protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    frame=generate(1000,DATA_SEED,noise=NOISE,prefix='MOD')
    development,test=train_test_split(frame,test_size=.2,stratify=frame.label,random_state=DATA_SEED)
    training,validation=train_test_split(development,test_size=.25,stratify=development.label,random_state=DATA_SEED+1)
    for name,part in [('train',training),('validation',validation),('test',test)]:frame.loc[part.index,'split']=name
    frame['scenario']='moderate_overlap';frame['generator_noise_sd']=NOISE
    preparation=inspect_dataset(frame)
    frame.to_csv(DATA/ACTIVE_DATASET,index=False)
    # Selection/fitting use training + validation records only. No test labels.
    candidates=[];candidate_rows=[]
    for depth,leaf in [(4,8),(6,5),(8,4),(None,6)]:
        pipe=Pipeline([('score_preprocess',ScorePreprocessor()),('classifier',RandomForestClassifier(
            n_estimators=240,max_depth=depth,min_samples_leaf=leaf,max_features=None,class_weight='balanced',random_state=DATA_SEED,n_jobs=1))])
        cv=cross_val_score(pipe,training[FEATURES],training.label,cv=StratifiedKFold(5,shuffle=True,random_state=DATA_SEED),scoring='f1_macro',n_jobs=1)
        pipe.fit(training[FEATURES],training.label)
        valid=metrics(pipe,validation)
        candidates.append(pipe);candidate_rows.append({'max_depth':depth,'min_samples_leaf':leaf,
            'training_cv_macro_f1_mean':float(cv.mean()),'training_cv_macro_f1_std':float(cv.std()),'validation_macro_f1':valid['macro_f1']})
    chosen=int(np.argmax([row['validation_macro_f1'] for row in candidate_rows]));pipeline=candidates[chosen]
    protocol['selected_index']=chosen;protocol['candidates']=candidate_rows
    (ARTIFACTS/'moderate_training_protocol.json').write_text(json.dumps(protocol,indent=2),encoding='utf-8')
    fresh=generate(500,FRESH_SEED,noise=NOISE,prefix='MOD-FINAL')
    fresh['split']='fresh_final_test';fresh['scenario']='moderate_overlap';fresh['generator_noise_sd']=NOISE
    inspect_dataset(fresh);fresh.to_csv(DATA/'synthetic_moderate_fresh_test.csv',index=False)
    hard=pd.read_csv(hard_path);hard_test=hard[hard.split=='test']
    stress=pd.read_csv(DATA/'synthetic_stress_participants.csv')
    models={'Random Forest fusion':(pipeline,FEATURES)}
    linear=make_pipeline(ScorePreprocessor(),StandardScaler(),LogisticRegression(max_iter=2000,random_state=DATA_SEED))
    linear.fit(training[FEATURES],training.label);models['Logistic regression fusion']=(linear,FEATURES)
    for title,column in zip(['Motor only','Voice only','Handwriting only'],FEATURES):
        single=Pipeline([('score_preprocess',ScorePreprocessor(features=(column,))),('classifier',RandomForestClassifier(n_estimators=160,max_depth=5,min_samples_leaf=8,random_state=DATA_SEED,n_jobs=1))])
        single.fit(training[[column]],training.label);models[title]=(single,[column])
    results={name:metrics(model,test,columns) for name,(model,columns) in models.items()}
    fresh_results={name:metrics(model,fresh,columns) for name,(model,columns) in models.items()}
    hard_results={name:metrics(model,hard_test,columns) for name,(model,columns) in models.items()}
    stress_results={name:metrics(model,stress,columns) for name,(model,columns) in models.items()}
    preprocessor=pipeline.named_steps['score_preprocess']
    preprocessing={'version':'score-schema-v1','fitted_on':'600 training records only','fit_rows':preprocessor.fit_rows_,
        'steps':['Select exactly MPS/VPS/HPS; exclude ID/label/split/provenance/scenario','Canonical feature ordering',
            'Validate numeric types and reject boolean scores','Reject missing, NaN and infinite values','Reject out-of-range scores; do not silently clip',
            'Pass validated 0–1 scores to Random Forest without standard scaling'],
        'training_min':preprocessor.training_min_,'training_max':preprocessor.training_max_,
        'normalization':'Upstream score probabilities already use [0,1]; no fitted tree scaler',
        'imputation':'None: incomplete live requests return no prediction',
        'raw_input_processing':'Image/audio/sensor preprocessing remains in Components 1–3; fusion receives their scores',
        'logistic_baseline':'Score validation then StandardScaler fitted within each training pipeline',
        'data_preparation':preparation}
    bundle={'pipeline':pipeline,'model':pipeline.named_steps['classifier'],'feature_columns':FEATURES,'labels':LABELS,
        'training_background':training[FEATURES].to_numpy(),'seed':DATA_SEED,'scope':SCOPE,'version':VERSION,
        'scenario':'moderate_overlap','noise_sd':NOISE,'dataset_filename':ACTIVE_DATASET,'preprocessing':preprocessing}
    joblib.dump(bundle,ARTIFACTS/'fusion_model.joblib')
    background_rows=set(map(tuple,training[FEATURES].to_numpy()))
    overlap={name:len(background_rows&set(map(tuple,part[FEATURES].to_numpy()))) for name,part in [('validation',validation),('test',test),('fresh_test',fresh),('retained_hard_test',hard_test)]}
    if any(overlap.values()):raise RuntimeError('Exact training-score overlap in an evaluation partition')
    evidence={'version':VERSION,'created_utc':datetime.now(timezone.utc).isoformat(),'scope':SCOPE,'clinical_validation':False,
        'scenario':'moderate_overlap','serving_model':'Random Forest fusion','features':FEATURES,'classes':LABELS,'class_order':LABELS,
        'data':{'n':1000,'per_class':200,'splits':{'train':600,'validation':200,'test':200},'filename':ACTIVE_DATASET,
            'sha256':sha(DATA/ACTIVE_DATASET),'seed':DATA_SEED,'generation':'label-first overlapping logistic-normal simulation',
            'assumed_profile_centers':PROFILES,'logit_noise_sd':NOISE,'not_derived_from_uploaded_patient_data':True},
        'model_selection':{'rule':protocol['selection'],'candidates':candidate_rows,'selected_index':chosen,'test_used':False},
        'results':results,'fresh_results':fresh_results,'fresh_test_n':500,
        'hard_benchmark':{'scope':'Same moderate-trained models evaluated on the retained harder scenario; this benchmark was already inspected',
            'noise_sd':.85,'source_filename':hard_path.name,'source_sha256':hard_hash,'original_file_unchanged':sha(hard_path)==hard_hash,
            'test_n':len(hard_test),'results':hard_results,'historical_hard_model_accuracy':.61},
        'stress_results':stress_results,'stress_scope':'Retained 500-record wider-overlap, imbalanced synthetic stress sample',
        'preprocessing':preprocessing,'leakage_checks':{'training_feature_overlap_with_evaluation':overlap,'fitted_preprocessor_rows':preprocessor.fit_rows_},
        'packages':{name:importlib.metadata.version(name) for name in ['numpy','pandas','scikit-learn','shap','lime','fastapi']}}
    (ARTIFACTS/'evaluation.json').write_text(json.dumps(evidence,indent=2),encoding='utf-8')
    (ARTIFACTS/'preprocessing_report.json').write_text(json.dumps(preprocessing,indent=2),encoding='utf-8')
    for filename,part in [('test_predictions.csv',test),('moderate_fresh_predictions.csv',fresh),('hard_benchmark_predictions.csv',hard_test)]:
        p=probabilities(pipeline,part[FEATURES]);out=part[['synthetic_id','label']].copy();out['prediction']=np.array(LABELS)[p.argmax(axis=1)]
        for i,label in enumerate(LABELS):out['p_'+label]=p[:,i]
        out.to_csv(ARTIFACTS/filename,index=False)
    pd.DataFrame([{'model':name,**{k:v for k,v in result.items() if isinstance(v,(float,int))}} for name,result in results.items()]).to_csv(ARTIFACTS/'model_comparison.csv',index=False)
    cases=[]
    for label in LABELS:
        row=validation[validation.label==label].sort_values('synthetic_id').iloc[0]
        cases.append({'name':f'{label} example','synthetic_label':label,'synthetic_id':row.synthetic_id,'split':'validation',
            'selection':'first ID in validation class; not prediction-selected','scenario':'moderate_overlap','scores':{key:float(row[key]) for key in FEATURES}})
    cases.extend([{'name':'Conflicting inputs','scores':dict(zip(FEATURES,[.90,.20,.55]))},{'name':'Borderline inputs','scores':dict(zip(FEATURES,[.50,.50,.50]))},{'name':'Missing voice','scores':dict(zip(FEATURES,[.75,None,.65]))}])
    (ARTIFACTS/'demo_cases.json').write_text(json.dumps(cases,indent=2),encoding='utf-8')
    plot_scenarios(evidence)
    print(json.dumps({'version':VERSION,'moderate_test':results['Random Forest fusion']['accuracy'],
        'moderate_macro_f1':results['Random Forest fusion']['macro_f1'],'moderate_fresh_test':fresh_results['Random Forest fusion']['accuracy'],
        'retained_hard_benchmark':hard_results['Random Forest fusion']['accuracy'],'preprocessor_fit_rows':preprocessor.fit_rows_,
        'original_hard_unchanged':sha(hard_path)==hard_hash,'selected':candidate_rows[chosen]},indent=2))
    return evidence

def plot_scenarios(evidence):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(8,5),layout='constrained');matrix=evidence['results']['Random Forest fusion']['confusion_matrix'];im=ax.imshow(matrix,cmap='Blues')
    ax.set(xticks=range(5),yticks=range(5),xticklabels=LABELS,yticklabels=LABELS,xlabel='Predicted class',ylabel='Assigned simulated class',title='Moderate-overlap PP1 demonstration · test n=200')
    for i,row in enumerate(matrix):
        for j,value in enumerate(row):ax.text(j,i,str(value),ha='center',va='center',color='white' if value>20 else '#17334b')
    fig.colorbar(im,ax=ax);fig.savefig(ARTIFACTS/'confusion_matrix.png',dpi=150);plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,4.5),layout='constrained');names=list(evidence['results']);values=[evidence['results'][n]['macro_f1'] for n in names]
    ax.barh(names,values,color=['#087e8b']+['#9ab9c3']*(len(names)-1));ax.set(xlim=(0,1),xlabel='Macro F1',title='Moderate scenario · same test for every model');ax.invert_yaxis()
    for i,value in enumerate(values):ax.text(value+.01,i,f'{value:.3f}',va='center')
    fig.savefig(ARTIFACTS/'model_comparison.png',dpi=150);plt.close(fig)

if __name__=='__main__':train_pp1()
