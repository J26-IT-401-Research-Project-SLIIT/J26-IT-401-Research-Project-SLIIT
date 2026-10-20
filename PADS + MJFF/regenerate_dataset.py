#!/usr/bin/env python3
"""Create a fully synthetic PP1 v2 dataset with transparent public-reference provenance.

This is NOT a clinically validated Parkinson dataset. It uses aggregate gait
reference statistics from a user-provided, publicly-described WearGait-derived
summary CSV. Other task patterns are literature-guided simulations, NOT fitted
from downloaded raw records of controls or patients.
"""
from __future__ import annotations

import csv
import io
import json
import hashlib
import math
import argparse
import sys
from pathlib import Path
from datetime import date
from zipfile import ZipFile, ZIP_DEFLATED
import numpy as np

ROOT_NAME = 'PP1_Public_Reference_Informed_Synthetic_v2'
FS = 100
DURATIONS = {'TS':10,'BS':10,'RS':10,'GI':180}
SOURCE_PAGE = {
    'PADS':'https://physionet.org/content/parkinsons-disease-smartwatch/1.0.0/',
    'WearGaitPD':'https://www.nature.com/articles/s41597-026-06806-2',
    'BrazilFTT':'https://doi.org/10.6084/m9.figshare.26940823.v1',
    'mPower':'https://www.synapse.org/mPower',
    'ArmTremor':'https://zenodo.org/records/4772130',
}


def rows_csv(records, headers=None):
    out=io.StringIO()
    if not records and headers is None: raise ValueError('No records')
    cols=headers if headers is not None else list(records[0].keys())
    w=csv.DictWriter(out, fieldnames=cols, extrasaction='ignore', lineterminator='\n')
    w.writeheader();w.writerows(records)
    return out.getvalue().encode('utf-8')


def xyz_csv(t, xyz, headers):
    assert xyz.shape==(len(t),3)
    # numpy formatting is much faster than CSV per-row Python loops
    buff=io.BytesIO(); buff.write((','.join(headers)+'\n').encode())
    v=np.column_stack((t, xyz))
    np.savetxt(buff, v, fmt='%.6f', delimiter=',')
    return buff.getvalue()


def load_gait_reference(path):
    with path.open(newline='',encoding='utf-8-sig') as f:
        rows=list(csv.DictReader(f))
    by_group={}
    for label in ('0','1'):
        sampled=[r for r in rows if r['label']==label]
        assert len(sampled)>=10
        frequency=np.array([float(r['gait_dominant_freq_hz']) for r in sampled])
        dynamic=np.array([float(r['acc_dynamic_rms_ms2']) for r in sampled])
        valid=np.isfinite(frequency)&np.isfinite(dynamic)&(frequency>0)&(dynamic>0)
        by_group[label]={'n':int(valid.sum()),'frequency':frequency[valid], 'dynamic_rms':dynamic[valid],
          'freq_quartiles_hz':np.quantile(frequency[valid],[.1,.25,.5,.75,.9]).round(5).tolist(),
          'dynamic_rms_quartiles_ms2':np.quantile(dynamic[valid],[.1,.25,.5,.75,.9]).round(5).tolist()}
    return by_group


def choose_levels(group, rng, count):
    # Cohort shapes are explicit simulator choices, not observed prevalence.
    # Healthy-like recordings sometimes have impaired movement values, and vice versa.
    levels=[]
    for i in range(count):
        if group=='healthy_like':
            latent=np.clip(rng.beta(1.15,5.0)*5.0,0,4)
        else:
            latent=np.clip(rng.beta(2.1,2.6)*5.0,0,4)
        # cross-domain variation: weakly correlated, not identical tests
        motor=np.clip(np.rint(latent+rng.normal(0,0.9,4)),0,4).astype(int)
        if group=='symptom_pattern_like' and rng.random()<.17:
            motor[rng.integers(0,4)]=0
        levels.append(motor)
    return levels


def synth_signals(levels, group, rng, gait_ref):
    ts_level,bs_level,rs_level,gi_level=map(int,levels)
    t10=np.arange(FS*10,dtype=np.float64)/FS
    t180=np.arange(FS*180,dtype=np.float64)/FS

    # Tremor oscillation: the frequency interval is literature-guided; amplitudes
    # below are uncalibrated simulation parameters, not clinical severity cutoffs.
    amp=np.array([.055,.25,.47,.78,1.32])[ts_level]*rng.uniform(.88,1.18)
    freq=rng.uniform(4.0,6.7) if ts_level>=1 else rng.uniform(6.0,10.0)
    phase=rng.uniform(0,2*np.pi)
    mod=.85+.16*np.sin(2*np.pi*rng.uniform(.17,.45)*t10+rng.uniform(0,2*np.pi))
    tremor=amp*mod*np.sin(2*np.pi*freq*t10+phase)
    ts_xyz=np.column_stack((.88*tremor, .37*amp*np.cos(2*np.pi*freq*t10+phase),
                            9.81+.68*tremor))
    ts_xyz+=rng.normal(0,.009,(len(t10),3))

    # Capacitive-touch-compatible timestamps only; no finger amplitude data.
    n=int([40,33,25,17,9][bs_level]+rng.integers(-2,3))
    n=max(4,n)
    jitter=np.array([.004,.011,.025,.045,.067])[bs_level]
    # A gradually increasing tap interval captures simulated fatigue; not label truth.
    trend=np.linspace(1-.06*bs_level/4, 1+.15*bs_level/4, n-1)
    increments=np.maximum(.055,trend*(9.35/(n-1))+rng.normal(0,jitter,n-1))
    taps=np.r_[.18,.18+np.cumsum(increments)]
    taps=.18+(taps-.18)*(rng.uniform(9.25,9.65)-.18)/(taps[-1]-.18)
    taps=np.round(taps,6)
    assert len(taps)==n and np.all(np.diff(taps)>0) and taps[-1]<10

    # Gyroscope measures active wrist angular velocity only; NOT passive rigidity.
    ang=np.array([3.30,2.09,1.50,.95,.33])[rs_level]*rng.uniform(.88,1.12)
    motion_hz=rng.uniform(.86,1.32)
    envelope=(.88+.12*np.cos(2*np.pi*t10/(4.8+rng.uniform(-1,1))))
    speed=ang*envelope*np.sin(2*np.pi*motion_hz*t10+rng.uniform(0,2*np.pi))
    rs_xyz=np.column_stack((speed, .07*ang*np.cos(2*np.pi*motion_hz*t10), .04*speed))
    rs_xyz+=rng.normal(0,.012,rs_xyz.shape)

    # Empirical group-level gait frequency & wrist motion scale originate from
    # the locally available 51-person extracted-summary table; not copied traces.
    ref=gait_ref['0' if group=='healthy_like' else '1']
    # Smoothly combine a sampled real-reference cadence with a simulator severity
    # target; the mixing weight is a modelling assumption, NOT derived from data.
    ref_freq=float(rng.choice(ref['frequency']))
    base_target=np.array([2.04,1.75,1.47,1.18,.86])[gi_level]
    gait_hz=float(np.clip(.72*base_target+.28*ref_freq+rng.normal(0,.03),.7,2.5))
    ref_rms=float(rng.choice(ref['dynamic_rms']))
    magnitude=np.clip(1.10*ref_rms,1.0,5.5)
    modulation=1+.08*np.sin(2*np.pi*t180/21)+.03*np.sin(2*np.pi*t180/7.2)
    drift=.009*np.sin(2*np.pi*t180/75)
    phi=2*np.pi*(gait_hz*t180 + .009*75/(2*np.pi)*(1-np.cos(2*np.pi*t180/75)))
    # Gradual rhythm irregularity without introducing missing sensor samples.
    phi+=.025*gi_level/4*np.sin(2*np.pi*t180/13.0)
    gi_xyz=np.column_stack((.19*magnitude*modulation*np.sin(phi+.24),
                            .14*magnitude*modulation*np.cos(phi+.8),
                            9.81+.83*magnitude*modulation*np.sin(phi)))
    gi_xyz+=rng.normal(0,.038,(len(t180),3))

    return t10,ts_xyz,taps,rs_xyz,t180,gi_xyz,{
      'ts_frequency_hz_simulated':round(freq,5),'ts_amplitude_ms2_simulated':round(amp,5),
      'bs_tap_count_simulated':int(n),'rs_peak_amp_rad_s_simulated':round(ang,5),
      'gait_base_freq_from_reference_hz':round(ref_freq,5),
      'gait_simulated_freq_hz':round(gait_hz,5),
      'gait_reference_dynamic_rms_ms2':round(ref_rms,5)}


def generate(output, summary_output, gait_feature_file, engine_file, n_per_group=100,seed=20261021):
    if output.exists() or summary_output.exists():
        raise FileExistsError('Refusing to overwrite existing files')
    import importlib.util
    spec=importlib.util.spec_from_file_location('pp1_engine_for_v2',str(engine_file))
    engine=importlib.util.module_from_spec(spec);spec.loader.exec_module(engine)
    assert engine.DURATIONS==DURATIONS
    rng=np.random.default_rng(seed)
    ref=load_gait_reference(gait_feature_file)
    distributions=[('healthy_like',choose_levels('healthy_like',rng,n_per_group)),
                   ('symptom_pattern_like',choose_levels('symptom_pattern_like',rng,n_per_group))]
    session_records=[];feature_records=[];fusion_records=[];n_labels={g:0 for g,_ in distributions}
    group_splits={g:['train']*70+['validation']*15+['test']*15 for g,_ in distributions}
    if n_per_group!=100:
        group_splits={g:['train']*int(.7*n_per_group)+['validation']*int(.15*n_per_group)+['test']*(n_per_group-int(.85*n_per_group)) for g,_ in distributions}
    for s in group_splits.values():rng.shuffle(s)

    with ZipFile(output,'w',ZIP_DEFLATED,compresslevel=3,allowZip64=True) as z:
      for group,levs in distributions:
       for idx,levels in enumerate(levs,1):
        sid=('HC' if group=='healthy_like' else 'SYM')+f'_SYN_{idx:03d}'
        t10,ts,taps,rs,t180,gi,params=synth_signals(levels,group,rng,ref)
        result=engine.assess_motor(ts,taps,rs,gi,fs=FS,mode='demo')
        real=engine.assess_motor(ts,taps,rs,gi,fs=FS,mode='real')
        assert real['scores'] is None and real['mps'] is None
        scores=result['scores']
        assert all(isinstance(scores[k],int) and 0<=scores[k]<=4 for k in DURATIONS)
        assert abs(result['mps']-sum(scores.values())/16)<1e-10
        n_labels[group]+=1
        prefix=f'{ROOT_NAME}/cases/{sid}/'
        z.writestr(prefix+'TS_10s.csv',xyz_csv(t10,ts,['time_s','ax_ms2','ay_ms2','az_ms2']))
        z.writestr(prefix+'BS_10s.csv',rows_csv([{'tap_time_s':f'{v:.6f}'} for v in taps]))
        z.writestr(prefix+'RS_10s.csv',xyz_csv(t10,rs,['time_s','gx_rad_s','gy_rad_s','gz_rad_s']))
        z.writestr(prefix+'GI_180s.csv',xyz_csv(t180,gi,['time_s','ax_ms2','ay_ms2','az_ms2']))
        session_records.append({'session_id':sid,'synthetic_cohort_profile':group,
            'real_patient':False,'clinical_diagnosis':'NOT_AVAILABLE','clinical_severity_labels_available':False,
            'split':group_splits[group][idx-1], 'data_origin':'fully_synthetic',
            'source_context':'PADS|WearGaitPD|BrazilFTT|mPower|ArmTremor',
            'gait_numeric_parameter_status':'empirical_group_reference_plus_simulation',
            'other_numeric_parameter_status':'literature_context_plus_simulation',
            'sampling_hz':FS,'ts_duration_s':10,'bs_duration_s':10,'rs_duration_s':10,'gi_duration_s':180,
            **{f'{k}_simulated_pattern_level':int(levels[j]) for j,k in enumerate(DURATIONS)},**params})
        f=result['features']
        feature_records.append({'session_id':sid,'synthetic_cohort_profile':group,'split':group_splits[group][idx-1],
             'ts_tremor_rms_ms2':round(f['TS']['tremor_rms_ms2'],8),
             'bs_tap_count':f['BS']['tap_count'], 'bs_tap_rate_per_min':round(f['BS']['tap_rate_per_min'],7),
             'bs_tap_interval_std_s':round(f['BS']['tap_interval_std_s'],8),
             'rs_mean_gyro_peak_rad_s':round(f['RS']['mean_gyro_peak_rad_s'],8) if f['RS']['mean_gyro_peak_rad_s'] is not None else None,
             'rs_movement_peaks':f['RS']['movement_peaks'],
             'gi_step_candidates':f['GI']['step_candidates'],
             'gi_cadence_steps_min':round(f['GI']['cadence_steps_min'],7),
             'gi_step_interval_std_s':round(f['GI']['step_interval_std_s'],8) if f['GI']['step_interval_std_s'] is not None else None})
        fusion_records.append({'session_id':sid,'synthetic_cohort_profile':group,
             'split':group_splits[group][idx-1], 'TS_demo':scores['TS'],'BS_demo':scores['BS'],
             'RS_active_mobility_proxy_demo':scores['RS'],'GI_demo':scores['GI'],
             'demo_MPS_0_1':round(result['mps'],6),
             'labels_are_clinician_rated':False,
             'score_provenance':'PP1 temporary demo thresholds; NOT clinical ground truth',
             'Parkinson_probability': 'NOT_ESTIMATED'})
        if idx%25==0:print(group,idx,flush=True)
      z.writestr(f'{ROOT_NAME}/sessions_and_provenance.csv',rows_csv(session_records))
      z.writestr(f'{ROOT_NAME}/features_extracted_by_existing_PP1_engine.csv',rows_csv(feature_records))
      z.writestr(f'{ROOT_NAME}/demo_rule_scores_for_fusion.csv',rows_csv(fusion_records))
      z.writestr(f'{ROOT_NAME}/public_source_reference_status.csv',rows_csv([
       {'source':'PADS','public_url':SOURCE_PAGE['PADS'],'domain':'TS and RS wrist IMU; HC and PD',
        'access':'public download; CC BY-NC-SA 4.0','numeric_fitting_status':'NOT FITTED: source task/unit context only',
        'protocol_difference':'Apple Watch tasks 10–20 s; no direct rigidity rating from wrist gyroscope'},
       {'source':'WearGaitPD','public_url':SOURCE_PAGE['WearGaitPD'],'domain':'GI gait, HC and PD',
        'access':'Synapse registration required; CC BY 4.0','numeric_fitting_status':'USED: aggregate frequency / RMS from supplied extracted 51-person wrist-feature table',
        'protocol_difference':'Short clinical gait tasks; 180s is simulated extrapolation; not measured on PP1 device'},
       {'source':'BrazilFTT','public_url':SOURCE_PAGE['BrazilFTT'],'domain':'BS touchscreen tap timestamps, healthy adults',
        'access':'publicly described Figshare dataset','numeric_fitting_status':'NOT FITTED: methodological reference only',
        'protocol_difference':'30-second smartphone screen vs 10-second capacitive touch'},
       {'source':'mPower','public_url':SOURCE_PAGE['mPower'],'domain':'BS tapping, HC and PD',
        'access':'Synapse registration and data use agreement required','numeric_fitting_status':'NOT FITTED: publication-level context only',
        'protocol_difference':'20-second dual-target tapping, access-gated; cannot infer PP1 amplitude'},
       {'source':'ArmTremor','public_url':SOURCE_PAGE['ArmTremor'],'domain':'TS accelerometer; no tremor, essential tremor, PD tremor',
        'access':'public download; CC BY 4.0','numeric_fitting_status':'NOT FITTED: domain reference only',
        'protocol_difference':'Different sensors and protocol; not directly copied or re-calibrated'}
      ]))
      manifest={'name':ROOT_NAME,'version':'2.0','sessions':len(session_records),'synthetic_healthy_like':n_labels['healthy_like'],
        'synthetic_symptom_pattern_like':n_labels['symptom_pattern_like'],
        'sensor_csv_files':len(session_records)*4,'sampling_hz':FS,'durations_s':DURATIONS,
        'group_profiles':'simulated profiles, not real diagnoses',
        'clinical_0_4_ground_truth':False,'MPS_is_PD_probability':False,
        'gait_reference':{'source':'user-provided GI_Wrist_Features_51_Participants.csv; includes wear-gait-derived measures',
                          'control_records':ref['0']['n'],'pd_labeled_records':ref['1']['n'],
                          'healthy_gait_frequency_quantiles_hz':ref['0']['freq_quartiles_hz'],
                          'symptomatic_gait_frequency_quantiles_hz':ref['1']['freq_quartiles_hz']},
        'other_domains':'research-source-informed simulated mechanics; not numerically fitted to actual HC/PD recordings',
        'train_validation_test':'synthetic session identifiers; participant realism and clinical generalization not established',
        'random_seed':seed,'source_urls':SOURCE_PAGE}
      z.writestr(f'{ROOT_NAME}/dataset_manifest.json',json.dumps(manifest,indent=2,ensure_ascii=False))
      z.writestr(f'{ROOT_NAME}/README_AND_LIMITATIONS.md',readme(len(session_records),ref))
      z.writestr(f'{ROOT_NAME}/MODEL_TRAINING_WARNING.md',train_warning())
    with ZipFile(output) as z:
        assert z.testzip() is None
        filelist=z.namelist()
        assert sum(x.endswith('GI_180s.csv') for x in filelist)==len(session_records)
        assert sum(x.endswith('TS_10s.csv') for x in filelist)==len(session_records)
        assert sum(x.endswith('BS_10s.csv') for x in filelist)==len(session_records)
        assert sum(x.endswith('RS_10s.csv') for x in filelist)==len(session_records)
    summary_output.write_bytes(rows_csv(fusion_records))
    return manifest


def readme(count,ref):
    return f'''# PP1 Reference-informed Synthetic Dataset v2\n\n**SIMULATED. NOT A DATASET OF REAL HEALTHY/PARKINSON'S PATIENTS.**\n\nThis archive contains **{count} independent fully synthetic assessment sessions**:\n- Healthy-like motor patterns (simulated): {count//2}\n- Parkinsonian-symptom-like motor patterns (simulated): {count//2}\n- Four CSVs per session: TS 10s accel, BS 10s capacitive-touch timestamps, RS 10s active wrist gyro, GI 180s wrist accel.\n- Sample rate: 100 Hz for accelerometer/gyroscope; taps are event timestamps.\n- Acceleration units: m/s^2, gyro: rad/s. Exact original PP1 software input column names are used.\n- No real person or patient is represented by a case ID. A case is not a clinically diagnosed patient.\n\n## Reference provenance (critical)\n\nResearch source descriptions support the choice of tasks and the broad movement modalities, but **do not fully determine the synthetic signals**.\n\n* **GI numeric anchors actually used**: a user-provided, previously extracted 51-person wrist gait feature table. It has {ref['0']['n']} healthy-control and {ref['1']['n']} PD-labeled records. Group-specific gait-frequency and dynamic-acceleration RMS measurements were summarized and used as contextual numeric anchors for simulated 180-second gait. Neither original real signals nor real participant identifiers were copied. This table is associated with a WearGait-PD derived workflow, but access rights should be checked before redistribution.\n* **TS tremor / RS wrist mobility**: PADS and Arm Tremor public repositories provide real clinical control/symptomatic modality references, but their raw participant-level recordings were NOT downloaded or numerically fitted in this build. Frequency/amplitude and movement-velocity rules are **simulation assumptions**, not extracted real-dataset distributions. RS is NOT a measurement of clinical rigidity.\n* **BS tapping**: healthy smartphone finger tapping dataset (30s) and mPower (20s healthy/PD with access registration) are literature references. Neither dataset was accessed for numeric BS fitting; the 10-second capacitive timestamps were simulated with hand-selected rate/irregularity settings. Finger tapping amplitude is unmeasured.\n\nSee `public_source_reference_status.csv` and verified URLs. The fact that clinical datasets are listed as sources does NOT mean a case is derived from a real patient's four tests.\n\n## Data format\n\nEach `cases/HC_SYN_001/` or `cases/SYM_SYN_001/` has:\n\n| File | Header | Samples / range |\n|---|---|---|\n| `TS_10s.csv` | time_s,ax_ms2,ay_ms2,az_ms2 | 1000 rows, 0–9.99s |\n| `BS_10s.csv` | tap_time_s | timestamps in [0,10) |\n| `RS_10s.csv` | time_s,gx_rad_s,gy_rad_s,gz_rad_s | 1000 rows, 0–9.99s |\n| `GI_180s.csv` | time_s,ax_ms2,ay_ms2,az_ms2 | 18000 rows, 0–179.99s |\n\n`features_extracted_by_existing_PP1_engine.csv` contains actual engine feature extraction results. `demo_rule_scores_for_fusion.csv` contains pseudo-labels from the **uncalibrated existing PP1 software rules**. Formula `MPS=(TS+BS+RS+GI)/16` is a temporary normalized demo index, NOT the probability or severity of Parkinson's disease. In real-device mode the existing engine must continue outputting `scores=None, mps=None` pending calibration.\n\nThe unverified hardware protocol should be mapped through a future adapter rather than adjusting hardware or inventing extra sensors.\n\n## Research and training boundaries\n\nThis is suitable for software integration and controlled simulator experiments. It cannot substantiate clinical exact severity accuracy, independent patient diagnosis performance, or 85% real-world MPS accuracy. Any Random Forest/XGBoost/CNN trained to reproduce these scores would learn the simulator and the software thresholds. Models trained on PADS 9-second wrist data cannot directly score these four device tasks.\n\n`synthetic_cohort_profile` records simulated group generation, not clinical condition. It must never be reported as an observed medical label.\n\nFor independent evaluation collect real PP1-device signals with clinician-rated reference 0–4 ratings and appropriate oversight; keep training/validation/test subjects disjoint.\n\n## Public data sources\n\n* PADS, PhysioNet: {SOURCE_PAGE['PADS']} (CC BY-NC-SA 4.0)\n* WearGait-PD: {SOURCE_PAGE['WearGaitPD']} (Synapse registration / CC BY 4.0)\n* Brazilian finger tapping healthy reference: {SOURCE_PAGE['BrazilFTT']}\n* mPower motor tasks: {SOURCE_PAGE['mPower']} (Synapse access conditions)\n* Arm Tremor Database: {SOURCE_PAGE['ArmTremor']} (CC BY 4.0)\n\n## Caution on licenses\n\nThis package includes independently generated simulated numerical values and derived group summary statistics, not redistributed patient-level datasets. Source dataset access and attribution terms must be honored when you later retrieve real records. Preserve this provenance file in any academic use.\n'''


def train_warning():
    return '''# Before model training\n\n* **Do not label these samples as real patients**. HC-like and symptom-like are simulation profiles.\n* **Do not call these labels clinically rated TS/BS/RS/GI 0–4 values**. Existing engine outputs are temporary rule pseudo-labels.\n* A model can show 95%+ **synthetic-rule imitation accuracy** without having any real-world clinical scoring value; do not advertise it as clinical performance.\n* For a demo, use synthetic subject-disjoint splits and report `synthetic simulator benchmark` as the evaluation.\n* For actual PP1 evaluation use original portable-device data + independent clinician ratings on the same subjects.\n* Missing sensor channels and missing whole tests must be flagged; do not invent absent tap events or label invalid MPS.\n* PADS trained RF/SVM/XGBoost/CNN/Hybrid CNN models are PD-vs-Healthy research classifiers with nonmatching protocols. Keep checkpoints separate.\n'''


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',default='/mnt/data/PP1_Public_Reference_Informed_Synthetic_v2.zip')
    parser.add_argument('--summary',default='/mnt/data/PP1_Public_Reference_Informed_Demo_Scores_v2.csv')
    parser.add_argument('--gait-reference',default='/mnt/data/GI_Wrist_Features_51_Participants.csv')
    parser.add_argument('--engine',default='/mnt/data/_pp1_engine_for_validation.py')
    parser.add_argument('--count-per-group',type=int,default=100)
    parser.add_argument('--seed',type=int,default=20261021)
    args=parser.parse_args()
    output=Path(args.output)
    result=generate(output,Path(args.summary),Path(args.gait_reference),Path(args.engine),args.count_per_group,args.seed)
    print('CREATED',str(output), output.stat().st_size/1e6,'MB',result['sessions'],'sessions')
