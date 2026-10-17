#!/usr/bin/env python3
"""Generate small FAKE PADS-shaped files to exercise software, NOT for research conclusions."""
import argparse, json
from pathlib import Path
import numpy as np

TASKS=['Relaxed','RelaxedTask','StretchHold']
CHANNELS=['Time','Accelerometer_X','Accelerometer_Y','Accelerometer_Z',
          'Gyroscope_X','Gyroscope_Y','Gyroscope_Z']
UNITS=['s','g','g','g','rad/s','rad/s','rad/s']


def make(root,n=48):
    root=Path(root)
    (root/'patients').mkdir(parents=True,exist_ok=True)
    (root/'movement'/'timeseries').mkdir(parents=True,exist_ok=True)
    gen=np.random.default_rng(2026)
    for p in range(1,n+1):
        pid=f'{p:03}'
        label=1 if p%2 else 0
        condition="Parkinson's" if label else 'Healthy'
        (root/'patients'/f'patient_{pid}.json').write_text(
            json.dumps({'resource_type':'patient','id':pid,'condition':condition},indent=2))
        sessions=[]
        for task in TASKS:
            records=[]
            for wrist in ['LeftWrist','RightWrist']:
                samples=2048 if task=='Relaxed' else 1024
                t=np.arange(samples)/100
                freq=4.5 if label else 2.7
                amp=.07 if label else .016
                acceleration=gen.normal(0,.012,size=(samples,3))
                acceleration[:,2]+=1.0
                acceleration[:,0]+=amp*np.sin(2*np.pi*freq*t)
                gyroscope=gen.normal(0,.016,size=(samples,3))
                gyroscope[:,1]+=(.08 if label else .025)*np.cos(2*np.pi*freq*t)
                matrix=np.column_stack([t,acceleration,gyroscope])
                name=f'timeseries/{pid}_{task}_{wrist}.txt'
                np.savetxt(root/'movement'/name,matrix,delimiter=',',fmt='%.10f')
                records.append({'device_location':wrist,'channels':CHANNELS,'units':UNITS,'file_name':name})
            sessions.append({'record_name':task,'rows':samples,'records':records})
        (root/'movement'/f'observation_{pid}.json').write_text(json.dumps({
            'resource_type':'observation','subject_id':pid,'sampling_rate':100,'session':sessions
        },indent=2))
    # Strict label handling sanity check: synthetic third diagnosis not included.
    (root/'patients'/'patient_999.json').write_text(json.dumps({'id':'999','condition':'Essential Tremor'}))
    print(f'Created FAKE dataset: {n} participants; 1 other diagnosis should be excluded. {root}')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',default='data/synthetic_demo');p.add_argument('--n',type=int,default=48)
    a=p.parse_args();make(a.root,a.n)
