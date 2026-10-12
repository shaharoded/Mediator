"""Sampled input perturbation study using the real Mediator TAK implementations.

Direct numeric RawConcept -> State/Trend only. Original SQLite DB is read-only.
All analysis is restricted to [admission, min(admission+48h, terminal event)].
Derived concepts, contexts, patterns and Transformer predictions are out of scope.
"""
from pathlib import Path
import concurrent.futures
import hashlib
import json
import logging
import os
MEDIATOR_ROOT = Path(__file__).resolve().parents[2]
PERSONAL = MEDIATOR_ROOT.parent
RESULTS_ROOT = Path(os.environ.get('MEDIATOR_ANALYSIS_OUTPUT', str(MEDIATOR_ROOT / 'backend/data/analysis')))
import pickle
import sqlite3
import sys
import time
import numpy as np
import pandas as pd

ROOT = RESULTS_ROOT / 'mediator-stability'
ROOT.mkdir(parents=True, exist_ok=True)
sys.path.insert(0,str(MEDIATOR_ROOT))
from core.mediator import Mediator
from core.tak.repository import set_tak_repository

SCOPE=RESULTS_ROOT/'threshold-proximity/repository_scope.json'
DB=MEDIATOR_ROOT/'backend/data/mediator.db'
REPO=MEDIATOR_ROOT/'backend/data/tak-repo.pkl'
SEED=20261012
CONDITIONS={'noise_1pct':('noise',.01), 'noise_5pct':('noise',.05),
            'remove_5pct':('remove',.05), 'remove_10pct':('remove',.10)}


def connect():
    c=sqlite3.connect(DB.as_uri()+'?mode=ro',uri=True)
    c.execute('pragma query_only=on')
    return c


def initialize():
    global repo, specs, conn, clipper
    logging.disable(logging.CRITICAL)
    with REPO.open('rb') as f: repo=pickle.load(f)
    set_tak_repository(repo)
    specs=json.loads(SCOPE.read_text())['definitions']
    conn=connect()
    # Use the original clipping method without initializing a DB writer.
    clipper=Mediator.__new__(Mediator)
    clipper.global_clippers={'ADMISSION':'START','RELEASE':'END','DEATH':'END'}


def intervals(frame, origin, end):
    if frame.empty: return []
    start=np.maximum((pd.to_datetime(frame.StartDateTime)-origin).dt.total_seconds(),0)
    finish=np.minimum((pd.to_datetime(frame.EndDateTime)-origin).dt.total_seconds(),end)
    rows=sorted((float(a),float(b),str(v)) for a,b,v in zip(start,finish,frame.Value) if b>a)
    if any(rows[i][0]<rows[i-1][1]-1e-7 for i in range(1,len(rows))):
        raise AssertionError('Overlapping intervals within one TAK')
    return rows


def compare(a,b):
    da=sum(y-x for x,y,_ in a); db=sum(y-x for x,y,_ in b)
    overlap=same=0.; i=j=0
    while i<len(a) and j<len(b):
        x=max(a[i][0],b[j][0]); y=min(a[i][1],b[j][1])
        if y>x:
            overlap+=y-x
            if a[i][2]==b[j][2]: same+=y-x
        if a[i][1]<=b[j][1]: i+=1
        else: j+=1
    return np.array([da,db,overlap,same,da+db-overlap],float)


def abstract(raw, spec, clippers, origin, end):
    concept=repo.get(spec['state']).derived_from
    parent=repo.get(concept)
    numeric=parent.apply(raw.copy())
    numeric=clipper._apply_global_clippers(numeric,clippers)
    state=repo.get(spec['state']).apply(numeric.copy())
    state=clipper._apply_global_clippers(state,clippers)
    outputs={'state':intervals(state,origin,end)}
    if spec.get('trend'):
        trend=repo.get(spec['trend']).apply(numeric.copy())
        trend=clipper._apply_global_clippers(trend,clippers)
        outputs['trend']=intervals(trend,origin,end)
    return outputs


def patient_job(item):
    ordinal,pid,stratum,weight=item
    raw=pd.read_sql_query('select PatientId,ConceptName,StartDateTime,EndDateTime,Value '
                          'from InputPatientData where PatientId=? order by StartDateTime,RowId',conn,params=(pid,))
    raw.StartDateTime=pd.to_datetime(raw.StartDateTime)
    raw.EndDateTime=pd.to_datetime(raw.EndDateTime)
    origin=raw.loc[raw.ConceptName=='ADMISSION','StartDateTime'].min()
    clippers=raw[raw.ConceptName.isin(['ADMISSION','RELEASE','DEATH'])].copy()
    terminal=clippers.loc[clippers.ConceptName.isin(['RELEASE','DEATH']),'EndDateTime'].max()
    stop=min(origin+pd.Timedelta(hours=48),terminal) if pd.notna(terminal) else origin+pd.Timedelta(hours=48)
    end=(stop-origin).total_seconds()
    if end<=0: raise ValueError('Nonpositive observation window')
    raw=raw[(raw.StartDateTime>=origin)&(raw.StartDateTime<=stop)].copy()
    sums={ (condition,repeat,family):np.zeros(5) for condition in CONDITIONS
           for repeat in range(2) for family in ['state','trend']}
    details=[]; observations=0; baseline_counts={'state':0,'trend':0}
    for k,(concept,spec) in enumerate(sorted(specs.items())):
        part=raw[raw.ConceptName.isin(spec['input_attributes'])].copy()
        number=pd.to_numeric(part.Value,errors='coerce')
        valid=np.zeros(len(part),bool)
        for attr,ranges in spec['input_attributes'].items():
            for lo,hi in ranges:
                valid|=(part.ConceptName==attr)&number.between(lo,hi,inclusive='both')
        part=part[valid].copy(); number=number[valid].to_numpy(float)
        if part.empty: continue
        part['AbstractionType']='raw-concept'
        base=abstract(part,spec,clippers,origin,end)
        observations+=len(part)
        for family,rows in base.items(): baseline_counts[family]+=len(rows)
        if ordinal<3:
            control=abstract(part,spec,clippers,origin,end)
            if base!=control: raise AssertionError('Unchanged-input control differs')
        cuts=np.asarray(spec['thresholds']); gap=np.diff(cuts)
        spacing=np.minimum(np.r_[gap[0],gap],np.r_[gap,gap[-1]])
        local_scale=spacing[np.abs(number[:,None]-cuts).argmin(axis=1)]
        for ci,(condition,(mode,amount)) in enumerate(CONDITIONS.items()):
            for repeat in range(2):
                rng=np.random.default_rng(np.random.SeedSequence([SEED,ordinal,k,ci,repeat]))
                altered=part.copy()
                if mode=='noise':
                    # Bounded, zero-mean additive changes; no clipping of invalid values.
                    altered['Value']=number+rng.uniform(-amount,amount,len(number))*local_scale
                else:
                    altered=altered[rng.random(len(altered))>=amount].copy()
                outputs=abstract(altered,spec,clippers,origin,end)
                for family,baseline in base.items():
                    result=compare(baseline,outputs[family])
                    sums[condition,repeat,family]+=result
                    details.append(dict(sample_index=ordinal,stratum=stratum,weight=weight,
                                        concept=concept,family=family,condition=condition,repeat=repeat,
                                        baseline_seconds=result[0],altered_seconds=result[1],
                                        overlap_seconds=result[2],same_label_seconds=result[3],union_seconds=result[4]))
    rows=[]
    for (condition,repeat,family),r in sums.items():
        rows.append(dict(sample_index=ordinal,stratum=stratum,weight=weight,condition=condition,
                         repeat=repeat,family=family,baseline_seconds=r[0],altered_seconds=r[1],
                         overlap_seconds=r[2],same_label_seconds=r[3],union_seconds=r[4]))
    return rows,details,dict(sample_index=ordinal,stratum=stratum,weight=weight,
                             observations=observations,baseline_state_intervals=baseline_counts['state'],
                             baseline_trend_intervals=baseline_counts['trend'],observed_hours=end/3600)


def sample(c,n):
    cohort=pd.read_csv(PERSONAL/'INTERVenE-Enc/data/source/context_data.csv')
    events=pd.read_sql_query("select PatientId,ConceptName,StartDateTime from InputPatientData "
                             "where ConceptName in ('ADMISSION','RELEASE','DEATH')",c)
    events.StartDateTime=pd.to_datetime(events.StartDateTime)
    admissions=events[events.ConceptName=='ADMISSION'].set_index('PatientId').StartDateTime
    terminals=events[events.ConceptName.isin(['RELEASE','DEATH'])].groupby('PatientId').StartDateTime.max()
    scope=json.loads(SCOPE.read_text())['definitions']
    aliases=sorted({a for spec in scope.values() for a in spec['input_attributes']})
    placeholders=','.join('?' for _ in aliases)
    query=f"""select r.PatientId,count(*) as numeric_count
               from InputPatientData a join InputPatientData r on a.PatientId=r.PatientId
               where a.ConceptName='ADMISSION' and r.ConceptName in ({placeholders})
               and r.StartDateTime>=a.StartDateTime
               and r.StartDateTime<=datetime(a.StartDateTime,'+48 hours')
               group by r.PatientId"""
    print('Counting first-48-hour numeric observations for sampling...',flush=True)
    counts=pd.read_sql_query(query,c,params=aliases).set_index('PatientId').numeric_count
    features=cohort[['PatientId','age_at_admission','gender']].copy().set_index('PatientId')
    features['numeric_count']=counts.reindex(features.index).fillna(0)
    features['terminal_span_hours']=(terminals.reindex(features.index)-admissions.reindex(features.index)).dt.total_seconds()/3600
    if features.terminal_span_hours.isna().any(): raise ValueError('Missing terminal events')
    features['density_bin']=pd.qcut(features.numeric_count.rank(method='first'),4,labels=False)
    features['span_bin']=pd.qcut(features.terminal_span_hours.rank(method='first'),4,labels=False)
    features['stratum']=features.density_bin*4+features.span_bin
    selected=[]; sampling=[]
    rng=np.random.default_rng(SEED)
    for s,group in features.groupby('stratum',sort=True):
        target=n//16+int(int(s)<n%16)
        target=min(target,len(group))
        chosen=np.sort(rng.choice(group.index.to_numpy(),target,replace=False))
        for pid in chosen: selected.append((int(pid),int(s),len(group)/target))
        sampling.append(dict(stratum=int(s),population_admissions=len(group),sample_admissions=target))
    if len(selected)!=n: raise ValueError('Requested sample unavailable across strata')
    summary={label:{column:dict(median=float(frame[column].median()),
                               q25=float(frame[column].quantile(.25)),q75=float(frame[column].quantile(.75)))
                    for column in ['numeric_count','terminal_span_hours','age_at_admission','gender']}
             for label,frame in [('cohort',features),('sample',features.loc[[x[0] for x in selected]])]}
    return [(i,*entry) for i,entry in enumerate(selected)],sampling,summary


def main():
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--patients',type=int,default=500)
    parser.add_argument('--pilot',type=int,default=0)
    args=parser.parse_args()
    started=time.monotonic()
    with connect() as c: selected,sampling,sample_summary=sample(c,args.patients)
    if args.pilot: selected=selected[:args.pilot]
    print(f'Running {len(selected)} admissions; 4 conditions x 2 realizations; actual Mediator TAKs',flush=True)
    aggregates=[]; details=[]; patients=[]
    with concurrent.futures.ProcessPoolExecutor(max_workers=4,initializer=initialize) as executor:
        futures=[executor.submit(patient_job,item) for item in selected]
        for i,future in enumerate(concurrent.futures.as_completed(futures),1):
            a,d,p=future.result(); aggregates.extend(a); details.extend(d); patients.append(p)
            if i%25==0 or i==len(selected):
                print(f'Completed {i}/{len(selected)} admissions; elapsed {time.monotonic()-started:.1f}s',flush=True)
    prefix='pilot_' if args.pilot else ''
    pd.DataFrame(aggregates).sort_values(['sample_index','condition','repeat','family']).to_csv(ROOT/(prefix+'patient_metrics.csv'),index=False)
    pd.DataFrame(details).to_csv(ROOT/(prefix+'concept_metrics.csv'),index=False)
    pd.DataFrame(patients).sort_values('sample_index').to_csv(ROOT/(prefix+'sample_summary.csv'),index=False)
    metadata=dict(sample_admissions=len(selected),sampling_seed=SEED,replicates=2,conditions=CONDITIONS,
                  sampling='Joint quartiles of first-48h numeric record counts and recorded admission-to-terminal-event span',
                  strata=sampling,cohort_and_sample=sample_summary,observation_hours=48,
                  implementations='Saved TAKRepository; RawConcept.apply, State.apply, Trend.apply, Mediator._apply_global_clippers',
                  scope=json.loads(SCOPE.read_text()),workers=4,elapsed_seconds=time.monotonic()-started,
                  metrics='Duration of baseline, altered, overlap, matching-label overlap, and union; evaluated only in 0–48h prefix',
                  database_access='Read-only SQLite, no output table writes',
                  exclusion='Derived numeric concepts, other abstraction families, and downstream predictions')
    (ROOT/(prefix+'metadata.json')).write_text(json.dumps(metadata,indent=2))
    print(f'Completed in {metadata["elapsed_seconds"]:.1f}s',flush=True)


if __name__=='__main__': main()
