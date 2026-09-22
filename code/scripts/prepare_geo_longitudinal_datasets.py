#!/usr/bin/env python3
from pathlib import Path
import csv, gzip
import numpy as np
import pandas as pd

BASE=Path('/home/zliu/ctsnn')
OUT=BASE/'project/data/processed'
OUT.mkdir(parents=True,exist_ok=True)

def read_geo(acc):
    path=BASE/'data_sources'/acc.lower()/f'{acc.upper()}_series_matrix.txt.gz'
    meta={}
    accessions=None
    with gzip.open(path,'rt',errors='replace') as handle:
        for line in handle:
            if line.startswith('!Sample_geo_accession'):
                accessions=next(csv.reader([line],delimiter='\t'))[1:]
            elif line.startswith('!Sample_characteristics_ch1'):
                vals=next(csv.reader([line],delimiter='\t'))[1:]
                key=vals[0].split(':',1)[0].strip().lower()
                meta[key]=[v.split(':',1)[1].strip() if ':' in v else v for v in vals]
            elif line.startswith('!series_matrix_table_begin'):
                break
    expr=pd.read_csv(path,sep='\t',comment='!',index_col=0,compression='gzip')
    expr.columns=[str(c) for c in expr.columns]
    expr.index=[str(i) for i in expr.index]
    if accessions is None or len(accessions)!=expr.shape[1]:
        raise RuntimeError(f'{acc}: metadata/expression sample mismatch')
    md=pd.DataFrame(meta,index=accessions).loc[expr.columns]
    return md,expr.apply(pd.to_numeric,errors='coerce')

def select_features(expr,n=1000):
    x=expr.copy()
    if np.nanpercentile(x.to_numpy(float),99)>100:
        x=np.log2(x.clip(lower=0)+1)
    var=x.var(axis=1,skipna=True).sort_values(ascending=False)
    keep=var[var>0].head(n).index
    z=x.loc[keep].T
    z.columns=[f'probe_{c}' for c in z.columns]
    return z

def prepare_ms(acc):
    md,expr=read_geo(acc)
    features=select_features(expr)
    visit=md['visit'].str.lower().map({'baseline':0.0,'follow-up year 1':1.0,'follow-up year 2':2.0})
    frame=pd.DataFrame({'ID':md['id'].astype(str),'time':visit,'label':md['disease'].str.contains('multiple sclerosis',case=False).astype(int)},index=md.index)
    frame=pd.concat([frame,features],axis=1).dropna(subset=['ID','time','label'])
    path=OUT/f'{acc.upper()}_MS_LONGITUDINAL.csv'
    frame.to_csv(path,index=False)
    return path,frame

def prepare_flu():
    acc='GSE48023'; md,expr=read_geo(acc); features=select_features(expr)
    titer=pd.read_csv(BASE/'data_sources/gse48023/GSE48023_H1N1_HAI_titers.txt.gz',sep='\t',compression='gzip')
    titer.index=titer.index.astype(str)
    response=(pd.to_numeric(titer['Day14'])-pd.to_numeric(titer['Day0'])).rename('label')
    times=md['time'].str.extract(r'Day(\d+)',expand=False).astype(float)
    frame=pd.DataFrame({'ID':md['subject'].astype(str),'time':times},index=md.index)
    frame['label']=frame['ID'].map(response)
    frame=pd.concat([frame,features],axis=1).dropna(subset=['ID','time','label'])
    path=OUT/'GSE48023_H1N1_RESPONSE_LONGITUDINAL.csv'
    frame.to_csv(path,index=False)
    return path,frame

records=[]
for acc in ('GSE41848','GSE41849'):
    path,frame=prepare_ms(acc)
    records.append({'Dataset':acc,'Path':str(path),'Observations':len(frame),'Participants':frame.ID.nunique(),'Visits':frame.time.nunique(),'Features':frame.shape[1]-3,'Outcome':'MS vs control','PositiveParticipants':frame.groupby('ID').label.first().sum()})
path,frame=prepare_flu()
records.append({'Dataset':'GSE48023','Path':str(path),'Observations':len(frame),'Participants':frame.ID.nunique(),'Visits':frame.time.nunique(),'Features':frame.shape[1]-3,'Outcome':'H1N1 HAI Day14-Day0','PositiveParticipants':np.nan})
audit=pd.DataFrame(records)
audit.to_csv(OUT/'NEW_GEO_DATASET_AUDIT.csv',index=False)
print(audit.to_string(index=False))
