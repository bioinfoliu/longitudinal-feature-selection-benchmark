#!/usr/bin/env python3
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import friedmanchisquare, studentized_range

ROOT=Path('/home/zliu/ctsnn/results/main_benchmark/final_benchmark')
metrics=pd.read_csv(ROOT/'repeat_level_metrics_all_tasks.csv')
method_order=['Lasso','ElasticNet','StabilitySelection','Stabl-RP','LongGroupLasso','LLSS','RandomForest','MutualInfo','Boruta','mRMR','SNN-FS','OutcomeSNN-FS','geeVerse']

# Rank methods within each task using the 50-repeat mean, then average multiple
# outcomes from the same source cohort so every cohort contributes once.
task_rows=[]
for (task,cohort,typ),g in metrics.groupby(['Task','Cohort','TaskType']):
    metric='AUROC' if typ=='classification' else 'RMSE'
    means=g.groupby('Method')[metric].mean().dropna()
    ranks=means.rank(ascending=(typ=='regression'),method='average')
    for method in ranks.index:
        task_rows.append({'Task':task,'Cohort':cohort,'TaskType':typ,'Method':method,'PerformanceMean':means[method],'TaskRank':ranks[method]})
task=pd.DataFrame(task_rows)
task.to_csv(ROOT/'task_mean_rank_matrix_long.csv',index=False)
cohort=(task.groupby(['Cohort','TaskType','Method'],as_index=False)
        .agg(CohortRank=('TaskRank','mean'),TasksRepresented=('Task','nunique')))
cohort['CohortWin']=(cohort['CohortRank']==cohort.groupby(['Cohort','TaskType'])['CohortRank'].transform('min')).astype(float)
cohort.to_csv(ROOT/'cohort_level_rank_matrix.csv',index=False)
summary=(cohort.groupby(['TaskType','Method'],as_index=False)
         .agg(MeanRank=('CohortRank','mean'),SDRank=('CohortRank','std'),WinRate=('CohortWin','mean'),Cohorts=('Cohort','nunique'))
         .sort_values(['TaskType','MeanRank']))
summary.to_csv(ROOT/'cohort_level_rank_summary.csv',index=False)

fried=[]
plot_data=[]
for typ,cg in cohort.groupby('TaskType'):
    wide=cg.pivot(index='Cohort',columns='Method',values='CohortRank')
    methods=[m for m in method_order if m in wide.columns and wide[m].notna().all()]
    wide=wide[methods].dropna()
    stat,p=friedmanchisquare(*[wide[m].to_numpy() for m in methods])
    k=len(methods); n=len(wide)
    q=studentized_range.ppf(0.95,k,np.inf)/np.sqrt(2)
    cd=float(q*np.sqrt(k*(k+1)/(6*n)))
    fried.append({'Scope':f'Source cohorts: {typ}','Methods':k,'Cohorts':n,'ChiSquare':stat,'PValue':p,'NemenyiCD05':cd})
    means=wide.mean().sort_values()
    plot_data.append((typ,means,cd,n))
pd.DataFrame(fried).to_csv(ROOT/'cohort_level_friedman_tests.csv',index=False)

fig,axes=plt.subplots(1,2,figsize=(14,6.2),constrained_layout=True)
colors={'SNN-FS':'#1F78B4','OutcomeSNN-FS':'#E31A1C','geeVerse':'#6A3D9A'}
for ax,(typ,means,cd,n) in zip(axes,plot_data):
    y=np.arange(len(means))
    ax.scatter(means.values,y,s=62,c=[colors.get(m,'#777777') for m in means.index],zorder=3)
    ax.set_yticks(y,means.index,fontsize=10)
    ax.set_xlabel('Cohort-equal-weighted mean rank (lower is better)')
    ax.set_title(f'{typ.capitalize()}: {n} source cohorts',weight='bold')
    ax.grid(axis='x',alpha=.25)
    ax.invert_yaxis()
    left=1.0
    top=-0.75
    ax.plot([left,left+cd],[top,top],color='black',lw=3,clip_on=False)
    ax.plot([left,left],[top-.08,top+.08],color='black',lw=2,clip_on=False)
    ax.plot([left+cd,left+cd],[top-.08,top+.08],color='black',lw=2,clip_on=False)
    ax.text(left+cd/2,top-.18,f'CD = {cd:.2f}',ha='center',va='top',fontsize=10)
fig.suptitle('Source-cohort sensitivity analysis of method ranks',fontsize=16,weight='bold')
fig.savefig(ROOT/'figure_cohort_weighted_average_rank_cd.png',dpi=350,bbox_inches='tight')
fig.savefig(ROOT/'figure_cohort_weighted_average_rank_cd.pdf',bbox_inches='tight')
plt.close(fig)
print(pd.DataFrame(fried).to_string(index=False))
