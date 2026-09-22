from pathlib import Path
from itertools import combinations
import numpy as np
import pandas as pd
root=Path('/home/zliu/ctsnn/results/main_benchmark/final_benchmark')
df=pd.read_csv(root/'selected_features_all_tasks.csv')
rows=[]
for (task, method), sub in df.groupby(['Task','Method'], sort=True):
    for budget in (3,5,10):
        panels={}
        for (repeat, fold), g in sub.groupby(['Repeat','Fold']):
            feats=set(g.loc[g['Rank']<=budget,'Feature'].astype(str))
            if feats:
                panels[(int(repeat),int(fold))]=feats
        vals=[]
        folds=sorted({f for _,f in panels})
        for fold in folds:
            reps=sorted(r for r,f in panels if f==fold)
            for r1,r2 in combinations(reps,2):
                a,b=panels[(r1,fold)],panels[(r2,fold)]
                union=a|b
                vals.append(len(a&b)/len(union) if union else np.nan)
        vals=np.asarray(vals,dtype=float)
        vals=vals[np.isfinite(vals)]
        if len(vals):
            meta=sub.iloc[0]
            rows.append({'Task':task,'Dataset':meta['Dataset'],'Cohort':meta['Cohort'],'TaskType':meta['TaskType'],'Method':method,'Budget':budget,'MeanJaccard':vals.mean(),'SDJaccard':vals.std(ddof=1) if len(vals)>1 else 0.0,'MedianJaccard':np.median(vals),'MinJaccard':vals.min(),'MaxJaccard':vals.max(),'PairCount':len(vals),'Repeats':sub['Repeat'].nunique(),'Folds':len(folds)})
out=pd.DataFrame(rows).sort_values(['Task','Budget','MeanJaccard'],ascending=[True,True,False])
out.to_csv(root/'selection_stability.csv',index=False)
print('wrote',len(out),'rows')
