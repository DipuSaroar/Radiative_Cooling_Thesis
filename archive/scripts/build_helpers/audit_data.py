import sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tmp/review_deps'))
import pandas as pd
import numpy as np
P=ROOT/'Dipu_Bhai_Thesis/processed'
cols=['T_amb','RH','v','G','T_sky','Q_net']
raw=pd.read_csv(P/'merged_dataset_104957.csv',parse_dates=['timestamp'])
model=pd.read_csv(P/'model_input_5min.csv',parse_dates=['timestamp']).set_index('timestamp')
valid=raw[['timestamp']+cols].dropna().set_index('timestamp').sort_index()
grid=valid.resample('5min').mean()
empty=grid.Q_net.isna()
groups=(empty!=empty.shift()).cumsum()
gaps=[]
for _,s in empty.groupby(groups):
    if s.iloc[0]: gaps.append((len(s),str(s.index[0]),str(s.index[-1])))
print('Raw rows',len(raw),'complete physical rows',len(valid),'grid slots',len(grid),'interpolated slots',int(empty.sum()),'percent',100*empty.mean())
print('Longest missing runs (slots,start,end)',sorted(gaps,reverse=True)[:8])
reconstructed=grid.interpolate(method='time',limit_area='inside').dropna().rolling('10min',min_periods=2).mean().dropna()
print('Saved table rows',len(model),'missing cells',int(model.isna().sum().sum()),'duplicates',int(model.index.duplicated().sum()),'irregular steps',int((model.index.to_series().diff().dropna()!=pd.Timedelta('5min')).sum()))
print('Reconstruction agrees',reconstructed.index.equals(model.index),np.allclose(reconstructed[cols],model[cols],rtol=1e-10,atol=1e-10))
print('Smoothed rows influenced by an interpolated slot',int(empty.astype(int).rolling(2).max().reindex(model.index).sum()))
print('Raw per-column missing',raw[cols].isna().sum().to_dict())
print('Available values discarded by joint dropna', (raw[cols].notna().sum()-len(valid)).to_dict())
print('Prepared stats',model[cols].agg(['min','max','mean']).round(3).to_string())
print('Raw Tsky floor',raw.T_sky.min(),'count',int((raw.T_sky==-50).sum()),'valid',int(raw.T_sky.notna().sum()))
print('Raw Q_net units',raw.Q_raw_unit.value_counts().to_dict())
print('Saved correlation',model[cols].corr().Q_net.to_dict())
print('Observed-only resampled correlation',grid[cols].corr().Q_net.to_dict())
