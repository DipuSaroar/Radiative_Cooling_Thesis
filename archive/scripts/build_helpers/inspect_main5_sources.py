from pathlib import Path
import pandas as pd
import numpy as np
P=Path('Dipu_Bhai_Thesis/processed')
r=pd.read_csv(P/'merged_raft.csv')
m=pd.read_csv(P/'merged_meteo.csv')
i=pd.read_csv(P/'merged_ir.csv')
l=pd.read_csv(P/'merged_dataset_104957.csv',parse_dates=['timestamp'])
r['timestamp']=pd.to_datetime(r.DATE+' '+r.TIME,format='%d/%m/%Y %H:%M:%S')
m['timestamp']=pd.to_datetime(m.TIME,format='%a %d %B %Y %H:%M:%S')
i['timestamp']=pd.to_datetime(i.TIMESTAMP)
for name,d in [('raft',r),('meteo',m),('ir',i),('legacy',l)]:
 print('\nSOURCE',name,'shape',d.shape,'span',d.timestamp.min(),d.timestamp.max(),'duplicate timestamps',d.timestamp.duplicated().sum())
 print(d.drop(columns='timestamp').head(2).to_string(index=False))
 print('cadence',d.timestamp.sort_values().diff().value_counts().head(5).to_dict())
print('IR types',i.groupby('RECORD_TYPE').agg(rows=('TIMESTAMP','size'),temperature=('TEMPERATURE_C','count'),tproc=('TPROC','count'),first=('timestamp','min'),last=('timestamp','max')).to_string())
print('RAFT extrema',r.HEAT_FLUX_V.describe().to_dict())
print('LEGACY ratio volts/flux', (l.Q_net_voltage_V/l.Q_net).describe().to_dict())
j=r.merge(l[['timestamp','Q_net_raw','Q_net_voltage_V','Q_net','Q_raw_unit']],on='timestamp',how='left',validate='one_to_one')
print('RAFT matched',j.Q_net.notna().sum(),'unmatched',j.Q_net.isna().sum(),'raw equals',np.isclose(pd.to_numeric(j.HEAT_FLUX_V,errors='coerce'),j.Q_net_raw,equal_nan=True).sum())
print('units',j.Q_raw_unit.value_counts(dropna=False).to_dict())
print(j.groupby('Q_raw_unit').agg(first=('timestamp','min'),last=('timestamp','max'),lo=('HEAT_FLUX_V','min'),hi=('HEAT_FLUX_V','max')).to_string())
print('RAFT files',r.groupby('SOURCE_FILE').HEAT_FLUX_V.agg(['size']).to_string())
print('legacy original conversions sample',j[['timestamp','HEAT_FLUX_V','Q_raw_unit','Q_net']].head(8).to_string(index=False))
