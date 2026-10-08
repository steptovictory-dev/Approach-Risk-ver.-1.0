from pathlib import Path
from collections import Counter
import json,sys,hashlib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
if hasattr(sys.stdout, 'reconfigure'):
 sys.stdout.reconfigure(encoding='utf8')
REGION=Path(__file__).resolve().parents[1]
SOURCE=REGION/'data/source/osn23_landings_merged.parquet.gzip'
OUT=REGION/'reports/visualization'
FIELDS=['registration','typecode','icaoaircrafttype','market_segment','AC_CLASS','C40_CROSS_TIME','C40_BEARING','C40_CROSS_LAT','C40_CROSS_LON']
def digest():
 h=hashlib.sha256()
 with SOURCE.open('rb') as f:
  for block in iter(lambda:f.read(4*1024**2),b''):h.update(block)
 return h.hexdigest()
def main():
 OUT.mkdir(parents=True,exist_ok=True)
 before=digest(); expected=json.loads((REGION/'reports/profile_summary.json').read_text(encoding='utf8'))
 assert before==expected['source_sha256']
 airport_parts=[]; month_parts=[]; sensitivity=[]; groups=[]; total=0; joint_same=True
 for batch in pq.ParquetFile(SOURCE).iter_batches(batch_size=100000,columns=['airport','landing_time','GoA','n_attempts',*FIELDS]):
  df=batch.to_pandas(ignore_metadata=True);total+=len(df)
  counts=pd.DataFrame({'airport':df['airport'],'month':df['landing_time'].dt.strftime('%Y-%m')})
  counts['rows']=1
  masks=[]
  for field in FIELDS:
   mask=df[field].isna()
   if pd.api.types.is_string_dtype(df[field]):mask=mask|df[field].eq('').fillna(False)
   counts[field]=mask.astype(int);masks.append(mask)
  joint_same=joint_same and all(df['C40_CROSS_TIME'].isna().equals(df[f].isna()) for f in ['AC_CLASS','C40_BEARING','C40_CROSS_LAT','C40_CROSS_LON'])
  airport_parts.append(counts.groupby(['airport','month']).sum(numeric_only=True))
  month_parts.append(counts.drop(columns='airport').groupby('month').sum(numeric_only=True))
  df['multiple']=df['n_attempts']>2;df['c40_available']=df['C40_CROSS_TIME'].notna()
  sensitivity.append(df.groupby(['airport','multiple'])['GoA'].agg(['size','sum']))
  groups.append(df.groupby('c40_available')['GoA'].agg(['size','sum']))
 def aggregate(parts):return pd.concat(parts).groupby(level=list(range(parts[0].index.nlevels))).sum().reset_index()
 am=aggregate(airport_parts);mm=aggregate(month_parts)
 for table,name in [(am,'missingness_airport_month.csv'),(mm,'missingness_month.csv')]:
  for field in FIELDS:table[field+'_percent']=100*table[field]/table['rows']
  table.to_csv(OUT/name,index=False)
 available=aggregate(groups).rename(columns={'size':'rows','sum':'go_arounds'})
 available['ga_percent']=100*available['go_arounds']/available['rows'];available.to_csv(OUT/'c40_availability.csv',index=False)
 multi=aggregate(sensitivity).rename(columns={'size':'rows','sum':'go_arounds'})
 multi['ga_percent']=100*multi['go_arounds']/multi['rows'];multi.to_csv(OUT/'attempt_sensitivity.csv',index=False)
 daily=pd.read_csv(REGION/'reports/airport_calendar.csv');daily['without_records']=daily['rows'].eq(0)
 shared=daily.groupby('date').agg(airports_without_records=('without_records','sum'),rows=('rows','sum')).reset_index()
 shared.to_csv(OUT/'shared_gaps.csv',index=False)
 all_missing=shared.loc[shared['airports_without_records'].eq(20),'date'].tolist()
 feb=pd.read_csv(REGION/'reports/airport_month_summary.csv')
 feb=feb.loc[feb['month'].eq('2020-02')].sort_values('go_arounds',ascending=False)
 feb.to_csv(OUT/'february_2020.csv',index=False)
 assert total==expected['rows'] and int(available['rows'].sum())==total
 assert digest()==before
 result={'rows':total,'source_sha256':before,'source_unchanged':True,'rows_deleted':0,
  'joint_null_mask_equal_AC_CLASS_C40':joint_same,'days_without_records_at_all_20_airports':all_missing,
  'c40_availability':available.to_dict(orient='records'),'missingness_definition':'physical null or exact empty string; whitespace not merged',
  'sensitivity_status':'descriptive partition only; no source or analytical rows removed'}
 (OUT/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
 print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)
if __name__=='__main__':main()
