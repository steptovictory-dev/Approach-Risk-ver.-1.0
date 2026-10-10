"""Исходные сообщения для четырех наблюдаемых числовых экстремумов; без очистки."""
import json
import pandas as pd
from audit_pilot_input import BASE

progress=json.loads((BASE/'reports/12_full_progress.json').read_text(encoding='utf8'))
assert progress['complete']
ranges=pd.read_csv(BASE/'reports/12_full_field_completeness.csv')
targets=[('qnh_hpa','minimum'),('qnh_hpa','maximum'),('dewpoint_c','minimum'),('cloud_ceiling_ft','maximum')]
bounds={f'{field}_{direction}':float(getattr(ranges.loc[ranges.field.eq(field),direction], 'min' if direction=='minimum' else 'max')()) for field,direction in targets}
examples=[]
affected=[]
for job in progress['jobs']:
    directory=BASE/'data/full/results'
    key=f"{job['airport']}_{job['year']}"
    messages=pd.read_parquet(directory/f'{key}_reports.parquet',columns=[
        'metar_id','station','metar_utc','metar','qnh_hpa','dewpoint_c','cloud_ceiling_ft','decoded_candidate','core_validated'])
    found=[]
    for field,direction in targets:
        reason=f'{field}_{direction}'
        part=messages.loc[messages[field].eq(bounds[reason])].copy()
        part['review_reason']=reason
        part['extreme_value']=bounds[reason]
        found.append(part)
    selected=pd.concat(found,ignore_index=True)
    if selected.empty:continue
    examples.append(selected)
    features=pd.read_parquet(directory/f'{key}_features.parquet',columns=['metar_id','source_row','GoA','status'])
    impact=features.loc[features.metar_id.isin(selected.metar_id)].groupby('metar_id').agg(
        linked_rows=('source_row','size'),linked_go_arounds=('GoA','sum'),
        linked_rows_within_60min=('status',lambda x:int(x.eq('matched').sum()))).reset_index()
    affected.append(impact)
result=pd.concat(examples,ignore_index=True).drop_duplicates(['metar_id','review_reason'])
impact=pd.concat(affected,ignore_index=True).groupby('metar_id')[['linked_rows','linked_go_arounds','linked_rows_within_60min']].sum().reset_index()
result=result.merge(impact,on='metar_id',how='left',validate='many_to_one')
result.to_csv(BASE/'reports/12_full_extreme_value_examples.csv',index=False)
print(result[['station','metar_utc','metar','review_reason','extreme_value','linked_rows','linked_go_arounds']].to_string(index=False),flush=True)
