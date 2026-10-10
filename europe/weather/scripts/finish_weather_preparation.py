"""Регистр остаточных решений и итоговая таблица согласованного сценария C40."""
import json
from datetime import datetime,timezone
import numpy as np
import pandas as pd
from audit_pilot_input import BASE, INPUT, digest

R=BASE/'reports'
DEST=BASE/'data/final_c40_weather_v1'
DEST.mkdir(parents=True,exist_ok=True)
NUMERIC=['wind_speed_kt','wind_gust_kt','wind_direction_deg','wind_variable_from_deg','wind_variable_to_deg',
    'visibility_m','cloud_ceiling_ft','lowest_known_cloud_base_ft','vertical_visibility_ft',
    'temperature_c','dewpoint_c','qnh_hpa']
OTHER=['wind_variable','wind_calm','gust_group_reported','visibility_relation','visibility_ndv','cavok',
    'cloud_ceiling_relation','cloud_ceiling_status','cloud_amount_unavailable','cloud_type_unavailable',
    'cumulonimbus_reported','towering_cumulus_reported','present_weather_tokens','present_weather_status']
protocol=json.loads((R/'22_final_weather_protocol.json').read_text(encoding='utf8'))
assert protocol['status']=='accepted' and protocol['assumed_latency_minutes']==15
application=json.loads((R/'21_cloud_v2_application.json').read_text(encoding='utf8'))
registry=pd.read_csv(R/'17_numeric_quality_preview.csv').set_index('metar_id')
remaining=pd.read_csv(R/'21_cloud_v2_residuals.csv',keep_default_na=False)
assert remaining.metar_id.is_unique and len(remaining)==1102

def classify(row):
    if row.core_validated:return 'validated_core_unparsed_tail'
    if 'parser_value_disagreement_weather' in row.decode_issues:return 'weather_meaning_unresolved'
    if 'cavok_with_explicit_weather_or_cloud' in row.decode_issues:return 'contradictory_cavok'
    if 'cloud' in row.decode_issues:return 'cloud_format_unresolved'
    if 'qnh' in row.decode_issues:return 'pressure_group_unresolved'
    if 'wind_' in row.decode_issues or 'visibility_' in row.decode_issues:return 'wind_visibility_structure_unresolved'
    return 'other_unverified_core'

remaining['decision_category']=remaining.apply(classify,axis=1)
remaining['model_action']=np.where(remaining.core_validated,'use_validated_core_only','model_weather_unknown')
remaining['source_action']='preserve_without_repair'
remaining.to_csv(R/'22_residual_decision_register.csv',index=False)
remaining.groupby(['decision_category','model_action']).agg(messages=('metar_id','size'),
    old_baseline_rows=('matched_60min_rows','sum')).reset_index().to_csv(R/'22_residual_decision_groups.csv',index=False)
seen=np.zeros(application['rows'],dtype=bool)
labels=0
files,counts,field_counts,quality_journal=[],[],[],[]
for job in application['jobs']:
    key=f"{job['airport']}_{job['year']}"
    path=BASE/f'data/full_cloud_v2/reports/{key}_reports.parquet'
    expected=next(f['sha256'] for f in job['files'] if f['path'].replace('\\','/')==f'reports/{key}_reports.parquet')
    assert digest(path)==expected
    reports=pd.read_parquet(path)
    assert reports.metar_id.is_unique
    reports['ambiguous']=reports.duplicated('metar_utc',keep=False)
    table=reports.sort_values('metar_utc').drop_duplicates('metar_utc').copy()
    table['assumed_available_utc']=table.metar_utc+pd.Timedelta(minutes=15)
    event_path=BASE/f'data/full/events/{key}.parquet'
    events=pd.read_parquet(event_path).sort_values('C40_CROSS_TIME')
    assert len(events)==job['rows'] and events.source_row.is_unique
    ids=events.source_row.to_numpy()
    assert not seen[ids].any()
    seen[ids]=True
    labels+=int(events.GoA.sum())
    right=table[['assumed_available_utc','metar_utc','metar_id','ambiguous']]
    links=pd.merge_asof(events,right,left_on='C40_CROSS_TIME',right_on='assumed_available_utc',
        direction='backward',allow_exact_matches=False)
    known=links.metar_utc.notna()
    assert links.loc[known,'assumed_available_utc'].lt(links.loc[known,'C40_CROSS_TIME']).all()
    links['age_minutes']=(links.C40_CROSS_TIME-links.metar_utc).dt.total_seconds()/60
    links.loc[links.ambiguous.eq(True),'metar_id']=None
    frame=links.merge(reports.drop(columns=['metar_utc','ambiguous']),on='metar_id',how='left',validate='many_to_one')
    assert len(frame)==len(events)
    # Повторное сопоставление после задержки, а не фильтрация связи сценария 0 минут.
    available=frame.metar_utc.notna() & frame.age_minutes.le(60) & ~frame.ambiguous.eq(True)
    verified=frame.decoded_candidate.eq(True)|frame.core_validated.eq(True)
    usable=available&verified
    frame['observation_usable']=available
    frame['weather_core_usable']=usable
    frame['availability_verified']=False
    frame['weather_protocol_version']='final-c40-weather-v1'
    frame['weather_missing_reason']=''
    frame.loc[~known,'weather_missing_reason']='no_previous_report'
    frame.loc[known&frame.age_minutes.gt(60),'weather_missing_reason']='older_than_60min'
    frame.loc[frame.ambiguous.eq(True),'weather_missing_reason']='ambiguous_timestamp'
    frame.loc[available&~verified,'weather_missing_reason']='unverified_observed_core'
    frame['residual_decision']=frame.metar_id.map(remaining.set_index('metar_id').decision_category).fillna('no_residual_issue')
    frame['numeric_quality_reasons']=frame.metar_id.map(registry.numeric_quality_reasons).fillna('not_flagged_by_current_checks')
    frame['numeric_mask_fields']=frame.metar_id.map(registry.proposed_mask_fields).fillna('')
    frame['numeric_quality_protocol_version']='numeric-quality-v1'
    for field in NUMERIC:
        frame[field]=pd.to_numeric(frame[field],errors='raise').astype('Float64')
    for field in OTHER:
        if field in ['wind_variable','wind_calm','gust_group_reported','visibility_ndv','cavok',
                     'cloud_amount_unavailable','cloud_type_unavailable','cumulonimbus_reported','towering_cumulus_reported']:
            frame[field]=frame[field].astype('boolean')
        else:frame[field]=frame[field].astype('string')
    for field in NUMERIC+OTHER:
        numeric_mask=frame.numeric_mask_fields.str.split(';').apply(lambda values:field in values)
        frame[field+'_model']=frame[field].mask(~usable|numeric_mask)
        if field in NUMERIC:
            selected=frame.loc[numeric_mask,['source_row','airport','GoA','metar_id',field,'weather_core_usable']].copy()
            selected=selected.rename(columns={field:'original_value'})
            selected['field']=field
            quality_journal.append(selected)
        assert frame.loc[~usable|numeric_mask,field+'_model'].isna().all()
        pd.testing.assert_series_equal(frame.loc[usable&~numeric_mask,field+'_model'],frame.loc[usable&~numeric_mask,field],check_names=False,check_dtype=False)
    # Вертикальная видимость сохраняется отдельно от высоты облачной нижней границы.
    frame['temporal_split']='not_assigned'
    for field in frame.select_dtypes(include='object'):
        frame[field]=frame[field].where(frame[field].notna(),None)
    output=DEST/f'{key}.parquet'
    frame.to_parquet(output,index=False)
    saved=pd.read_parquet(output)
    pd.testing.assert_frame_equal(saved,frame)
    pd.testing.assert_frame_equal(saved[events.columns].reset_index(drop=True),events.reset_index(drop=True))
    assert digest(path)==expected
    for label,part in frame.groupby('GoA'):
        counts.append(dict(airport=job['airport'],year=job['year'],GoA=bool(label),rows=len(part),
            observation_usable=int(part.observation_usable.sum()),weather_core_usable=int(part.weather_core_usable.sum()),
            cor_usable=int((part.weather_core_usable & part.correction_reported.eq(True)).sum()),
            numeric_flagged_rows=int(part.numeric_mask_fields.ne('').sum())))
        for field in NUMERIC+OTHER:
            field_counts.append(dict(airport=job['airport'],year=job['year'],GoA=bool(label),field=field,
                rows=len(part),present_model=int(part[field+'_model'].notna().sum())))
    files.append(dict(name=output.name,sha256=digest(output),source_reports_sha256=expected,rows=len(saved)))
    print('final weather verified',key,len(saved),flush=True)
assert seen.all() and labels==application['go_arounds'] and digest(INPUT)==application['input_sha256']
pd.DataFrame(counts).to_csv(R/'23_final_weather_coverage.csv',index=False)
pd.DataFrame(field_counts).to_csv(R/'23_final_weather_fields.csv',index=False)
quality=pd.concat(quality_journal,ignore_index=True)
quality.to_csv(R/'23_final_numeric_mask_journal.csv',index=False)
report=dict(version='final-c40-weather-v1',rows=len(seen),go_arounds=labels,
    input_sha256=application['input_sha256'],all_source_rows_present_once=True,
    events_and_labels_unchanged=True,source_reports_unchanged=True,strict_assumed_availability_before_c40=True,
    max_age_minutes=60,assumed_latency_minutes=15,include_cor=True,imputation=False,
    availability_verified=False,numeric_mask_registry_preserved=True,
    numeric_flagged_rows=int(quality.source_row.nunique()),numeric_masked_cells=len(quality),
    rows_deleted=0,training_executed=False,temporal_split_assigned=False,
    created_utc=datetime.now(timezone.utc).isoformat(),implementation_sha256=digest(BASE/'scripts/finish_weather_preparation.py'),files=files)
(R/'23_final_weather_application.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
print({k:v for k,v in report.items() if k!='files'},flush=True)
