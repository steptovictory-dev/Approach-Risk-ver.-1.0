"""Сравнить cloud-v2 на полном архиве, сохранив предыдущие данные и маски CP-10."""
import json
import numpy as np
import pandas as pd
from audit_pilot_input import BASE, INPUT, digest
from decode_metar import decode as old_decode
from decode_metar_cloud_v2 import decode, split_sections, TEMPERATURE, PRESSURE
from core_audit_cloud_v2 import audit_core

R=BASE/'reports'
DEST=BASE/'data/full_cloud_v2'
for name in ['reports','features','numeric_features']:(DEST/name).mkdir(parents=True,exist_ok=True)
progress=json.loads((R/'12_full_progress.json').read_text(encoding='utf8'))
numeric=json.loads((R/'18_numeric_mask_application.json').read_text(encoding='utf8'))
numeric_hashes={item['name']:item['output_sha256'] for item in numeric['files']}
integrity=json.loads((R/'12_full_integrity.json').read_text(encoding='utf8'))
feature_hashes={item['name']:item['sha256'] for item in integrity['feature_files']}
assert digest(INPUT)==progress['jobs'][0]['input_sha256']

def targeted(raw):
    tokens=split_sections(raw)[0].split()
    end=next((i for i,t in enumerate(tokens) if TEMPERATURE.fullmatch(t) or PRESSURE.fullmatch(t)),len(tokens))
    return any(t in ['///CB','///TCU','/////////'] for t in tokens[:end])

def equal(left,right):
    if pd.isna(left) and pd.isna(right):return True
    return left==right

allowed={'lowest_known_cloud_base_ft','lowest_known_cloud_base_relation','cumulonimbus_reported',
    'towering_cumulus_reported','parser_status','parser_error','decode_issues','decoded_candidate',
    'core_validated','core_prefix','core_issues'}

# Совместимость с первоначальными пилотами проверяется по всем сохраненным полям.
pilot_rows=0
for name in ['04_decoded_reports.parquet','09_decoded_reports.parquet']:
    previous=pd.read_parquet(BASE/'data'/name)
    records=[]
    for row in previous.itertuples():
        record=decode(row.metar,row.station,row.metar_utc.to_pydatetime())
        record['archive_month']=row.archive_month
        records.append(record)
    repeated=pd.DataFrame(records)
    repeated['metar_utc']=pd.to_datetime(repeated.metar_utc,utc=True).astype('datetime64[ns, UTC]')
    pd.testing.assert_frame_equal(repeated,previous,check_dtype=False)
    pilot_rows+=len(previous)
print('pilots unchanged',pilot_rows,flush=True)

changes, cases, jobs, residuals=[],[],[],[]
seen=np.zeros(integrity['rows'],dtype=bool)
labels=0
for job in progress['jobs']:
    key=f"{job['airport']}_{job['year']}"
    report_path=BASE/f'data/full/results/{key}_reports.parquet'
    old_report_hash=next(item['sha256'] for item in job['files'] if item['name']==report_path.name)
    assert digest(report_path)==old_report_hash
    previous=pd.read_parquet(report_path)
    revised=previous.copy()
    candidate=previous.metar.map(targeted)
    for index in previous.index[candidate]:
        row=previous.loc[index]
        values=decode(row.metar,row.station,row.metar_utc.to_pydatetime())
        valid,prefix,issues=audit_core(row.metar,row.station,row.metar_utc,values)
        values.update(core_validated=valid,core_prefix=prefix,core_issues=issues)
        changed=[]
        for field,value in values.items():
            if equal(row[field],value):continue
            assert field.startswith('cloud_') or field in allowed, (row.metar_id,field)
            revised.at[index,field]=value
            changed.append(field)
            changes.append(dict(metar_id=row.metar_id,station=row.station,metar_utc=row.metar_utc,
                field=field,old_value=row[field],new_value=value))
        cases.append(dict(metar_id=row.metar_id,station=row.station,metar_utc=row.metar_utc,metar=row.metar,
            changed_fields=';'.join(changed),old_decoded_candidate=bool(row.decoded_candidate),
            new_decoded_candidate=bool(values['decoded_candidate']),old_core_validated=bool(row.core_validated),
            new_core_validated=bool(valid),remaining_issues=values['decode_issues']))
    # Все сообщения вне согласованной области должны оставаться идентичными.
    pd.testing.assert_frame_equal(revised.loc[~candidate],previous.loc[~candidate])
    for field in previous:
        if not (field.startswith('cloud_') or field in allowed):
            pd.testing.assert_series_equal(revised[field],previous[field])
    output_reports=DEST/'reports'/report_path.name
    revised.to_parquet(output_reports,index=False)
    pd.testing.assert_frame_equal(pd.read_parquet(output_reports),revised)

    source_features=BASE/f'data/full/results/{key}_features.parquet'
    assert digest(source_features)==feature_hashes[source_features.name]
    before=pd.read_parquet(source_features)
    after=before.copy()
    update_fields=[f for f in previous if f.startswith('cloud_') or f in allowed]
    indexed=revised.set_index('metar_id')
    linked=before.metar_id.isin(previous.loc[candidate,'metar_id'])
    for field in update_fields:
        if field in after:
            after.loc[linked,field]=after.loc[linked,'metar_id'].map(indexed[field]).to_numpy()
    output_features=DEST/'features'/source_features.name
    after.to_parquet(output_features,index=False)
    pd.testing.assert_frame_equal(pd.read_parquet(output_features),after)
    for field in before:
        if field not in update_fields:pd.testing.assert_series_equal(after[field],before[field])
    pd.testing.assert_frame_equal(after.loc[~linked],before.loc[~linked])
    ids=after.source_row.to_numpy()
    assert after.source_row.is_unique and not seen[ids].any()
    seen[ids]=True
    labels+=int(after.GoA.sum())

    source_numeric=BASE/'data/numeric_quality_v1'/source_features.name
    assert digest(source_numeric)==numeric_hashes[source_features.name]
    old_numeric=pd.read_parquet(source_numeric)
    new_numeric=old_numeric.copy()
    assert old_numeric.source_row.equals(after.source_row)
    for field in ['cloud_ceiling_ft','lowest_known_cloud_base_ft','decoded_candidate','core_validated']:
        new_numeric[field]=after[field]
    for field in ['cloud_ceiling_ft','lowest_known_cloud_base_ft']:
        mask=new_numeric.numeric_mask_fields.str.split(';').apply(lambda fields:field in fields)
        new_numeric[field+'_model']=new_numeric[field].mask(mask)
    for field in old_numeric:
        if field not in ['cloud_ceiling_ft','lowest_known_cloud_base_ft','decoded_candidate','core_validated',
                         'cloud_ceiling_ft_model','lowest_known_cloud_base_ft_model']:
            pd.testing.assert_series_equal(old_numeric[field],new_numeric[field])
    for field in [f[:-6] for f in new_numeric if f.endswith('_model')]:
        mask=new_numeric.numeric_mask_fields.str.split(';').apply(lambda fields:field in fields)
        assert new_numeric.loc[mask,field+'_model'].isna().all()
        pd.testing.assert_series_equal(new_numeric.loc[~mask,field+'_model'],new_numeric.loc[~mask,field],check_names=False)
    output_numeric=DEST/'numeric_features'/source_features.name
    new_numeric.to_parquet(output_numeric,index=False)
    pd.testing.assert_frame_equal(pd.read_parquet(output_numeric),new_numeric)
    assert digest(source_numeric)==numeric_hashes[source_features.name]
    assert digest(source_features)==feature_hashes[source_features.name] and digest(report_path)==old_report_hash

    impact=before.loc[before.status.eq('matched')].groupby('metar_id').agg(
        matched_60min_rows=('source_row','size'),matched_60min_go_arounds=('GoA','sum')).reset_index()
    residuals.append(revised.loc[~revised.decoded_candidate,['metar_id','station','metar_utc','metar','decode_issues','core_validated']].merge(impact,on='metar_id',how='left'))
    match=before.status.eq('matched')
    jobs.append(dict(airport=job['airport'],year=job['year'],rows=len(after),target_messages=int(candidate.sum()),
        linked_target_rows=int((match&linked).sum()),linked_target_go_arounds=int(after.loc[match&linked,'GoA'].sum()),
        old_decoded_rows=int((match&before.decoded_candidate.eq(True)).sum()),
        new_decoded_rows=int((match&after.decoded_candidate.eq(True)).sum()),
        old_core_rows=int((match&before.core_validated.eq(True)).sum()),
        new_core_rows=int((match&after.core_validated.eq(True)).sum()),
        files=[dict(path=str(path.relative_to(DEST)),sha256=digest(path)) for path in [output_reports,output_features,output_numeric]]))
    print('cloud-v2 verified',key,'targets',int(candidate.sum()),flush=True)

assert seen.all() and labels==integrity['go_arounds'] and digest(INPUT)==integrity['input_sha256']
case_table=pd.DataFrame(cases).drop_duplicates('metar_id')
case_table.to_csv(R/'21_cloud_v2_cases.csv',index=False)
change_table=pd.DataFrame(changes).drop_duplicates(['metar_id','field'])
change_table.to_csv(R/'21_cloud_v2_field_changes.csv',index=False)
remaining=pd.concat(residuals,ignore_index=True)
remaining[['matched_60min_rows','matched_60min_go_arounds']]=remaining[['matched_60min_rows','matched_60min_go_arounds']].fillna(0).astype(int)
remaining=remaining.groupby(['metar_id','station','metar_utc','metar','decode_issues','core_validated'],dropna=False)[['matched_60min_rows','matched_60min_go_arounds']].sum().reset_index()
remaining.to_csv(R/'21_cloud_v2_residuals.csv',index=False)
pd.DataFrame([{k:v for k,v in j.items() if k!='files'} for j in jobs]).to_csv(R/'21_cloud_v2_coverage.csv',index=False)
summary=dict(version='cloud-v2',rows=integrity['rows'],go_arounds=labels,input_sha256=integrity['input_sha256'],
    unique_target_messages=len(case_table),unique_changed_messages=int(case_table.changed_fields.ne('').sum()),
    resolved_full_messages=int((~case_table.old_decoded_candidate&case_table.new_decoded_candidate).sum()),
    remaining_residual_messages=len(remaining),pilot_rows_unchanged=pilot_rows,
    all_source_rows_present_once=True,previous_files_unchanged=True,numeric_masks_preserved=True,
    station_time_and_metar_links_unchanged=True,non_cloud_numeric_values_unchanged=True,
    rows_deleted=0,training_executed=False,implementation_sha256=digest(BASE/'scripts/rebuild_cloud_v2.py'),
    decoder_sha256=digest(BASE/'scripts/decode_metar_cloud_v2.py'),original_decoder_sha256=digest(BASE/'scripts/decode_metar.py'),
    jobs=jobs)
(R/'21_cloud_v2_application.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
print({k:v for k,v in summary.items() if k!='jobs'},flush=True)
