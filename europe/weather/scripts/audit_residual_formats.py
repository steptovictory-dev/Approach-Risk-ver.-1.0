"""Воспроизводимый аудит всех остаточных форматов без изменения декодера."""
from collections import Counter
import json
import re
import pandas as pd
from audit_pilot_input import BASE, digest
from decode_metar import decode, split_sections
from core_audit import audit_core

R=BASE/'reports'
source=R/'12_full_residuals.csv'
frame=pd.read_csv(source,keep_default_na=False)
assert frame.metar_id.is_unique and len(frame)==2425
token_counts=Counter()
checks=[]
for row in frame.itertuples():
    stamp=pd.Timestamp(row.metar_utc)
    result=decode(row.metar,row.station,stamp.to_pydatetime())
    assert result['metar_id']==row.metar_id
    assert result['decode_issues']==row.decode_issues and not result['decoded_candidate']
    core_ok,_,_=audit_core(row.metar,row.station,stamp,result)
    assert core_ok==(str(row.core_validated).lower()=='true')
    observed=split_sections(row.metar)[0]
    tokens=observed.split()
    flags=[]
    if '///TCU' in tokens or '///CB' in tokens:flags.append('convective_type_without_amount_height')
    if '/////////' in tokens:flags.append('nine_slash_cloud_placeholder')
    if '//////' in tokens:flags.append('six_slash_placeholder')
    if 'parser_value_disagreement_weather' in row.decode_issues:flags.append('weather_parser_disagreement')
    if 'unclassified_weather_after_qnh' in row.decode_issues:flags.append('unclassified_tail_after_qnh')
    if 'visibility_format_outside_pilot_scope' in row.decode_issues:flags.append('visibility_outside_current_scope')
    if 'wind_group_missing_or_multiple' in row.decode_issues:flags.append('wind_missing_or_multiple')
    if 'parser_value_disagreement_cloud' in row.decode_issues:flags.append('other_cloud_parser_disagreement')
    if 'parser_error' in row.decode_issues:flags.append('strict_parser_error')
    if not flags:flags.append('other_issue_requires_individual_review')
    # Приоритетная категория описывает наблюдаемый формат, а не причину ошибки станции.
    primary=flags[0]
    unparsed=re.search(r"Unparsed groups in body '(.*?)' while processing",row.parser_error)
    checks.append(dict(metar_id=row.metar_id,station=row.station,metar_utc=row.metar_utc,
        metar=row.metar,observed_text=observed,primary_category=primary,
        format_flags=';'.join(flags),decode_issues=row.decode_issues,parser_error=row.parser_error,
        core_validated=core_ok,unparsed_groups=unparsed.group(1) if unparsed else '',
        towering_cumulus_reported=result['towering_cumulus_reported'],
        cumulonimbus_reported=result['cumulonimbus_reported'],
        linked_rows=int(row.matched_rows),linked_go_arounds=int(row.matched_go_arounds),
        replay_matches_saved=True))
    token_counts.update(set(tokens))
cases=pd.DataFrame(checks)
progress=json.loads((R/'12_full_progress.json').read_text(encoding='utf8'))
integrity=json.loads((R/'12_full_integrity.json').read_text(encoding='utf8'))
source_hashes={item['name']:item['sha256'] for item in integrity['feature_files']}
impacts=[]
for job in progress['jobs']:
    path=BASE/'data/full/results'/f"{job['airport']}_{job['year']}_features.parquet"
    assert digest(path)==source_hashes[path.name]
    links=pd.read_parquet(path,columns=['source_row','metar_id','GoA','status'])
    part=links.loc[links.metar_id.isin(cases.metar_id) & links.status.eq('matched')]
    impacts.append(part.groupby('metar_id').agg(matched_60min_rows=('source_row','size'),
        matched_60min_go_arounds=('GoA','sum')).reset_index())
impact=pd.concat(impacts,ignore_index=True).groupby('metar_id')[['matched_60min_rows','matched_60min_go_arounds']].sum().reset_index()
cases=cases.merge(impact,on='metar_id',how='left',validate='one_to_one')
cases[['matched_60min_rows','matched_60min_go_arounds']]=cases[['matched_60min_rows','matched_60min_go_arounds']].fillna(0).astype(int)
cases.to_csv(R/'19_residual_format_cases.csv',index=False)
groups=cases.groupby('primary_category').agg(messages=('metar_id','size'),
    core_validated_messages=('core_validated','sum'),linked_rows=('linked_rows','sum'),
    linked_go_arounds=('linked_go_arounds','sum'),matched_60min_rows=('matched_60min_rows','sum'),
    matched_60min_go_arounds=('matched_60min_go_arounds','sum')).reset_index().sort_values('messages',ascending=False)
assert groups.messages.sum()==2425
groups.to_csv(R/'19_residual_format_groups.csv',index=False)
issues=cases.assign(issue=cases.decode_issues.str.split(';')).explode('issue').groupby('issue').agg(
    messages=('metar_id','size'),linked_rows=('linked_rows','sum')).reset_index().sort_values('messages',ascending=False)
issues.to_csv(R/'19_residual_issue_counts.csv',index=False)
pd.DataFrame([dict(token=token,messages=count) for token,count in token_counts.most_common()]).to_csv(R/'19_residual_token_inventory.csv',index=False)
errors=cases.loc[cases.parser_error.ne('')].groupby('parser_error').agg(messages=('metar_id','size')).reset_index().sort_values('messages',ascending=False)
errors.to_csv(R/'19_residual_parser_errors.csv',index=False)
cases.loc[cases.unparsed_groups.ne('')].groupby('unparsed_groups').agg(messages=('metar_id','size'),
    matched_60min_rows=('matched_60min_rows','sum')).reset_index().sort_values('messages',ascending=False).to_csv(R/'19_residual_unparsed_groups.csv',index=False)
summary=dict(messages=len(cases),exclusive_category_total=int(groups.messages.sum()),
    all_residuals_replayed=True,all_saved_issue_strings_reproduced=True,all_saved_core_flags_reproduced=True,
    source_sha256=digest(source),decoder_sha256=digest(BASE/'scripts/decode_metar.py'),
    audit_script_sha256=digest(BASE/'scripts/audit_residual_formats.py'),
    decoder_changed=False,weather_features_changed=False,rows_deleted=0,training_executed=False,
    links_scope='all baseline statuses; no 60-minute filter in source residual aggregate',
    format_flags_overlap=True,primary_categories_exclusive=True)
summary.update(matched_60min_rows=int(cases.matched_60min_rows.sum()),
    matched_60min_go_arounds=int(cases.matched_60min_go_arounds.sum()),
    matched_60min_core_validated_rows=int(cases.loc[cases.core_validated,'matched_60min_rows'].sum()))
(R/'19_residual_format_audit.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
print(groups.to_string(index=False),flush=True)
print(errors.head(15).to_string(index=False),flush=True)
