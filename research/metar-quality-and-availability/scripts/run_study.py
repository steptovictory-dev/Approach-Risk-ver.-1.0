"""Чувствительность METAR к правилам доступности; без обучения и изменения входов."""
from pathlib import Path
import hashlib
import json
import re
import sys
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parents[1]
PROJECT = BASE.parents[1]
WEATHER = PROJECT / 'europe/weather'
R = BASE / 'reports'
R.mkdir(exist_ok=True)
def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

APPLICATION = json.loads((WEATHER / 'reports/23_final_weather_application.json').read_text(encoding='utf8'))
INPUT = PROJECT / 'europe/data/analytical/06_05_cases_excluded.parquet'
assert sha(INPUT) == APPLICATION['input_sha256']
SCENARIOS = [('primary',60,15,True),('delay0',60,0,True),('delay5',60,5,True),
             ('delay30',60,30,True),('age30',30,15,True),('age90',90,15,True),('no_cor',60,15,False)]
NUMERIC = ['wind_speed_kt','wind_gust_kt','wind_direction_deg','visibility_m',
           'cloud_ceiling_ft','lowest_known_cloud_base_ft','temperature_c','dewpoint_c','qnh_hpa']
OTHER = ['present_weather_tokens','cumulonimbus_reported','towering_cumulus_reported','cavok']
THRESHOLD = dict(wind_speed_kt=5,wind_gust_kt=5,wind_direction_deg=30,visibility_m=1000,
                 cloud_ceiling_ft=500,lowest_known_cloud_base_ft=500,temperature_c=2,dewpoint_c=2,qnh_hpa=2)
registry = pd.read_csv(WEATHER / 'reports/17_numeric_quality_preview.csv',keep_default_na=False).set_index('metar_id')
summaries, differences, daily, missing, samples, provenance = [],[],[],[],[],[]
seen = np.zeros(APPLICATION['rows'],dtype=bool)
labels = 0

def select(events,reports,age,delay,include_cor):
    table = reports if include_cor else reports.loc[~reports.correction_reported.eq(True)]
    # Неоднозначность исходного времени проверяется до исключения COR.
    table = table.sort_values('metar_utc').drop_duplicates('metar_utc')
    times = table.metar_utc.astype('datetime64[ns, UTC]').astype('int64').to_numpy()
    query = events.C40_CROSS_TIME.astype('datetime64[ns, UTC]').astype('int64').to_numpy()
    pos = np.searchsorted(times+delay*60_000_000_000,query,side='left')-1
    has = pos >= 0
    picked = table.iloc[np.maximum(pos,0)].reset_index(drop=True).copy()
    old = (query-times[np.maximum(pos,0)])/60_000_000_000
    observation = has & (old<=age) & ~picked.ambiguous.to_numpy()
    verified = (picked.decoded_candidate.eq(True)|picked.core_validated.eq(True)).to_numpy()
    usable = observation & verified
    ids = picked.metar_id.to_numpy().copy()
    ids[~has] = ''
    values = {}
    for field in NUMERIC+OTHER:
        is_masked = picked.numeric_mask_fields.str.split(';').apply(lambda x:field in x).to_numpy()
        values[field] = picked[field].mask(~usable|is_masked)
    reason = np.select([~has,has&picked.ambiguous.to_numpy(),has&(old>age),observation&~verified],
                       ['no_previous_report','ambiguous_time','older_than_limit','unverified_core'],default='usable')
    return dict(ids=ids,has=has,age=old,observation=observation,usable=usable,values=values,reason=reason,picked=picked)

for item in APPLICATION['files']:
    key = item['name'].removesuffix('.parquet')
    path = WEATHER / 'data/full_cloud_v2/reports' / f'{key}_reports.parquet'
    assert sha(path)==item['source_reports_sha256']
    reports = pd.read_parquet(path).sort_values('metar_utc').reset_index(drop=True)
    reports['ambiguous'] = reports.duplicated('metar_utc',keep=False)
    reports['numeric_mask_fields'] = reports.metar_id.map(registry.proposed_mask_fields).fillna('')
    event_path = WEATHER / 'data/full/events' / item['name']
    events = pd.read_parquet(event_path,columns=['source_row','airport','GoA','C40_CROSS_TIME']).sort_values('C40_CROSS_TIME').reset_index(drop=True)
    assert len(events)==item['rows'] and events.source_row.is_unique
    assert not seen[events.source_row].any()
    seen[events.source_row]=True
    labels += int(events.GoA.sum())
    dates = events.C40_CROSS_TIME.dt.floor('D')
    months = events.C40_CROSS_TIME.dt.month.to_numpy()
    seasons = np.select([np.isin(months,[12,1,2]),np.isin(months,[3,4,5]),np.isin(months,[6,7,8])],['DJF','MAM','JJA'],default='SON')
    primary = select(events,reports,60,15,True)
    final_path = WEATHER / 'data/final_c40_weather_v1' / item['name']
    assert sha(final_path)==item['sha256']
    final = pd.read_parquet(final_path,columns=['source_row','metar_id','weather_core_usable']+[f+'_model' for f in NUMERIC+OTHER])
    assert np.array_equal(events.source_row.to_numpy(),final.source_row.to_numpy())
    assert np.array_equal(primary['ids'],final.metar_id.fillna('').to_numpy())
    assert np.array_equal(primary['usable'],final.weather_core_usable.to_numpy())
    for field in NUMERIC+OTHER:
        pd.testing.assert_series_equal(primary['values'][field].astype(final[field+'_model'].dtype),final[field+'_model'],check_names=False)
    rare = reports.observed_text.str.contains(r'///CB|///TCU|/////////',regex=True)
    rare |= ~reports.decoded_candidate.eq(True)
    rare |= reports.metar_id.isin(registry.index)
    regular = reports.sample(n=min(12,len(reports)),random_state=20261011)
    audit = pd.concat([regular,reports.loc[rare]]).drop_duplicates('metar_id').copy()
    audit['sample_regular'] = audit.metar_id.isin(regular.metar_id)
    audit['sample_enriched'] = audit.metar_id.isin(reports.loc[rare,'metar_id'])
    samples.append(audit)
    for name,age,delay,include_cor in SCENARIOS:
        state = primary if name=='primary' else select(events,reports,age,delay,include_cor)
        changed = state['ids']!=primary['ids']
        any_change = np.zeros(len(events),dtype=bool)
        for field in NUMERIC+OTHER:
            a,b = primary['values'][field],state['values'][field]
            missing_changed = (a.isna()!=b.isna()).to_numpy()
            both = (a.notna()&b.notna()).to_numpy()
            if field in NUMERIC:
                av = pd.to_numeric(a).to_numpy(dtype=float,na_value=np.nan)
                bv = pd.to_numeric(b).to_numpy(dtype=float,na_value=np.nan)
                delta = np.abs(av-bv)
                if field=='wind_direction_deg':delta = np.minimum(delta,360-delta)
                value_changed = both & (delta>1e-9)
                large = both & (delta>=THRESHOLD[field])
            else:
                value_changed = both & a.astype('string').fillna('UNKNOWN').ne(b.astype('string').fillna('UNKNOWN')).to_numpy()
                delta = np.zeros(len(events))
                large = np.zeros(len(events),dtype=bool)
            any_change |= missing_changed|value_changed
            for label in [False,True]:
                part = events.GoA.eq(label).to_numpy()
                differences.append(dict(job=key,scenario=name,GoA=label,field=field,rows=int(part.sum()),
                    paired_present=int((both&part).sum()),missing_changed=int((missing_changed&part).sum()),
                    value_changed=int((value_changed&part).sum()),threshold_changed=int((large&part).sum()),
                    absolute_difference_sum=float(np.nansum(delta[both&part])),threshold=THRESHOLD.get(field,np.nan)))
        counter = pd.DataFrame(dict(airport=events.airport,year=events.C40_CROSS_TIME.dt.year,season=seasons,
            GoA=events.GoA,scenario=name,rows=1,observation=state['observation'].astype(int),usable=state['usable'].astype(int),
            message_changed=changed.astype(int),information_changed=any_change.astype(int),
            cor_usable=(state['usable']&state['picked'].correction_reported.eq(True).to_numpy()).astype(int)))
        summaries.append(counter.groupby(['airport','year','season','GoA','scenario']).sum().reset_index())
        day_frame = pd.DataFrame(dict(day=dates,GoA=events.GoA,scenario=name,rows=1,usable=state['usable'].astype(int),primary=primary['usable'].astype(int)))
        daily.append(day_frame.groupby(['day','GoA','scenario']).sum().reset_index())
        reasons = pd.DataFrame(dict(airport=events.airport,GoA=events.GoA,scenario=name,reason=state['reason']))
        missing.append(reasons.groupby(['airport','GoA','scenario','reason']).size().rename('rows').reset_index())
    assert sha(path)==item['source_reports_sha256'] and sha(final_path)==item['sha256']
    provenance.append(dict(job=key,event_sha256=sha(event_path),reports_sha256=sha(path),final_sha256=sha(final_path),rows=len(events)))
    print('verified',key,flush=True)

assert seen.all() and labels==APPLICATION['go_arounds'] and sha(INPUT)==APPLICATION['input_sha256']
summary = pd.concat(summaries,ignore_index=True)
summary.to_csv(R/'02_scenario_strata.csv',index=False)
overall = summary.groupby(['scenario','GoA']).sum(numeric_only=True).drop(columns='year').reset_index()
for field in ['observation','usable','message_changed','information_changed','cor_usable']:
    overall[field+'_percent'] = 100*overall[field]/overall.rows
overall.to_csv(R/'02_scenario_summary.csv',index=False)
pd.DataFrame(differences).to_csv(R/'03_field_differences.csv',index=False)
pd.concat(missing).groupby(['airport','GoA','scenario','reason']).rows.sum().reset_index().to_csv(R/'03_missing_reasons.csv',index=False)
day_counts = pd.concat(daily).groupby(['day','GoA','scenario'])[['rows','usable','primary']].sum().reset_index()
day_counts.to_csv(R/'04_daily_counts.csv',index=False)
audit_sample = pd.concat(samples).drop_duplicates('metar_id').sort_values(['station','metar_utc'])
audit_sample.to_parquet(BASE/'data/audit_messages.parquet',index=False)
audit_sample[['metar_id','station','metar_utc','sample_regular','sample_enriched']].to_csv(R/'05_audit_sample_registry.csv',index=False)
manifest = dict(rows=len(seen),go_arounds=labels,jobs=len(provenance),input_sha256=APPLICATION['input_sha256'],
    protocol_sha256=sha(BASE/'protocol.md'),implementation_sha256=sha(Path(__file__)),sources=provenance,
    primary_reproduced=True,sources_unchanged=True,all_rows_once=True,training_executed=False,
    audit_messages=len(audit_sample),scenarios=[dict(name=n,age=a,delay=d,include_cor=c) for n,a,d,c in SCENARIOS],
    packages=dict(python=sys.version,pandas=pd.__version__,numpy=np.__version__))
(R/'02_study_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
print(overall.to_string(index=False),flush=True)
