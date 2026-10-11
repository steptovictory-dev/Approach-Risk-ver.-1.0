"""Отдельное чтение исходных токенов и парная временная устойчивость."""
from pathlib import Path
import hashlib
import json
import re
import sys
from datetime import datetime,timezone
import numpy as np
import pandas as pd

BASE=Path(__file__).resolve().parents[1]
PROJECT=BASE.parents[1]
R=BASE/'reports'
sample=pd.read_parquet(BASE/'data/audit_messages.parquet')

def lexical_numbers(raw):
    # Отдельный алгоритм: только однозначные основные токены, без рабочего парсера.
    words=raw.rstrip('=').split()
    stop=next((i for i,t in enumerate(words) if t in ['TEMPO','BECMG','NOSIG','RMK','FCST','INTER'] or re.fullmatch(r'PROB\d\d|(?:FM|TL|AT)\d{4}',t)),len(words))
    words=words[:stop]
    output={}
    pairs=[t for t in words if re.fullmatch(r'M?\d\d/M?\d\d',t)]
    def signed(t):return -int(t[1:]) if t.startswith('M') else int(t)
    if len(pairs)==1:
        t,d=pairs[0].split('/')
        output.update(temperature_c=(pairs[0],signed(t)),dewpoint_c=(pairs[0],signed(d)))
        # Группы после температурной пары не использовать как наблюдаемую видимость.
        before=words[:words.index(pairs[0])]
    else:before=words
    pressures=[t for t in words if re.fullmatch(r'Q\d{4}',t)]
    if len(pressures)==1:output['qnh_hpa']=(pressures[0],int(pressures[0][1:]))
    winds=[t for t in before if re.fullmatch(r'(?:\d{3}|VRB)\d{2,3}(?:G\d{2,3})?KT',t)]
    if len(winds)==1:
        wind=winds[0]
        speed=wind[3:-2].split('G')[0]
        output['wind_speed_kt']=(wind,int(speed))
        if wind[:3]!='VRB' and not (wind[:3]=='000' and int(speed)==0):
            output['wind_direction_deg']=(wind,int(wind[:3]))
        following=before[before.index(wind)+1:]
        visibility=[t for t in following if re.fullmatch(r'\d{4}(?:NDV)?',t)]
        if len(visibility)==1:
            token=visibility[0]
            number=int(token[:4])
            output['visibility_m']=(token,10000 if number==9999 else 50 if number==0 else number)
            output['visibility_relation']=(token,'ge' if number==9999 else 'lt' if number==0 else 'eq')
    return output

records=[]
for row in sample.itertuples():
    for field,(token,expected) in lexical_numbers(row.metar).items():
        actual=getattr(row,field)
        agrees=not pd.isna(actual) and (actual==expected if isinstance(expected,str) else abs(float(actual)-expected)<1e-8)
        records.append(dict(metar_id=row.metar_id,station=row.station,field=field,token=token,expected=expected,
            actual=actual,agrees=agrees,sample_regular=row.sample_regular,sample_enriched=row.sample_enriched))
audit=pd.DataFrame(records)
audit.to_csv(R/'05_lexical_validation.csv',index=False)
audit.groupby(['field','agrees']).size().rename('comparisons').reset_index().to_csv(R/'05_lexical_summary.csv',index=False)

# Ожидания по руководствам Met Office и Météo-France задаются явно, до вызова декодера.
fixtures=[
    ('partial_cb','24010KT 9999 ///CB 10/08 Q1002',dict(cumulonimbus_reported=True,cloud_amount_unavailable=True,cloud_ceiling_ft=None)),
    ('partial_tcu','24010KT 9999 ///TCU 10/08 Q1002',dict(towering_cumulus_reported=True,cloud_amount_unavailable=True,cloud_ceiling_ft=None)),
    ('all_cloud_unknown','24010KT 9999 ///////// 10/08 Q1002',dict(cloud_amount_unavailable=True,cloud_type_unavailable=True,cloud_ceiling_ft=None,cavok=False)),
    ('known_low_cloud','24010KT 9999 BKN008 10/08 Q1002',dict(cloud_ceiling_ft=800,lowest_known_cloud_base_ft=800)),
    ('vertical_visibility','24010KT 0100 VV002 10/08 Q1002',dict(vertical_visibility_ft=200,cloud_ceiling_ft=None)),
    ('visibility_lower_bound','24010KT 9999 FEW020 10/08 Q1002',dict(visibility_m=10000,visibility_relation='ge')),
    ('negative_temperature','24010KT 8000 FEW020 M03/M05 Q1002',dict(temperature_c=-3,dewpoint_c=-5,qnh_hpa=1002)),
    ('trend_separation','24010KT 9999 FEW020 10/08 Q1002 TEMPO BKN004CB',dict(cumulonimbus_reported=False,cloud_ceiling_ft=None)),
]
sys.path.insert(0,str(PROJECT/'europe/weather/scripts'))
from decode_metar_cloud_v2 import decode
fixture_results=[]
for name,body,expectations in fixtures:
    raw='LFPG 191155Z '+body
    actual=decode(raw,'LFPG',datetime(2021,1,19,11,55,tzinfo=timezone.utc))
    for field,value in expectations.items():
        fixture_results.append(dict(case=name,metar=raw,field=field,expected=value,actual=actual[field],agrees=actual[field]==value))
pd.DataFrame(fixture_results).to_csv(R/'05_semantic_fixtures.csv',index=False)
assert all(x['agrees'] for x in fixture_results)

daily=pd.read_csv(R/'04_daily_counts.csv',parse_dates=['day'])
days=pd.date_range(daily.day.min(),daily.day.max(),freq='D')
bootstrap=[]
for (scenario,label),group in daily.groupby(['scenario','GoA']):
    group=group.set_index('day')[['rows','usable','primary']].reindex(days,fill_value=0)
    counts=group[['rows','usable','primary']].to_numpy(dtype=float)
    estimate=100*(counts[:,1].sum()-counts[:,2].sum())/counts[:,0].sum()
    for block in [7,14,28]:
        rng=np.random.default_rng(20261011)
        values=[]
        for repeat in range(2000):
            starts=rng.integers(0,len(days),size=int(np.ceil(len(days)/block)))
            indices=((starts[:,None]+np.arange(block))%len(days)).ravel()[:len(days)]
            totals=counts[indices].sum(axis=0)
            values.append(100*(totals[1]-totals[2])/totals[0])
        lower,upper=np.quantile(values,[.025,.975])
        bootstrap.append(dict(scenario=scenario,GoA=label,block_days=block,replicates=2000,
            difference_pp=estimate,lower_pp=lower,upper_pp=upper))
pd.DataFrame(bootstrap).to_csv(R/'04_block_bootstrap.csv',index=False)
report=dict(sample_messages=len(sample),lexical_comparisons=len(audit),lexical_disagreements=int((~audit.agrees).sum()),
    semantic_fixture_cases=len(fixtures),semantic_fixture_comparisons=len(fixture_results),semantic_fixtures_pass=True,
    physical_weather_independently_verified=False,independent_human_annotation=False,training_executed=False,
    seed=20261011,bootstrap_replicates=2000)
(R/'05_validation_manifest.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
print(report,flush=True)
