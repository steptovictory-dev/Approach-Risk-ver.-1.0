"""Глобальная кратность и заполненность всей базы без чтения ее целиком в RAM."""
import json
import numpy as np
import pandas as pd
from audit_pilot_input import BASE, INPUT, digest
from full_weather_pipeline import write_json

def main():
    report=BASE/'reports'
    protocol=json.loads((report/'11_full_protocol.json').read_text(encoding='utf8'))
    progress=json.loads((report/'12_full_progress.json').read_text(encoding='utf8'))
    assert progress['complete'] and progress['completed']==progress['planned']
    seen=np.zeros(protocol['rows'],dtype=bool)
    fields=['wind_speed_kt','wind_gust_kt','visibility_m','cloud_ceiling_ft','temperature_c','dewpoint_c','qnh_hpa']
    counts=[]
    gaps=[]
    labels=0
    sources=[]
    formats=[]
    for job in progress['jobs']:
        path=BASE/'data/full/results'/f"{job['airport']}_{job['year']}_features.parquet"
        frame=pd.read_parquet(path,columns=['source_row','airport','GoA','C40_CROSS_TIME','metar_utc','age_minutes','status',*fields])
        ids=frame.source_row.to_numpy()
        assert np.all((ids>=0)&(ids<len(seen))) and len(np.unique(ids))==len(ids)
        assert not seen[ids].any()
        seen[ids]=True
        assert len(frame)==job['rows'] and int(frame.GoA.sum())==job['go_arounds']
        labels+=int(frame.GoA.sum())
        known=frame.metar_utc.notna()
        assert frame.loc[known,'metar_utc'].lt(frame.loc[known,'C40_CROSS_TIME']).all()
        for label,part in frame.groupby('GoA'):
            for field in fields:
                valid=part[field].dropna()
                counts.append(dict(airport=job['airport'],year=job['year'],GoA=bool(label),field=field,
                    total=len(part),present=len(valid),minimum=float(valid.min()) if len(valid) else None,
                    maximum=float(valid.max()) if len(valid) else None))
        frame['month']=frame.C40_CROSS_TIME.dt.strftime('%Y-%m')
        for (month,label),part in frame.groupby(['month','GoA']):
            gaps.append(dict(airport=job['airport'],month=month,GoA=bool(label),total=len(part),
                no_previous=int(part.metar_utc.isna().sum()),older_than_60=int(part.age_minutes.gt(60).sum()),
                older_than_90=int(part.age_minutes.gt(90).sum()),ambiguous=int(part.status.eq('ambiguous').sum()),
                maximum_observation_age_minutes=float(part.age_minutes.max()) if part.metar_utc.notna().any() else None))
        sources.append(dict(name=path.name,sha256=digest(path)))
        messages=pd.read_parquet(BASE/'data/full/results'/f"{job['airport']}_{job['year']}_reports.parquet",
            columns=['metar_id','wind_unit','qnh_unit','visibility_unit','auto_reported','correction_reported'])
        for (wind,qnh,visibility),part in messages.groupby(['wind_unit','qnh_unit','visibility_unit'],dropna=False):
            formats.append(dict(airport=job['airport'],year=job['year'],wind_unit=wind,qnh_unit=qnh,
                visibility_unit=visibility,reports=len(part),auto_reports=int(part.auto_reported.sum()),
                cor_reports=int(part.correction_reported.sum())))
        print('verified',job['airport'],job['year'],len(frame),flush=True)
    assert seen.all() and labels==protocol['go_arounds']
    assert digest(INPUT)==protocol['input_sha256']
    pd.DataFrame(counts).to_csv(report/'12_full_field_completeness.csv',index=False)
    pd.DataFrame(gaps).to_csv(report/'12_full_age_and_missingness.csv',index=False)
    pd.DataFrame(formats).to_csv(report/'12_full_formats.csv',index=False)
    scenarios=pd.read_csv(report/'12_full_scenarios.csv')
    for _,part in scenarios.groupby(['latency_minutes','maximum_age_minutes','quality','exclude_cor']):
        assert int(part.total.sum())==protocol['rows']
        assert int(part.loc[part.GoA.eq(True),'total'].sum())==protocol['go_arounds']
        assert (part.candidates<=part.total).all() and part.candidates.ge(0).all()
    write_json(report/'12_full_integrity.json',dict(rows=protocol['rows'],go_arounds=labels,
        all_source_rows_present_once=True,strict_previous_observation=True,input_unchanged=True,
        input_sha256=protocol['input_sha256'],feature_files=sources,rows_deleted=0))
    print('GLOBAL INTEGRITY VERIFIED',protocol['rows'],labels,flush=True)

if __name__=='__main__':main()
