"""Возобновляемая обработка всей актуальной базы; обучение не выполняется."""
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path
import json
import re
import sys
import time
import requests
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from audit_pilot_input import BASE, INPUT, digest
from decode_metar import decode
from core_audit import audit_core

DATA=BASE/'data/full'
R=BASE/'reports'
PROTOCOL=R/'11_full_protocol.json'
COUNTRIES={'ED':'DE','EG':'GB','EH':'NL','EI':'IE','EK':'DK','EP':'PL','ES':'SE',
    'LE':'ES','LF':'FR','LI':'IT','LO':'AT','LS':'CH'}
def write_json(path,value):
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    temporary.replace(path)

def prepare():
    counts=pd.read_csv(R/'07_c40_month_counts.csv')
    audit=json.loads((R/'01_input_audit.json').read_text(encoding='utf8'))
    assert digest(INPUT)==audit['input_sha256']
    assert int(counts.rows.sum())==audit['rows'] and int(counts.go_arounds.sum())==audit['go_arounds']
    metadata=[]
    rawdir=BASE/'data/station_metadata'
    rawdir.mkdir(parents=True,exist_ok=True)
    for airport in sorted(counts.airport.unique()):
        network=COUNTRIES[airport[:2]]+'__ASOS'
        url=f'https://mesonet.agron.iastate.edu/geojson/network/{network}.geojson'
        path=rawdir/f'{network}.geojson'
        sidecar=path.with_suffix('.json')
        if not path.exists():
            response=requests.get(url,timeout=60)
            response.raise_for_status()
            assert response.json()['type']=='FeatureCollection'
            path.write_bytes(response.content)
            write_json(sidecar,dict(url=url,sha256=digest(path),downloaded_utc=datetime.now(timezone.utc).isoformat()))
        source=json.loads(sidecar.read_text(encoding='utf8'))
        assert source['url']==url and source['sha256']==digest(path)
        features=json.loads(path.read_text(encoding='utf8'))['features']
        found=[f for f in features if f['id']==airport]
        assert len(found)==1,(airport,network)
        item=found[0]
        metadata.append(dict(airport=airport,network=network,properties=item['properties'],
            coordinates=item['geometry']['coordinates'],**source))
        print('metadata',airport,item['properties']['sname'],flush=True)
    jobs=[]
    for (airport,year),part in counts.assign(year=counts.month.str[:4]).groupby(['airport','year']):
        first,last=part.month.min(),part.month.max()
        start=pd.Timestamp(first+'-01',tz='UTC')
        end=pd.Timestamp(last+'-01',tz='UTC')+pd.offsets.MonthBegin(1)
        jobs.append(dict(airport=airport,year=year,start_utc=start.isoformat(),end_utc=end.isoformat(),
            months=sorted(part.month.tolist()),rows=int(part.rows.sum()),go_arounds=int(part.go_arounds.sum())))
    protocol=dict(created_utc=datetime.now(timezone.utc).isoformat(),input_sha256=audit['input_sha256'],
        rows=audit['rows'],go_arounds=audit['go_arounds'],airports=len(metadata),groups=len(counts),
        selection='All nonempty airport-month groups in the current analytical input, by C40 UTC; no label-based selection',
        primary_age_minutes=60,sensitivity_age_minutes=[30,60,90],assumed_latency_minutes=[0,5,10,15],
        observation_time_is_availability=False,training_executed=False,station_metadata=metadata,jobs=jobs)
    if PROTOCOL.exists():
        existing=json.loads(PROTOCOL.read_text(encoding='utf8'))
        for key in ['input_sha256','rows','go_arounds','selection','station_metadata','jobs']:
            assert existing[key]==protocol[key],key
    else:
        assert not (R/'12_full_progress.json').exists()
        write_json(PROTOCOL,protocol)
    print('protocol',len(jobs),'station-year requests',len(counts),'airport-month groups',flush=True)

def partition_events(protocol):
    directory=DATA/'events'
    directory.mkdir(parents=True,exist_ok=True)
    marker=directory/'manifest.json'
    if marker.exists():
        previous=json.loads(marker.read_text(encoding='utf8'))
        assert previous['input_sha256']==protocol['input_sha256']
        for source in previous['files']:
            assert digest(directory/source['name'])==source['sha256']
        return
    writers={}
    sizes={}
    offset=0
    try:
        for batch in pq.ParquetFile(INPUT).iter_batches(columns=['flight_id','icao24','airport','landing_time','GoA','C40_CROSS_TIME'],batch_size=200000):
            frame=batch.to_pandas()
            frame['source_row']=range(offset,offset+len(frame))
            offset+=len(frame)
            frame['year']=frame.C40_CROSS_TIME.dt.strftime('%Y')
            for (airport,year),part in frame.groupby(['airport','year']):
                key=f'{airport}_{year}'
                table=pa.Table.from_pandas(part.drop(columns='year'),preserve_index=False)
                if key not in writers:
                    writers[key]=pq.ParquetWriter(directory/f'{key}.parquet',table.schema)
                    sizes[key]=0
                writers[key].write_table(table)
                sizes[key]+=len(part)
    finally:
        for writer in writers.values():writer.close()
    assert offset==protocol['rows']
    for job in protocol['jobs']:
        assert sizes[f"{job['airport']}_{job['year']}"]==job['rows']
    files=[dict(name=f'{key}.parquet',rows=size,sha256=digest(directory/f'{key}.parquet')) for key,size in sorted(sizes.items())]
    write_json(marker,dict(input_sha256=protocol['input_sha256'],files=files))
    print('partitioned',offset,'rows',flush=True)

def fetch(job):
    airport,year=job['airport'],job['year']
    directory=DATA/'raw'
    directory.mkdir(parents=True,exist_ok=True)
    path=directory/f'{airport}_{year}.csv'
    sidecar=path.with_suffix('.json')
    start=pd.Timestamp(job['start_utc'])-pd.Timedelta(days=1)
    end=pd.Timestamp(job['end_utc'])
    params=dict(station=airport,data='metar',sts=start.isoformat(),ets=end.isoformat(),tz='Etc/UTC',
        format='onlycomma',report_type=[3,4],direct='no')
    url=requests.Request('GET','https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py',params=params).prepare().url
    if not path.exists():
        for attempt in range(3):
            try:
                response=requests.get(url,timeout=60)
                response.raise_for_status()
                frame=pd.read_csv(StringIO(response.text),dtype=str,keep_default_na=False,comment='#')
                assert {'station','valid','metar'}<=set(frame.columns)
                break
            except (requests.RequestException,AssertionError,pd.errors.ParserError) as error:
                if attempt==2:raise
                print('retry',airport,year,type(error).__name__,flush=True)
                time.sleep(2)
        path.write_bytes(response.content)
        write_json(sidecar,dict(url=url,sha256=digest(path),downloaded_utc=datetime.now(timezone.utc).isoformat()))
        time.sleep(1.1)
    source=json.loads(sidecar.read_text(encoding='utf8'))
    assert source['url']==url and source['sha256']==digest(path)
    raw=pd.read_csv(path,dtype=str,keep_default_na=False,comment='#')
    raw['metar_utc']=pd.to_datetime(raw.valid,utc=True).astype('datetime64[ns, UTC]')
    assert raw.station.eq(airport).all() and raw.metar_utc.ge(start).all() and raw.metar_utc.lt(end).all()
    if len(raw):
        timestamps=raw.metar.str.extract(r'\b'+airport+r'\s+(\d{6})Z\b')[0]
        assert timestamps.notna().all() and timestamps.eq(raw.metar_utc.dt.strftime('%d%H%M')).all()
    return raw,source

def process(job,protocol):
    key=f"{job['airport']}_{job['year']}"
    directory=DATA/'results'
    directory.mkdir(parents=True,exist_ok=True)
    marker=directory/f'{key}.json'
    implementation=dict(decoder=digest(BASE/'scripts/decode_metar.py'),core_audit=digest(BASE/'scripts/core_audit.py'),
        pipeline=digest(Path(__file__)))
    if marker.exists():
        result=json.loads(marker.read_text(encoding='utf8'))
        assert result['input_sha256']==protocol['input_sha256']
        for item in result['files']:
            assert digest(directory/item['name'])==item['sha256']
        if result['implementation']==implementation:
            return result
        print('recompute after implementation change',key,flush=True)
    raw,source=fetch(job)
    unique=raw.drop_duplicates(['metar_utc','metar']).copy()
    unique['ambiguous']=unique.duplicated('metar_utc',keep=False)
    records=[]
    for row in unique.itertuples(index=False):
        result=decode(row.metar,row.station,row.metar_utc.to_pydatetime())
        valid,prefix,issues=audit_core(row.metar,row.station,row.metar_utc,result)
        result.update(core_validated=valid,core_prefix=prefix,core_issues=issues)
        records.append(result)
    # Пустая выгрузка не подменяется другой станцией и не теряет события.
    if not records:
        empty=decode(f"{job['airport']} 010000Z",job['airport'],pd.Timestamp(job['start_utc']).to_pydatetime())
        empty.update(core_validated=False,core_prefix='',core_issues='no_reports')
        reports=pd.DataFrame([empty]).iloc[:0].copy()
    else:reports=pd.DataFrame(records)
    reports['metar_utc']=pd.to_datetime(reports.metar_utc,utc=True).astype('datetime64[ns, UTC]')
    assert reports.metar_id.is_unique
    reports.to_parquet(directory/f'{key}_reports.parquet',index=False)
    events=pd.read_parquet(DATA/'events'/f'{key}.parquet').sort_values('C40_CROSS_TIME')
    assert len(events)==job['rows'] and events.source_row.is_unique and int(events.GoA.sum())==job['go_arounds']
    right=unique.merge(reports[['metar_utc','metar','metar_id','decoded_candidate','core_validated','correction_reported']],
        on=['metar_utc','metar'],how='left',validate='one_to_one').drop_duplicates('metar_utc').sort_values('metar_utc')
    rows=[]
    baseline=None
    for latency in protocol['assumed_latency_minutes']:
        table=right.copy()
        table['assumed_available_utc']=table.metar_utc+pd.Timedelta(minutes=latency)
        joined=pd.merge_asof(events,table[['assumed_available_utc','metar_utc','metar','metar_id','ambiguous','decoded_candidate','core_validated','correction_reported']],
            left_on='C40_CROSS_TIME',right_on='assumed_available_utc',direction='backward',allow_exact_matches=False)
        joined['age_minutes']=(joined.C40_CROSS_TIME-joined.metar_utc).dt.total_seconds()/60
        assert joined.loc[joined.metar_utc.notna(),'age_minutes'].gt(latency).all()
        joined['month']=joined.C40_CROSS_TIME.dt.strftime('%Y-%m')
        if latency==0:baseline=joined.copy()
        for (month,label),part in joined.groupby(['month','GoA']):
            for age in protocol['sensitivity_age_minutes']:
                timely=part.age_minutes.le(age)&~part.ambiguous.eq(True)
                for quality in ['observation','decoded_candidate','core_validated']:
                    candidate=timely if quality=='observation' else timely&part[quality].eq(True)
                    for without_cor in [False,True]:
                        accepted=candidate&~part.correction_reported.eq(True) if without_cor else candidate
                        rows.append(dict(airport=job['airport'],year=job['year'],month=month,GoA=bool(label),
                            latency_minutes=latency,maximum_age_minutes=age,quality=quality,exclude_cor=without_cor,
                            total=len(part),candidates=int(accepted.sum())))
    scenario=pd.DataFrame(rows)
    scenario.to_csv(directory/f'{key}_scenarios.csv',index=False)
    baseline['status']='matched'
    baseline.loc[baseline.metar_utc.isna(),'status']='missing'
    baseline.loc[baseline.age_minutes.gt(60),'status']='older_than_60min'
    baseline.loc[baseline.ambiguous.eq(True),'status']='ambiguous'
    baseline.loc[baseline.ambiguous.eq(True),['metar','metar_id']]=None
    enriched=baseline.merge(reports.drop(columns=['metar','metar_utc','station','decoded_candidate','core_validated','correction_reported']),
        on='metar_id',how='left',validate='many_to_one')
    assert len(enriched)==len(events) and enriched.source_row.is_unique
    enriched['availability_verified']=False
    enriched.to_parquet(directory/f'{key}_features.parquet',index=False)
    affected=baseline.groupby('metar_id').agg(matched_rows=('source_row','size'),matched_go_arounds=('GoA','sum')).reset_index()
    residual=reports.loc[~reports.decoded_candidate].merge(affected,on='metar_id',how='left',validate='one_to_one')
    residual[['matched_rows','matched_go_arounds']]=residual[['matched_rows','matched_go_arounds']].fillna(0).astype(int)
    residual.to_csv(directory/f'{key}_residuals.csv',index=False)
    # Изменение снимка архива относительно 12 месячных пилотов учитывается явно.
    comparisons=[]
    for month in job['months']:
        previous=BASE/'data/raw'/f"{job['airport']}_{month}.csv"
        if previous.exists():
            old=pd.read_csv(previous,dtype=str,keep_default_na=False)
            stamps=pd.to_datetime(old.valid,utc=True)
            subset=raw.loc[raw.metar_utc.ge(stamps.min())&raw.metar_utc.le(stamps.max())]
            oldset=set(zip(old.valid,old.metar))
            newset=set(zip(subset.valid,subset.metar))
            comparisons.append(dict(month=month,old_only=len(oldset-newset),new_only=len(newset-oldset)))
    names=[f'{key}_{suffix}' for suffix in ['reports.parquet','features.parquet','scenarios.csv','residuals.csv']]
    result=dict(airport=job['airport'],year=job['year'],rows=len(events),go_arounds=int(events.GoA.sum()),
        raw_reports=len(raw),unique_reports=len(reports),identical_duplicates=len(raw)-len(unique),
        ambiguous_times=int(unique.loc[unique.ambiguous,'metar_utc'].nunique()),
        residual_reports=len(residual),core_failed_reports=int((~reports.core_validated).sum()),
        source=source,input_sha256=protocol['input_sha256'],implementation=implementation,
        files=[dict(name=n,sha256=digest(directory/n)) for n in names],pilot_snapshot_comparison=comparisons)
    write_json(marker,result)
    return result

def run():
    protocol=json.loads(PROTOCOL.read_text(encoding='utf8'))
    assert digest(INPUT)==protocol['input_sha256']
    partition_events(protocol)
    completed=[]
    for i,job in enumerate(protocol['jobs']):
        result=process(job,protocol)
        completed.append(result)
        write_json(R/'12_full_progress.json',dict(completed=len(completed),planned=len(protocol['jobs']),
            rows_processed=sum(r['rows'] for r in completed),jobs=completed,complete=False))
        print(f"{i+1}/{len(protocol['jobs'])} {job['airport']} {job['year']}: {result['raw_reports']} METAR, {result['residual_reports']} residuals",flush=True)
    assert sum(r['rows'] for r in completed)==protocol['rows']
    assert sum(r['go_arounds'] for r in completed)==protocol['go_arounds']
    assert digest(INPUT)==protocol['input_sha256']
    write_json(R/'12_full_progress.json',dict(completed=len(completed),planned=len(protocol['jobs']),
        rows_processed=protocol['rows'],jobs=completed,complete=True,input_unchanged=True,rows_deleted=0,training_executed=False))
    scenarios=pd.concat([pd.read_csv(DATA/'results'/f"{r['airport']}_{r['year']}_scenarios.csv") for r in completed],ignore_index=True)
    scenarios.to_csv(R/'12_full_scenarios.csv',index=False)
    residuals=pd.concat([pd.read_csv(DATA/'results'/f"{r['airport']}_{r['year']}_residuals.csv",keep_default_na=False) for r in completed],ignore_index=True)
    # Предшествующие сутки могут попадать в два запроса. Кратность источника
    # учитывается в manifest, реестр сообщений сворачивается по metar_id.
    totals=residuals.groupby('metar_id')[['matched_rows','matched_go_arounds']].sum().reset_index()
    residuals.drop_duplicates('metar_id').drop(columns=['matched_rows','matched_go_arounds']).merge(
        totals,on='metar_id',validate='one_to_one').to_csv(R/'12_full_residuals.csv',index=False)
    print('COMPLETE',protocol['rows'],'rows',protocol['go_arounds'],'GoA',flush=True)

if __name__=='__main__':
    assert sys.argv[1:] in [['prepare'],['run']]
    prepare() if sys.argv[1]=='prepare' else run()
