"""Другой день OpenSky: выбор рейсов до просмотра результатов модели."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import tarfile
import xml.etree.ElementTree as ET
import pandas as pd
import pyarrow.parquet as pq
import requests

STUDY = Path(__file__).resolve().parents[1]
ROOT = STUDY.parents[1]


def main(day='2019-10-28', folder=None, report_prefix='10_external', max_cases=10):
    start = pd.Timestamp(day+' 22:00:00',tz='UTC')
    end = start+pd.Timedelta(hours=1)
    folder = Path(folder) if folder is not None else STUDY/'data/external_day'
    folder.mkdir(parents=True,exist_ok=True)
    model = STUDY/'data/public_pilot/08_ridge_model.npz'
    fields = ['flight_id','icao24','airport','C40_CROSS_TIME','landing_time']
    frame = pq.read_table(ROOT/'europe/data/analytical/06_05_cases_excluded.parquet',columns=fields,
                         filters=[('airport','in',['EGCC','EPWA']),('C40_CROSS_TIME','>=',start.to_pydatetime()),
                                  ('C40_CROSS_TIME','<',end.to_pydatetime())]).to_pandas(ignore_metadata=True)
    # Окно целиком помещается в один публичный час
    cases = frame[(frame.C40_CROSS_TIME >= start+pd.Timedelta(minutes=5)) &
                  (frame.landing_time <= end-pd.Timedelta(minutes=2)) &
                  (frame.landing_time > frame.C40_CROSS_TIME)].sort_values(['C40_CROSS_TIME','flight_id']).head(max_cases)
    if cases.empty:
        raise ValueError('Нет кандидатов на заранее выбранную дату')
    assert cases.flight_id.is_unique
    protocol_path = STUDY/f'reports/{report_prefix}_protocol.json'
    protocol = {'version':'10-v1' if report_prefix == '10_external' else '11-data-v1','day':day,'hour':'22','airports':['EGCC','EPWA'],
                'selection':f'first {max_cases} chronological C40 cases, C40 >=22:05, landing <=22:58 UTC',
                'cases':cases.flight_id.tolist(),'model_sha256':hashlib.sha256(model.read_bytes()).hexdigest(),
                'refit':False,'max_archive_bytes':400*1024*1024,'written_utc':datetime.now(timezone.utc).isoformat()}
    if protocol_path.exists():
        old = json.loads(protocol_path.read_text(encoding='utf8'))
        for key in ['day','cases','model_sha256']:
            assert old[key] == protocol[key], key
    else:
        protocol_path.write_text(json.dumps(protocol,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    cases.to_csv(folder/'cases.csv',index=False)
    print('Выбраны до оценки модели:',cases[['flight_id','airport']].to_dict('records'),flush=True)
    # Проверяем обычный и исторический скрытый префиксы каталога
    objects = []
    for prefix in [f'states/{day}/22/',f'states/.{day}/22/']:
        response = requests.get('https://s3.opensky-network.org/data-samples',
            params={'list-type':'2','prefix':prefix,'max-keys':1000},timeout=(15,40))
        response.raise_for_status()
        for node in ET.fromstring(response.content):
            if node.tag.split('}')[-1] == 'Contents':
                objects.append({c.tag.split('}')[-1]:c.text for c in node})
    obj = next(o for o in objects if o['Key'].endswith(f'states_{day}-22.csv.tar'))
    assert int(obj['Size']) < protocol['max_archive_bytes']
    url = 'https://s3.opensky-network.org/data-samples/'+obj['Key']
    path = folder/f'states_{day}-22.csv.tar'
    if not path.exists():
        partial = path.with_suffix('.tar.partial')
        r = requests.get(url,stream=True,timeout=(15,60))
        r.raise_for_status()
        size = 0
        with partial.open('wb') as stream:
            for block in r.iter_content(1024*1024):
                size += len(block)
                if size > protocol['max_archive_bytes']: raise ValueError('Лимит размера')
                stream.write(block)
        assert size == int(obj['Size'])
        partial.replace(path)
    assert path.stat().st_size == int(obj['Size'])
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''): digest.update(block)
    manifest = {'url':url,'bytes':path.stat().st_size,'sha256':digest.hexdigest(),'etag':obj.get('ETag'),
                'checked_utc':datetime.now(timezone.utc).isoformat(),'cases':cases.flight_id.tolist()}
    selected = []
    ids = set(cases.icao24.str.lower())
    with tarfile.open(path) as archive:
        for member in archive.getmembers():
            if not member.isfile() or not member.name.endswith(('.csv','.csv.gz')): continue
            for chunk in pd.read_csv(archive.extractfile(member),compression='gzip' if member.name.endswith('.gz') else None,
                                     chunksize=100000,dtype={'icao24':'string'},low_memory=False):
                hit = chunk[chunk.icao24.str.lower().isin(ids)].copy()
                if len(hit): selected.append(hit)
    observed = pd.concat(selected,ignore_index=True).sort_values('time')
    observed.to_csv(folder/'state_vectors.csv',index=False)
    tracks, audits = [], []
    for case in cases.itertuples():
        t = pd.Timestamp(case.C40_CROSS_TIME).timestamp()
        landing = pd.Timestamp(case.landing_time).timestamp()
        track = observed[(observed.icao24.str.lower() == case.icao24.lower()) &
                         observed.time.between(t-300,landing+120)].copy()
        signs = sorted(track.callsign.dropna().str.strip().unique().tolist())
        expected = case.flight_id.rsplit('_',1)[0]
        # Не переименовываем неожиданные callsign и не заменяем пропавшие рейсы другими
        match = expected in signs
        audits.append({'flight_id':case.flight_id,'rows':len(track),'expected_callsign':expected,
                       'callsigns':signs,'callsign_present':match})
        if match and len(track):
            track['case_flight_id'] = case.flight_id
            track['airport'] = case.airport
            tracks.append(track)
    if not tracks: raise ValueError('Не найдено сопоставленных рейсов')
    combined = pd.concat(tracks,ignore_index=True)
    output = folder/'external_geometry.csv'
    combined.to_csv(output,index=False)
    manifest.update(flights=audits,selected_sha256=hashlib.sha256(output.read_bytes()).hexdigest())
    (STUDY/f'reports/{report_prefix}_download.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps(manifest,ensure_ascii=False,indent=2),flush=True)


if __name__ == '__main__': main()
