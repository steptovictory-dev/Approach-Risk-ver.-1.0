"""Открытые почасовые снимки OpenSky для одной из 147 пар, без Trino."""
from pathlib import Path
import json
import gzip
import tarfile
import urllib.request
import pandas as pd
from audit_feature_availability import ROOT, REPORT, digest

FOLDER=ROOT/'europe/data/source/public_pair_134'
BASE='https://s3.opensky-network.org/data-samples'


def main():
    FOLDER.mkdir(parents=True,exist_ok=True)
    manifest=[]
    points=[]
    for hour in ['04','05']:
        name=f'states_2019-03-04-{hour}.csv.tar'
        url=f'{BASE}/states/.2019-03-04/{hour}/{name}'
        path=FOLDER/name
        if not path.exists():
            temporary=path.with_suffix('.tar.part')
            print('Downloading',name,flush=True)
            with urllib.request.urlopen(url,timeout=45) as response,temporary.open('wb') as stream:
                while chunk:=response.read(1024*1024):
                    stream.write(chunk)
            temporary.replace(path)
        print('Reading',name,flush=True)
        with tarfile.open(path) as archive:
            members=[m for m in archive.getmembers() if m.isfile() and m.name.endswith('.csv.gz')]
            assert len(members)==1
            with archive.extractfile(members[0]) as source,gzip.GzipFile(fileobj=source) as csv:
                for frame in pd.read_csv(csv,chunksize=100_000,dtype={'icao24':'string','callsign':'string','squawk':'string'}):
                    selected=frame.loc[frame.icao24.eq('780b33')].copy()
                    if len(selected): points.append(selected)
        manifest.append(dict(url=url,path=str(path.relative_to(ROOT)),bytes=path.stat().st_size,sha256=digest(path)))
    frame=pd.concat(points,ignore_index=True).sort_values('time') if points else pd.DataFrame()
    assert len(frame)>0,'Целевой борт отсутствует; нельзя подтвердить пару'
    frame.to_parquet(FOLDER/'780b33_raw_state_vectors.parquet',index=False)
    for name in ['README.txt','LICENSE.txt']:
        path=FOLDER/name
        if not path.exists():
            with urllib.request.urlopen(f'{BASE}/states/{name}',timeout=25) as response:
                path.write_bytes(response.read(100000))
    result=dict(pair_id=134,icao24='780b33',date='2019-03-04',airport='LFPG',
                retrieved_utc_date='2026-10-08',source='OpenSky public state-vector snapshots',
                independent_of_derived_landing_table=True,independent_sensor_network=False,
                rows=len(frame),files=manifest,columns=frame.columns.tolist(),
                filtered_file=str((FOLDER/'780b33_raw_state_vectors.parquet').relative_to(ROOT)),
                filtered_sha256=digest(FOLDER/'780b33_raw_state_vectors.parquet'))
    (REPORT/'04_07_public_source_manifest.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':
    main()
