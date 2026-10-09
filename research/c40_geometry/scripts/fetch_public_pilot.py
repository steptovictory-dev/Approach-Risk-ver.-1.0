"""Один публичный час OpenSky, потоковая фильтрация выбранного борта."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import tarfile
import pandas as pd
import requests
import argparse

STUDY=Path(__file__).resolve().parents[1]


def main(hour='22'):
    listings=json.loads((STUDY/'reports/05_public_day_listings.json').read_text(encoding='utf8'))
    objects=[obj for item in listings for obj in item.get('objects',[])]
    obj=next(obj for obj in objects if obj['Key'].endswith(f'/{hour}/states_2019-10-21-{hour}.csv.tar'))
    assert int(obj['Size'])<400*1024*1024
    url='https://s3.opensky-network.org/data-samples/'+obj['Key']
    folder=STUDY/'data/public_pilot';folder.mkdir(parents=True,exist_ok=True)
    path=folder/f'states_2019-10-21-{hour}.csv.tar'
    manifest_path=STUDY/f'reports/05_public_pilot_download{ "" if hour=="22" else "_23"}.json'
    if not path.exists():
        response=requests.get(url,stream=True,timeout=(15,60))
        response.raise_for_status()
        partial=path.with_suffix('.tar.partial')
        total=0
        with partial.open('wb') as stream:
            for chunk in response.iter_content(1024*1024):
                total+=len(chunk)
                if total>400*1024*1024:raise ValueError('Превышен лимит пилота')
                stream.write(chunk)
        assert total==int(obj['Size'])
        partial.replace(path)
    sha=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):sha.update(block)
    manifest=dict(url=url,checked_utc=datetime.now(timezone.utc).isoformat(),bytes=path.stat().st_size,
                  sha256=sha.hexdigest(),catalog_etag=obj.get('ETag'),archive=str(path.relative_to(STUDY)),status='downloaded')
    manifest_path.write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf8')
    print('Архив сохранен:',path.stat().st_size,'байт',flush=True)
    selected=[];rows=0
    with tarfile.open(path) as archive:
        members=[member for member in archive.getmembers() if member.isfile()]
        print('Файлы архива:',[(m.name,m.size) for m in members][:5],flush=True)
        for member in members:
            if not (member.name.endswith('.csv') or member.name.endswith('.csv.gz')):continue
            stream=archive.extractfile(member)
            for chunk in pd.read_csv(stream,compression='gzip' if member.name.endswith('.gz') else None,chunksize=100000,dtype={'icao24':'string'},low_memory=False):
                rows+=len(chunk)
                if 'icao24' not in chunk.columns:raise ValueError('Не обнаружена колонка icao24')
                match=chunk[chunk.icao24.str.lower()=='489221'].copy()
                if len(match):selected.append(match)
    manifest.update(scanned_rows=rows,aircraft='489221',selected_rows=sum(map(len,selected)))
    if selected:
        frame=pd.concat(selected,ignore_index=True)
        selected_path=folder/f'ENT73JH_21100_state_vectors{ "" if hour=="22" else "_23"}.csv'
        frame.to_csv(selected_path,index=False)
        manifest['selected_file']=str(selected_path.relative_to(STUDY))
        print(frame.head(3).to_string(index=False),flush=True)
    manifest_path.write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf8')
    print(json.dumps(manifest),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--hour',choices=['22','23'],default='22')
    main(parser.parse_args().hour)
