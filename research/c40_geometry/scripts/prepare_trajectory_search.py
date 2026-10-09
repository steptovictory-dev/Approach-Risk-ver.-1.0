"""Снимки первичных источников и узкие запросы для шести случаев EGCC."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import requests
import pandas as pd

ROOT=Path(__file__).resolve().parents[3]
STUDY=ROOT/'research/c40_geometry'


def main():
    files=STUDY/'literature/files/trajectory_search'
    files.mkdir(parents=True,exist_ok=True)
    sources={
        'author_tree.json':'https://api.github.com/repos/figuetbe/OSN23-GoA/git/trees/main?recursive=1',
        'author_README.md':'https://raw.githubusercontent.com/figuetbe/OSN23-GoA/main/README.md',
        'author_notebook.ipynb':'https://raw.githubusercontent.com/figuetbe/OSN23-GoA/main/OSN_paper_code.ipynb',
        'zenodo_record.json':'https://zenodo.org/api/records/10210178',
        'opensky_trino.html':'https://openskynetwork.github.io/opensky-api/trino.html',
        'opensky_scientific.html':'https://opensky-network.org/data/scientific',
    }
    manifest=[]
    for name,url in sources.items():
        entry=dict(url=url,checked_utc=datetime.now(timezone.utc).isoformat())
        try:
            response=requests.get(url,timeout=35)
            response.raise_for_status()
            path=files/name;path.write_bytes(response.content)
            entry.update(status='saved',file=str(path.relative_to(ROOT)),sha256=hashlib.sha256(response.content).hexdigest())
        except Exception as error:entry.update(status='failed',error=str(error))
        manifest.append(entry)
    (STUDY/'literature/trajectory_search_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    notebook_path=files/'author_notebook.ipynb'
    excerpts=[]
    if notebook_path.exists():
        notebook=json.loads(notebook_path.read_text(encoding='utf8'))
        for index,cell in enumerate(notebook['cells']):
            text=''.join(cell.get('source',[]))
            if any(term.lower() in text.lower() for term in ['C40','history(','from_file','CPR','asma']):
                excerpts.append(dict(cell=index,cell_type=cell['cell_type'],source=text))
        (STUDY/'reports/04_author_code_cells.json').write_text(json.dumps(excerpts,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    queries=STUDY/'queries';queries.mkdir(exist_ok=True)
    cases=pd.read_csv(STUDY/'reports/03_egcc_case_review.csv')
    plan=[]
    for row in cases.itertuples():
        crossing=pd.Timestamp(row.C40_CROSS_TIME)
        start=crossing-pd.Timedelta(minutes=5)
        end=pd.Timestamp(row.landing_time)+pd.Timedelta(minutes=2)
        first=int(start.timestamp());last=int(end.timestamp())
        hours=list(range(first//3600*3600,(last-1)//3600*3600+1,3600))
        assert len(hours)<=2
        query=f"""-- План запроса; не выполнялся. Время UTC, исходные позиции не фильтруем по свежести.
-- {row.flight_id}; C40 {crossing.isoformat()}
SELECT time, icao24, callsign, lat, lon, velocity, heading,
       baroaltitude, geoaltitude, vertrate, onground,
       lastposupdate, lastcontact, hour
FROM minio.osky.state_vectors_data4
WHERE hour IN ({', '.join(map(str,hours))})
  AND icao24 = '{row.icao24}'
  AND time >= {first} AND time < {last}
ORDER BY time;
"""
        (queries/f'{row.flight_id}.sql').write_text(query,encoding='utf8')
        plan.append(dict(flight_id=row.flight_id,icao24=row.icao24,c40_utc=crossing.isoformat(),
           query_start_utc=start.isoformat(),query_end_utc=end.isoformat(),hour_partitions=hours,
           status='not_executed_access_not_confirmed',query_file=f'queries/{row.flight_id}.sql'))
    (STUDY/'reports/04_trajectory_query_plan.json').write_text(json.dumps(plan,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    for entry in manifest:print(entry['status'],entry['url'],flush=True)
    if (files/'author_tree.json').exists():
        tree=json.loads((files/'author_tree.json').read_text(encoding='utf8'))
        print('Git tree commit:',tree.get('sha'),'truncated:',tree.get('truncated'),flush=True)
        for item in tree.get('tree',[]):
            if item['type']=='blob':print(item['path'],item.get('size'),flush=True)
    print('Подготовлено шесть запросов с фильтрами hour и icao24; запросы не выполнялись.',flush=True)


if __name__=='__main__':main()
