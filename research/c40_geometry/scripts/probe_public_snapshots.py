"""Проверяет только публичные каталоги OpenSky без аутентификации."""
from pathlib import Path
import json
import requests
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
import pandas as pd


def inspect_day(day):
    results=[]
    for prefix in [f'states/{day}/',f'states/.{day}/']:
        url='https://s3.opensky-network.org/data-samples'
        entry=dict(day=day,prefix=prefix)
        try:
            response=requests.get(url,params={'list-type':'2','prefix':prefix,'max-keys':1000},timeout=25)
            response.raise_for_status()
            root=ET.fromstring(response.content)
            objects=[]
            for node in root:
                if node.tag.split('}')[-1]=='Contents':
                    record={child.tag.split('}')[-1]:child.text for child in node}
                    objects.append(record)
            entry.update(status='listed',objects=objects)
        except Exception as error:entry.update(status='failed',error=str(error))
        results.append(entry)
    return results

STUDY=Path(__file__).resolve().parents[1]


def main():
    folder=STUDY/'literature/files/public_snapshots';folder.mkdir(parents=True,exist_ok=True)
    urls=[('browser','https://opensky-network.org/datasets/'),
          ('root','https://s3.opensky-network.org/data-samples?list-type=2&prefix=states/&delimiter=/'),
          ('pilot_day','https://s3.opensky-network.org/data-samples?list-type=2&prefix=states/2019-10-21/&delimiter=/')]
    results=[]
    for name,url in urls:
        entry=dict(name=name,url=url)
        try:
            response=requests.get(url,timeout=25)
            entry['http_status']=response.status_code;entry['final_url']=response.url
            response.raise_for_status()
            (folder/f'{name}.txt').write_text(response.text,encoding='utf8')
            entry['status']='saved'
            if name!='browser':
                node=ET.fromstring(response.content)
                entry['items']=[element.text for element in node.iter() if element.tag.split('}')[-1] in ('Key','Prefix')]
                print(name,entry['items'][:40],flush=True)
            else:print(response.text[:2000],flush=True)
        except Exception as error:entry.update(status='failed',error=str(error));print(name,entry['error'],flush=True)
        results.append(entry)
    (STUDY/'reports/05_public_catalog_probe.json').write_text(json.dumps(results,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    candidates=pd.read_csv(STUDY/'reports/05_public_monday_candidates.csv')
    days=sorted(set(pd.to_datetime(candidates.C40_CROSS_TIME).dt.strftime('%Y-%m-%d')))
    listings=[]
    with ThreadPoolExecutor(max_workers=3) as pool:
        for batch in pool.map(inspect_day,days):
            listings.extend(batch)
            for entry in batch:print(entry['prefix'],len(entry.get('objects',[])),entry.get('objects',[])[:2],flush=True)
    (STUDY/'reports/05_public_day_listings.json').write_text(json.dumps(listings,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    for name in ['README.txt','LICENSE.txt']:
        response=requests.get(f'https://s3.opensky-network.org/data-samples/states/{name}',timeout=25)
        response.raise_for_status()
        (folder/name).write_text(response.text,encoding='utf8')
        print(name,response.text[:3000],flush=True)


if __name__=='__main__':main()
