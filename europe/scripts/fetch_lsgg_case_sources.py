"""Сохранение официальных материалов для проверки случая LSGG."""
from concurrent.futures import ThreadPoolExecutor
import json
import urllib.request
from pathlib import Path
from audit_feature_availability import ROOT, digest

FOLDER=ROOT/'docs/sources/files/lsgg_case_review'
SOURCES={
 'gva_annual_report_2017.pdf':'https://www.gva.ch/en/Downloads/Aeroport/2017_Rapport_Annuel.pdf?ext=.',
 'gva_annual_report_2018.pdf':'https://www.gva.ch/getmedia/7e61960f-cf6a-41fd-b489-ad788fbc39c3/2018_Rapport-Annuel_EN.pdf?ext=.pdf',
 'eurocontrol_runway_performance_2019.pdf':'https://www.eurocontrol.int/archive_download/all/node/12569',
 'traffic_navigation.py':'https://raw.githubusercontent.com/xoolive/traffic/master/src/traffic/algorithms/navigation.py',
}


def fetch(item):
    name,url=item
    path=FOLDER/name
    try:
        if not path.exists():
            request=urllib.request.Request(url,headers={'User-Agent':'ApproachRiskResearch/1.0'})
            with urllib.request.urlopen(request,timeout=30) as response: data=response.read(30*1024**2+1)
            assert len(data)<=30*1024**2
            if name.endswith('.pdf'): assert data.startswith(b'%PDF')
            path.write_bytes(data)
        return dict(name=name,url=url,status='retrieved',bytes=path.stat().st_size,sha256=digest(path))
    except Exception as error:
        return dict(name=name,url=url,status='failed',error=str(error))


if __name__=='__main__':
    FOLDER.mkdir(parents=True,exist_ok=True)
    with ThreadPoolExecutor(max_workers=4) as pool: result=list(pool.map(fetch,SOURCES.items()))
    path=ROOT/'europe/reports/attempts/06_02_source_manifest.json'
    path.write_text(json.dumps(dict(checked_date='2026-10-08',sources=result),ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    for item in result: print(item['name'],item['status'],item.get('error',''))
