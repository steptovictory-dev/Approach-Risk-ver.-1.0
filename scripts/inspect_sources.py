import json, urllib.request, urllib.error, re
from pathlib import Path
root=Path.cwd()
for year in (2025,2026):
    url=f'https://raw.githubusercontent.com/adsblol/globe_history_{year}/main/PREFERRED_RELEASES.txt'
    body=urllib.request.urlopen(url,timeout=60).read()
    (root/f'docs/sources/adsblol_{year}_preferred_releases.txt').write_bytes(body)
    line=next(line for line in body.decode().splitlines() if f'{year}.01.15-' in line)
    tag=line.split('/download/')[1].split('/')[0]
    urls=line.split(',')
    html=urllib.request.urlopen(f'https://github.com/adsblol/globe_history_{year}/releases/expanded_assets/{tag}',timeout=60).read().decode()
    (root/f'docs/sources/adsblol_{year}_pilot_assets.html').write_text(html,encoding='utf8')
    print(year,tag,'sizes',re.findall(r'[0-9.]+\s*[MG]B',html))
    assets=[]
    for url in urls:
        req=urllib.request.Request(url,method='HEAD')
        with urllib.request.urlopen(req,timeout=60) as r:
            assets.append({'url':url,'bytes':int(r.headers.get('Content-Length',0))})
    meta={'year':year,'date':f'{year}-01-15','tag':tag,'assets':assets,'metadata_method':'preferred releases and HTTP HEAD'}
    (root/f'docs/sources/adsblol_{year}_pilot_release.json').write_text(json.dumps(meta,indent=2),encoding='utf8')
    print('assets',assets)
try:
    meta=json.load(urllib.request.urlopen('https://zenodo.org/api/records/10210178',timeout=60))
    (root/'docs/sources/zenodo_10210178.json').write_text(json.dumps(meta,indent=2),encoding='utf8')
    print('Zenodo license',meta['metadata'].get('license'))
except Exception as e: print('Zenodo API',str(e))
import pyarrow.parquet as pq
p=pq.ParquetFile(root/'europe/data/source/osn23_landings_merged.parquet.gzip')
print('EU rows',p.metadata.num_rows,'schema',p.schema_arrow)
