from pathlib import Path
import urllib.request
import json
import hashlib

root = Path(__file__).resolve().parents[2]
folder = root / 'docs/sources/files/aircraft_code_review'
folder.mkdir(parents=True, exist_ok=True)
sources = {
    'icao_designators.html': 'https://www.icao.int/sites/default/files/publications/Doc8643/SiteAssets/designators.html',
    'author_notebook.ipynb': 'https://raw.githubusercontent.com/figuetbe/OSN23-GoA/main/OSN_paper_code.ipynb',
    'faa_7360_1D.pdf': 'https://www.faa.gov/documentLibrary/media/Order/Order_7360.1D_Aircraft_Type_Designators_FINAL.pdf',
    'icao_foreword.pdf': 'https://cfapps.icao.int/doc8643/Fo.en.pdf',
}
manifest = []
for name, url in sources.items():
    path = folder / name
    try:
        if not path.exists():
            request = urllib.request.Request(url, headers={'User-Agent': 'ApproachRiskResearch/1.0'})
            with urllib.request.urlopen(request, timeout=60) as response:
                data = response.read(20 * 1024**2 + 1)
            if len(data) > 20 * 1024**2:
                raise ValueError('Reference exceeds 20 MiB limit')
            path.write_bytes(data)
        manifest.append({'name': name, 'url': url, 'bytes': path.stat().st_size,
                         'sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'status': 'retrieved'})
        print(name, path.stat().st_size, flush=True)
    except Exception as error:
        manifest.append({'name': name, 'url': url, 'status': 'failed', 'error': str(error)})
        print(name, type(error).__name__, str(error), flush=True)
report = root / 'europe/reports/preparation/aircraft_reference_sources.json'
report.write_text(json.dumps({'checked_date': '2026-10-08', 'sources': manifest}, indent=2) + '\n', encoding='utf8')
