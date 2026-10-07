import hashlib
import json
from pathlib import Path
from urllib.request import urlopen
import xml.etree.ElementTree as ET

import pymupdf as fitz

REGION = Path(__file__).resolve().parents[1]
CYCLE = '2610'


def main():
    catalog_url = f'https://aeronav.faa.gov/d-tpp/{CYCLE}/xml_data/d-tpp_Metafile.xml'
    with urlopen(catalog_url, timeout=45) as response:
        catalog = ET.fromstring(response.read())
    records = []
    for airport in catalog.iter('airport_name'):
        if airport.get('icao_ident') not in ['KCMA', 'KSFB']:
            continue
        code = airport.get('icao_ident')
        folder = REGION / 'documents/airports' / code / 'files'
        folder.mkdir(parents=True, exist_ok=True)
        for record in airport.findall('record'):
            if record.findtext('chart_code') not in ['IAP', 'APD']:
                continue
            filename = record.findtext('pdf_name')
            url = f'https://aeronav.faa.gov/d-tpp/{CYCLE}/{filename}'
            with urlopen(url, timeout=45) as response:
                raw = response.read()
            assert raw.startswith(b'%PDF')
            target = folder / filename
            target.write_bytes(raw)
            doc = fitz.open(target)
            doc[0].get_pixmap(matrix=fitz.Matrix(1.7, 1.7)).save(str(target.with_suffix('.png')))
            (Path('tmp') / (code + '_' + filename + '.txt')).write_text(
                '\n'.join(page.get_text() for page in doc), encoding='utf-8')
            records.append({'airport': code, 'title': record.findtext('chart_name'),
                            'cycle': CYCLE, 'from': catalog.get('from_edate'), 'to': catalog.get('to_edate'),
                            'url': url, 'file': str(target.relative_to(REGION)),
                            'sha256': hashlib.sha256(raw).hexdigest(), 'pages': len(doc),
                            'historical_2019': False})
    (REGION / 'documents/airports/chart_manifest.json').write_text(
        json.dumps(records, ensure_ascii=False, indent=2), encoding='utf-8')
    print('Official FAA charts downloaded:', len(records))


if __name__ == '__main__':
    main()
