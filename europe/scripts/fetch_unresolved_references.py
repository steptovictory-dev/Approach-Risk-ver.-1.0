from pathlib import Path
import urllib.request
import hashlib
import json
import pymupdf

ROOT = Path(__file__).resolve().parents[2]
FOLDER = ROOT / 'docs/sources/files/unresolved_aircraft_review'
FOLDER.mkdir(parents=True, exist_ok=True)
SOURCES = {
    'anac_2023.pdf': 'https://www.anac.pt/SiteCollectionDocuments/Informacao_Gestao/Relatorios_Gestao/Relatorio_de_Atividades_Gest%C3%A3o_e_Contas_2023.pdf',
    'faa_rvsm_2026_06.pdf': 'https://www.faa.gov/sites/faa.gov/files/2026-06/NAARMO_US_IGA_RVSM_Approvals_Part_135_91_Section_3_0.pdf',
    'luxembourg_2023_04_17.pdf': 'https://dac.gouvernement.lu/content/dam/gouv_dac/documents/navigabilit%C3%A9/releve-aeronefs/2023/relev-aronefs-17-04-2023.pdf',
    'romania_regops.pdf': 'https://www.caa.ro/uploads/pages/RegOPS.pdf',
}
records = []
for name, url in SOURCES.items():
    path = FOLDER / name
    try:
        request = urllib.request.Request(url, headers={'User-Agent': 'ApproachRiskResearch/1.0'})
        with urllib.request.urlopen(request, timeout=30) as response:
            data = response.read(20 * 1024**2 + 1)
        assert len(data) <= 20 * 1024**2 and data.startswith(b'%PDF'), 'Response is not a PDF or exceeds limit'
        path.write_bytes(data)
        doc = pymupdf.open(path)
        relevant = []
        for i, page in enumerate(doc):
            text = page.get_text()
            if any(code in text for code in ['CS-DOU', 'N51VE', 'LX-PHS', 'YR-BMN']):
                relevant.append(i + 1)
                page.get_pixmap(matrix=pymupdf.Matrix(1.2, 1.2)).save(str(FOLDER / f'{path.stem}_page_{i+1}.png'))
        records.append({'name': name, 'url': url, 'sha256': hashlib.sha256(data).hexdigest(),
                        'bytes': len(data), 'relevant_pdf_pages': relevant, 'status': 'retrieved'})
    except Exception as error:
        records.append({'name': name, 'url': url, 'status': 'unavailable', 'reason': str(error)})
    print(name, records[-1]['status'], flush=True)
(ROOT / 'europe/reports/preparation/05_official_sources.json').write_text(
    json.dumps({'checked_date': '2026-10-08', 'sources': records}, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
