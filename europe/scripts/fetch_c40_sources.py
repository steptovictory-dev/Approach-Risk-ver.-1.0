"""Сохраняем первичные материалы C40 и контрольные суммы для воспроизводимости."""
from pathlib import Path
import hashlib
import json
import urllib.request
import pymupdf

ROOT = Path(__file__).resolve().parents[2]
FOLDER = ROOT / 'docs/sources/files/c40_review'
SOURCES = {
    'eurocontrol_avenir_pillar1_2025.pdf': 'https://www.eurocontrol.int/sites/default/files/2025-09/avenir-wg-pillar-1-final-report.pdf',
    'author_OSN_paper_code.ipynb': 'https://raw.githubusercontent.com/figuetbe/OSN23-GoA/main/OSN_paper_code.ipynb',
}


def main():
    FOLDER.mkdir(parents=True, exist_ok=True)
    records = []
    for name, url in SOURCES.items():
        path = FOLDER / name
        try:
            request = urllib.request.Request(url, headers={'User-Agent': 'ApproachRiskResearch/1.0'})
            with urllib.request.urlopen(request, timeout=30) as response:
                data = response.read(20 * 1024**2 + 1)
            assert len(data) <= 20 * 1024**2
            if name.endswith('.pdf'):
                assert data.startswith(b'%PDF'), 'Получен не PDF.'
            else:
                assert 'cells' in json.loads(data), 'Получен не ноутбук.'
            path.write_bytes(data)
            records.append({'url': url, 'path': str(path.relative_to(ROOT)), 'status': 'retrieved',
                            'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest()})
            if name.endswith('.pdf'):
                with pymupdf.open(path) as document:
                    document[42].get_pixmap(matrix=pymupdf.Matrix(1.4, 1.4)).save(str(FOLDER / 'eurocontrol_pdf_page_43.png'))
            else:
                cells = json.loads(data)['cells']
                matches = []
                for i, cell in enumerate(cells):
                    source = ''.join(cell.get('source', []))
                    if any(word in source for word in ['C40_CROSS_TIME', '.merge(', 'drop_duplicates']):
                        matches.append({'cell_index': i, 'source': source})
                (FOLDER / 'author_c40_cells.json').write_text(json.dumps(matches, ensure_ascii=False, indent=2), encoding='utf8')
                for match in matches:
                    print('AUTHOR CELL', match['cell_index'], match['source'][:2000], flush=True)
        except Exception as error:
            records.append({'url': url, 'status': 'local_download_unavailable', 'reason': str(error),
                            'reviewed_via_web': name == 'eurocontrol_avenir_pillar1_2025.pdf'})
        print(name, records[-1]['status'], flush=True)
    paper = ROOT / 'docs/sources/files/Figuet_et_al_2023_go_arounds.pdf'
    with paper.open('rb') as stream:
        records.append({'url': 'https://doi.org/10.59490/joas.2023.7219',
                        'path': str(paper.relative_to(ROOT)), 'status': 'existing_local_copy_reviewed',
                        'sha256': hashlib.file_digest(stream, 'sha256').hexdigest(), 'relevant_pdf_pages': [3, 4, 5]})
    with pymupdf.open(paper) as document:
        document[2].get_pixmap(matrix=pymupdf.Matrix(1.4, 1.4)).save(str(FOLDER / 'article_pdf_page_3.png'))
        document[3].get_pixmap(matrix=pymupdf.Matrix(1.4, 1.4)).save(str(FOLDER / 'article_pdf_page_4.png'))
    records.append({'url': 'https://zenodo.org/records/10210178',
                    'status': 'read_via_web', 'relevant_section': 'Description, fields C40; no new local snapshot'})
    (ROOT / 'europe/reports/preparation/06_c40_sources.json').write_text(json.dumps({
        'checked_date': '2026-10-08', 'sources': records,
        'dataset_first_vs_last_entry': 'not_established',
        'newer_eurocontrol_method_is_not_proof_of_dataset_generation_method': True,
    }, ensure_ascii=False, indent=2) + '\n', encoding='utf8')


if __name__ == '__main__':
    main()
