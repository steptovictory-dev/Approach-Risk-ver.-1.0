"""Сохраняет исторические страницы AD 2 и журнал получения из UK AIP."""
from pathlib import Path
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import re
import requests
from html.parser import HTMLParser


class TextParser(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts = []
        self.hidden = 0
        self.stack = []
        self.metadata = {}

    def handle_starttag(self, tag, attrs):
        attributes = dict(attrs)
        if tag == 'meta' and attributes.get('name'):
            self.metadata[attributes['name']] = attributes.get('content')
        hidden = tag in ('script', 'style', 'del') or 'display: none' in attributes.get('style', '')
        if tag not in ('meta', 'br', 'hr', 'img', 'link', 'input'):
            self.stack.append((tag, hidden))
            self.hidden += int(hidden)

    def handle_endtag(self, tag):
        if self.stack and self.stack[-1][0] == tag:
            self.hidden -= int(self.stack.pop()[1])

    def handle_data(self, data):
        if not self.hidden and data.strip():
            self.parts.append(data.strip())

ROOT = Path(__file__).resolve().parents[3]
STUDY = ROOT / 'research/c40_geometry'
FILES = STUDY / 'literature/files'
SLICES = [('EGCC', '2018-12-06'), ('EGCC', '2019-01-03'),
          ('EGCC', '2019-11-07'), ('EGCC', '2020-01-30'),
          ('EGCC', '2020-02-27'), ('EGCC', '2021-02-25'),
          ('EGLL', '2019-12-05'), ('EGLL', '2023-05-18')]


def fetch(item):
    airport, airac = item
    url = f'https://www.aurora.nats.co.uk/htmlAIP/Publications/{airac}-AIRAC/html/eAIP/EG-AD-2.{airport}-en-GB.html'
    path = FILES / f'NATS_{airport}_AIRAC_{airac}_AD2.html'
    entry = dict(airport=airport, airac=airac, url=url, checked_utc=datetime.now(timezone.utc).isoformat())
    try:
        if path.exists():
            content = path.read_bytes()
        else:
            response = requests.get(url, timeout=35)
            entry['http_status'] = response.status_code
            entry['final_url'] = response.url
            response.raise_for_status()
            content = response.content
        decoded = content.decode('utf8', errors='replace')
        entry['document_dates'] = sorted(set(re.findall(r'20\d{2}-\d{2}-\d{2}', decoded)))
        # Проверяем наличие аэропорта и раздела, а не только ответ HTTP 200.
        if airport not in decoded or not re.search(r'AD.?2.?2|AD-2\.2', decoded):
            raise ValueError('Не обнаружен раздел AD 2.2; ответ сохраненным AIP не считается')
        path.write_bytes(content)
        entry.update(status='saved', file=str(path.relative_to(ROOT)), sha256=hashlib.sha256(content).hexdigest(), bytes=len(content))
        parser = TextParser()
        parser.feed(decoded)
        text = ' '.join(parser.parts)
        entry['document_metadata'] = parser.metadata
        entry['effective_date_matches_requested'] = parser.metadata.get('EM.effectiveDateStart') == airac
        if not entry['effective_date_matches_requested']:
            raise ValueError('Дата вступления в силу в HTML не совпадает с запрошенным AIRAC')
        path.with_suffix('.txt').write_text(text, encoding='utf8')
        entry['text_file'] = str(path.with_suffix('.txt').relative_to(ROOT))
        position = text.find('ARP coordinates')
        if position < 0:
            position = text.find('ARP COORD')
        entry['arp_context'] = text[max(0,position-80):position+600] if position >= 0 else 'requires_manual_inspection'
        if (airport, airac) in [('EGCC','2020-02-27'),('EGLL','2019-12-05')]:
            pdf_url = url.replace('html/eAIP/', 'pdf/').replace('-en-GB.html', '.pdf')
            pdf_path = path.with_suffix('.pdf')
            if not pdf_path.exists():
                pdf_response = requests.get(pdf_url, timeout=35)
                pdf_response.raise_for_status()
                if b'%PDF-' not in pdf_response.content[:1024]:
                    raise ValueError('Печатная версия не PDF')
                pdf_path.write_bytes(pdf_response.content)
            from pypdf import PdfReader
            pdf_text = '\n'.join(page.extract_text() or '' for page in PdfReader(pdf_path).pages)
            pdf_path.with_suffix('.pdf.txt').write_text(pdf_text, encoding='utf8')
            entry['print_pdf'] = dict(url=pdf_url, file=str(pdf_path.relative_to(ROOT)), sha256=hashlib.sha256(pdf_path.read_bytes()).hexdigest())
    except Exception as error:
        if entry.get('effective_date_matches_requested') and entry.get('arp_context'):
            entry['print_pdf_error'] = f'{type(error).__name__}: {error}'
        else:
            entry.update(status='failed', error=f'{type(error).__name__}: {error}')
    return entry


def main():
    FILES.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=3) as pool:
        results = list(pool.map(fetch, SLICES))
    (STUDY / 'literature/historical_ad_manifest.json').write_text(json.dumps(results, ensure_ascii=False, indent=2)+'\n', encoding='utf8')
    for entry in results:
        print(entry['airport'],entry['airac'],entry['status'],entry.get('arp_context',entry.get('error')), flush=True)


if __name__ == '__main__':
    main()
