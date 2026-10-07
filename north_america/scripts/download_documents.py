import hashlib
import json
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from urllib.parse import urljoin, urlparse
from urllib.request import Request, urlopen

from pypdf import PdfReader


ROOT = Path(__file__).resolve().parents[1]


class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.links = []

    def handle_starttag(self, tag, attrs):
        if tag == 'a':
            href = dict(attrs).get('href')
            if href:
                self.links.append(href)


def get(url):
    request = Request(url, headers={'User-Agent': 'ApproachRiskResearch/0.1'})
    with urlopen(request, timeout=45) as response:
        # Не принимаем редирект на сторонний домен за официальный источник
        official = ('faa.gov', 'tc.canada.ca', 'dc-ansp.org', 'caabahamas.com')
        host = urlparse(response.url).hostname or ''
        if not any(host == domain or host.endswith('.' + domain) for domain in official):
            raise ValueError('Ответ получен не с официального домена')
        return response.read(), response.url


def pdf_link(page, contains):
    content, final_url = get(page)
    parser = Links()
    parser.feed(content.decode('utf-8', errors='replace'))
    candidates = [urljoin(final_url, href) for href in parser.links
                  if '.pdf' in href.lower() and contains.lower() in href.lower()]
    if not candidates:
        raise ValueError(f'PDF со строкой {contains!r} на странице не найден')
    return candidates[0]


def main():
    sources = [
        ('usa', 'FAA_Aviation_Weather_Handbook_28B.pdf',
         'https://www.faa.gov/sites/faa.gov/files/FAA-H-8083-28B.pdf', None),
        ('usa', 'FAA_AIM_current.pdf',
         'https://www.faa.gov/air_traffic/publications/index.cfm', 'AIM_Basic'),
        ('canada', 'TC_AIM_current.pdf',
         'https://tc.canada.ca/en/aviation/publications/transport-canada-aeronautical-information-manual-tc-aim-tp-14371', 'access_en'),
        ('dutch_caribbean', 'AIP_AIRAC_03_26.pdf',
         'https://dc-ansp.org/eAIS/eaip-dutch-caribbean/', '03-26'),
    ]
    records = []
    for country, filename, page, contains in sources:
        record = {'jurisdiction': country, 'source_page': page,
                  'checked_utc': datetime.now(timezone.utc).isoformat(),
                  'review_status': 'downloaded_not_fully_reviewed'}
        try:
            url = pdf_link(page, contains) if contains else page
            content, final_url = get(url)
            if not content.startswith(b'%PDF-'):
                raise ValueError('Полученный файл не PDF')
            path = ROOT / 'documents' / country / 'files' / filename
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
            reader = PdfReader(path)
            record.update({
                'url': final_url, 'file': str(path.relative_to(ROOT)),
                'bytes': len(content), 'sha256': hashlib.sha256(content).hexdigest(),
                'pages': len(reader.pages), 'status': 'downloaded',
            })
            print(f'{country}: {filename}, {len(reader.pages)} pages', flush=True)
        except Exception as error:
            record.update({'status': 'unavailable', 'error': str(error)})
            print(f'{country}: {error}', flush=True)
        records.append(record)
        (ROOT / 'documents' / 'download_manifest.json').write_text(
            json.dumps(records, ensure_ascii=False, indent=2) + '\n', encoding='utf-8',
        )


if __name__ == '__main__':
    main()
