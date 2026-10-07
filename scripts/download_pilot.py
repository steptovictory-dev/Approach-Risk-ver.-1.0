from pathlib import Path
import concurrent.futures, hashlib, json, time, urllib.request
ROOT = Path(__file__).resolve().parents[1]
MAX_BYTES = 6 * 2**30

def fetch(url, target, expected=None):
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and (expected is None or target.stat().st_size == expected):
        return target
    partial = target.with_name(target.name + '.part')
    if expected is None:
        raise ValueError('Expected length is required')
    chunk_size = 32 * 1024 * 1024
    offset = partial.stat().st_size if partial.exists() else 0
    with partial.open('ab') as output:
        while offset < expected:
            end = min(offset + chunk_size, expected) - 1
            for attempt in range(4):
                try:
                    ranged_url = url + f'?pilot_range={offset}-{end}'
                    request = urllib.request.Request(ranged_url, headers={
                        'User-Agent': 'Approach-Risk-ML-data-audit',
                        'Range': f'bytes={offset}-{end}'})
                    with urllib.request.urlopen(request, timeout=45) as response:
                        content_range = response.headers.get('Content-Range', '')
                        if response.status != 206 or not content_range.startswith(f'bytes {offset}-{end}/'):
                            raise ValueError('Server did not honor exact byte range')
                        data = response.read()
                    if len(data) != end - offset + 1:
                        raise ValueError('Incomplete range')
                    output.write(data)
                    output.flush()
                    offset += len(data)
                    if offset % (256 * 1024 * 1024) < chunk_size:
                        print(f'{target.name}: {offset / expected:.0%}', flush=True)
                    break
                except Exception:
                    if attempt == 3:
                        raise
                    time.sleep(2 ** attempt)
    if partial.stat().st_size != expected:
        raise ValueError('Incomplete download')
    partial.replace(target)
    print(f'Completed {target.name}: {expected:,} bytes', flush=True)
    return target

def checksum(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            digest.update(chunk)
    return digest.hexdigest()

def main():
    jobs = []
    for year in (2025, 2026):
        meta = json.loads((ROOT / f'docs/sources/adsblol_{year}_pilot_release.json').read_text(encoding='utf8'))
        for asset in meta['assets']:
            jobs.append((asset['url'], ROOT / f'data/raw/usa/{year}-01-15' / asset['url'].rsplit('/', 1)[-1], asset['bytes']))
    total = sum(size for _, _, size in jobs)
    if total > MAX_BYTES:
        raise ValueError(f'Pilot exceeds limit: {total}')
    print(f'Pilot budget: {total:,} bytes for two complete days; no annual downloads.', flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda args: fetch(*args), jobs))
    files = [{'path': str(path.relative_to(ROOT)).replace('\\', '/'), 'url': url, 'bytes': path.stat().st_size, 'sha256': checksum(path), 'retrieved_on': '2026-10-08'} for url, path, _ in jobs]
    eu = ROOT / 'europe/data/source/osn23_landings_merged.parquet.gzip'
    files.insert(0, {'path': str(eu.relative_to(ROOT)).replace('\\','/'), 'url': 'https://zenodo.org/records/10210178/files/osn23_landings_merged.parquet.gzip?download=1', 'bytes': eu.stat().st_size, 'sha256': checksum(eu), 'md5': '46e1b5cdb2c078b996649c6660738b9e', 'retrieved_on': 'unknown: reused local file; integrity checked 2026-10-08', 'license': 'CC-BY-4.0'})
    (ROOT / 'docs/sources/download_manifest.json').write_text(json.dumps({'verified_on': '2026-10-08', 'usa_download_limit_bytes': MAX_BYTES, 'files': files}, indent=2), encoding='utf8')
    print('Manifest with SHA256 saved.', flush=True)

if __name__ == '__main__':
    main()
