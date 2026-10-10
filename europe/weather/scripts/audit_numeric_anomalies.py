"""Диагностический аудит без изменения погодных признаков и исходной выборки."""
import gzip
import hashlib
import json
from datetime import datetime, timezone
import re
import requests
import pandas as pd
from audit_pilot_input import BASE

R = BASE / 'reports'
CACHE = BASE / 'data/anomaly_source_audit'
CACHE.mkdir(parents=True, exist_ok=True)
progress = json.loads((R / '12_full_progress.json').read_text(encoding='utf8'))
extremes = pd.read_csv(R / '12_full_extreme_value_examples.csv')
selected, neighbors, checks = [], [], []
for job in progress['jobs']:
    key = f"{job['airport']}_{job['year']}"
    messages = pd.read_parquet(BASE / f'data/full/results/{key}_reports.parquet')
    # 850–1100 гПа — широкий диагностический интервал, не норматив и не правило очистки.
    rules = {'qnh_outside_diagnostic_interval': messages.qnh_hpa.notna() & ~messages.qnh_hpa.between(850, 1100),
             'dewpoint_above_temperature': messages.dewpoint_c.gt(messages.temperature_c),
             'observed_extreme': messages.metar_id.isin(extremes.metar_id)}
    for reason, mask in rules.items():
        part = messages.loc[mask, ['metar_id','station','metar_utc','metar','qnh_hpa','temperature_c','dewpoint_c','cloud_ceiling_ft','decoded_candidate','core_validated']].copy()
        part['reason'] = reason
        selected.append(part)
    targets = extremes.loc[extremes.station.eq(job['airport']) & pd.to_datetime(extremes.metar_utc, utc=True).dt.year.eq(int(job['year']))]
    messages['metar_utc'] = pd.to_datetime(messages.metar_utc, utc=True)
    other_mask = rules['qnh_outside_diagnostic_interval'] | rules['dewpoint_above_temperature']
    for candidate in messages.loc[other_mask & ~messages.metar_id.isin(extremes.metar_id)].itertuples():
        stamp = candidate.metar_utc
        window = messages.loc[messages.metar_utc.between(stamp-pd.Timedelta(hours=3), stamp+pd.Timedelta(hours=3)),
            ['metar_id','station','metar_utc','metar','qnh_hpa','temperature_c','dewpoint_c','cloud_ceiling_ft']].sort_values('metar_utc').copy()
        window['case_id'] = candidate.metar_id
        window['offset_minutes'] = (window.metar_utc-stamp).dt.total_seconds()/60
        neighbors.append(window)
    for case in targets.itertuples():
        stamp = pd.Timestamp(case.metar_utc)
        window = messages.loc[messages.metar_utc.between(stamp-pd.Timedelta(hours=3), stamp+pd.Timedelta(hours=3)),
            ['metar_id','station','metar_utc','metar','qnh_hpa','temperature_c','dewpoint_c','cloud_ceiling_ft']].sort_values('metar_utc').copy()
        window['case_id'] = case.metar_id
        window['offset_minutes'] = (window.metar_utc-stamp).dt.total_seconds()/60
        neighbors.append(window)
        for hour in pd.date_range(stamp.floor('h'), periods=3, freq='h'):
            name = hour.strftime('%y%m%d%H.sao.gz')
            directory = 'https://mesonet-longterm.agron.iastate.edu/archive/raw/sao/' + hour.strftime('%Y_%m/')
            url = directory + name
            path = CACHE / name
            record = dict(case_id=case.metar_id, url=url, archive_hour_utc=str(hour), independent_provider=False)
            try:
                if not path.exists():
                    listing = requests.get(directory, timeout=45)
                    listing.raise_for_status()
                    assert re.search(r'href=[\"\']'+re.escape(name)+r'[\"\']', listing.text), 'File absent from archive index'
                    response = requests.get(url, timeout=45)
                    response.raise_for_status()
                    gzip.decompress(response.content)
                    path.write_bytes(response.content)
                    (path.with_suffix(path.suffix+'.json')).write_text(json.dumps(dict(url=url, retrieved_utc=datetime.now(timezone.utc).isoformat(), sha256=hashlib.sha256(response.content).hexdigest()), indent=2)+'\n', encoding='utf8')
                content = path.read_bytes()
                raw = gzip.decompress(content).decode('utf8', errors='replace')
                normalized = ' '.join(raw.split())
                record.update(status='retrieved', sha256=hashlib.sha256(content).hexdigest(), exact_text_present=' '.join(case.metar.split()) in normalized)
            except (requests.RequestException, AssertionError, OSError, EOFError) as error:
                record.update(status='unavailable', error=str(error), exact_text_present=False)
            checks.append(record)
            print(case.station, hour, record['status'], record['exact_text_present'], flush=True)

candidates = pd.concat(selected, ignore_index=True).drop_duplicates(['metar_id','reason'])
impact = []
for job in progress['jobs']:
    features = pd.read_parquet(BASE / f"data/full/results/{job['airport']}_{job['year']}_features.parquet", columns=['metar_id','source_row','GoA','status'])
    part = features.loc[features.metar_id.isin(candidates.metar_id) & features.status.eq('matched')]
    impact.append(part.groupby('metar_id').agg(linked_rows=('source_row','size'),linked_go_arounds=('GoA','sum')).reset_index())
counts = pd.concat(impact).groupby('metar_id')[['linked_rows','linked_go_arounds']].sum().reset_index()
candidates = candidates.merge(counts, on='metar_id', how='left', validate='many_to_one')
candidates[['linked_rows','linked_go_arounds']] = candidates[['linked_rows','linked_go_arounds']].fillna(0).astype(int)
candidates.to_csv(R / '15_numeric_candidates.csv', index=False)
pd.concat(neighbors, ignore_index=True).to_csv(R / '15_extreme_neighbors.csv', index=False)
pd.DataFrame(checks).to_csv(R / '15_source_archive_checks.csv', index=False)
summary = dict(scope='93 station-year blocks; QNH diagnostic interval and dewpoint-temperature consistency; four observed extremes with +/-3h neighbors', unique_candidates=int(candidates.metar_id.nunique()), candidate_reason_counts=candidates.groupby('reason').metar_id.nunique().to_dict(), rows_changed=0, rows_deleted=0, independent_source_verified=False)
(R / '15_numeric_audit.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2)+'\n', encoding='utf8')
print(summary, flush=True)
