"""Пилот архивного сопоставления METAR строго до C40; без обучения."""
from datetime import datetime, timezone
from io import StringIO
import json
import re
import time
import pandas as pd
import pyarrow.parquet as pq
import requests
from audit_pilot_input import ROOT, BASE, INPUT, digest


def main(expansion=False):
    audit = json.loads((BASE / 'reports/01_input_audit.json').read_text(encoding='utf8'))
    prefix = '08' if expansion else '03'
    if expansion:
        protocol = json.loads((BASE / 'reports/07_expansion_protocol.json').read_text(encoding='utf8'))
        assert protocol['input_sha256'] == audit['input_sha256']
        assert protocol['status'] == 'protocol_frozen_before_new_metar_download'
        audit['selected_airports'] = protocol['selected_airports']
    assert digest(INPUT) == audit['input_sha256']
    frames = []
    offset = 0
    columns = ['flight_id', 'icao24', 'airport', 'landing_time', 'GoA', 'C40_CROSS_TIME']
    for batch in pq.ParquetFile(INPUT).iter_batches(columns=columns, batch_size=200_000):
        frame = batch.to_pandas()
        frame['source_row'] = range(offset, offset + len(frame))
        offset += len(frame)
        month = frame.C40_CROSS_TIME.dt.strftime('%Y-%m')
        frames.append(frame.loc[frame.airport.isin(audit['selected_airports'])
                                & month.isin(['2019-01', '2019-07'])].copy())
    events = pd.concat(frames, ignore_index=True)
    assert events.source_row.is_unique
    if expansion:
        actual = events.assign(month=events.C40_CROSS_TIME.dt.strftime('%Y-%m')).groupby(
            ['airport','month']).agg(rows=('GoA','size'),go_arounds=('GoA','sum')).reset_index()
        pd.testing.assert_frame_equal(actual,pd.DataFrame(protocol['groups']),check_dtype=False)
    rawdir = BASE / 'data/raw'
    rawdir.mkdir(parents=True, exist_ok=True)
    manifests, outputs = [], []
    for airport in audit['selected_airports']:
        for month in ['2019-01', '2019-07']:
            start = pd.Timestamp(month + '-01', tz='UTC')
            end = start + pd.offsets.MonthBegin(1)
            params = dict(station=airport, data='metar', sts=(start-pd.Timedelta(days=1)).isoformat(),
                          ets=end.isoformat(), tz='Etc/UTC', format='onlycomma',
                          report_type=[3, 4], direct='no')
            url = requests.Request('GET', 'https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py',
                                   params=params).prepare().url
            path = rawdir / f'{airport}_{month}.csv'
            sidecar = path.with_suffix('.json')
            if not path.exists():
                response = requests.get(url, timeout=60)
                response.raise_for_status()
                check = pd.read_csv(StringIO(response.text), comment='#', dtype=str, keep_default_na=False)
                assert {'station', 'valid', 'metar'} <= set(check.columns) and len(check)
                path.write_bytes(response.content)
                sidecar.write_text(json.dumps(dict(url=url, sha256=digest(path),
                    downloaded_utc=datetime.now(timezone.utc).isoformat()), indent=2)+'\n', encoding='utf8')
                time.sleep(1.1)
            source = json.loads(sidecar.read_text(encoding='utf8'))
            assert source['url'] == url and source['sha256'] == digest(path)
            weather = pd.read_csv(path, comment='#', dtype=str, keep_default_na=False)
            weather['metar_utc'] = pd.to_datetime(weather.valid, utc=True).astype('datetime64[ns, UTC]')
            assert weather.station.eq(airport).all()
            assert weather.metar_utc.ge(start-pd.Timedelta(days=1)).all()
            assert weather.metar_utc.lt(end).all()
            for row in weather.itertuples():
                match = re.search(r'\b'+airport+r'\s+(\d{6})Z\b', row.metar)
                assert match and match[1] == row.metar_utc.strftime('%d%H%M'), row.metar
            unique = weather.drop_duplicates(['metar_utc', 'metar']).copy()
            unique['ambiguous'] = unique.duplicated('metar_utc', keep=False)
            unique['correction'] = unique.metar.str.contains(r'\bCOR\b', regex=True)
            # Представитель неоднозначного времени нужен только для статуса, не погодных признаков.
            right = unique.drop_duplicates('metar_utc').sort_values('metar_utc')
            part = events.loc[events.airport.eq(airport) & events.C40_CROSS_TIME.ge(start)
                              & events.C40_CROSS_TIME.lt(end)].sort_values('C40_CROSS_TIME')
            joined = pd.merge_asof(part, right[['metar_utc', 'metar', 'ambiguous', 'correction']],
                left_on='C40_CROSS_TIME', right_on='metar_utc', direction='backward',
                allow_exact_matches=False, tolerance=pd.Timedelta(minutes=90))
            joined['age_minutes'] = (joined.C40_CROSS_TIME-joined.metar_utc).dt.total_seconds()/60
            joined['month'] = month
            joined['availability_verified'] = False
            joined['status'] = 'matched'
            joined.loc[joined.metar_utc.isna(), 'status'] = 'missing_or_older_than_90min'
            joined.loc[joined.ambiguous.eq(True), 'status'] = 'ambiguous'
            joined.loc[joined.age_minutes.gt(60) & joined.status.eq('matched'), 'status'] = 'older_than_60min'
            joined.loc[joined.status.eq('ambiguous'), 'metar'] = None
            matched = joined.metar_utc.notna()
            assert joined.loc[matched, 'age_minutes'].gt(0).all()
            assert len(joined) == len(part)
            outputs.append(joined)
            manifests.append(dict(airport=airport, month=month, **source,
                raw_reports=len(weather), identical_duplicates=len(weather)-len(unique),
                ambiguous_times=int(unique.loc[unique.ambiguous, 'metar_utc'].nunique()),
                correction_reports=int(unique.correction.sum())))
            print(airport, month, len(weather), 'reports;', len(joined), 'rows', flush=True)
    result = pd.concat(outputs, ignore_index=True)
    assert len(result) == len(events) and result.source_row.is_unique
    summaries = []
    for (airport, month, label), part in result.groupby(['airport', 'month', 'GoA']):
        for age in [30, 60, 90]:
            covered = part.age_minutes.le(age) & ~part.ambiguous.eq(True) & part.metar_utc.notna()
            summaries.append(dict(airport=airport, month=month, GoA=bool(label),
                max_age_minutes=age, rows=len(part), covered=int(covered.sum()),
                coverage_percent=100*covered.mean(), ambiguous=int(part.ambiguous.eq(True).sum()),
                correction_matches=int((covered & part.correction.eq(True)).sum())))
    result.to_parquet(BASE / f'data/{prefix}_pilot_matches.parquet', index=False)
    pd.DataFrame(summaries).to_csv(BASE / f'reports/{prefix}_coverage.csv', index=False)
    assert digest(INPUT) == audit['input_sha256']
    journal = dict(input_sha256=audit['input_sha256'], input_unchanged=True, rows=len(result),
        go_arounds=int(result.GoA.sum()), downloads=manifests, availability_verified=False,
        matching='strictly previous observation; 60 min primary; ambiguous texts rejected',
        checked_utc=datetime.now(timezone.utc).isoformat(), rows_deleted=0)
    (BASE / f'reports/{prefix}_pilot_manifest.json').write_text(
        json.dumps(journal, ensure_ascii=False, indent=2)+'\n', encoding='utf8')
    print(pd.DataFrame(summaries).query('max_age_minutes == 60').to_string(index=False))


if __name__ == '__main__':
    main()
