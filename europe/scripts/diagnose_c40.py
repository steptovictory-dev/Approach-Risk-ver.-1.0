"""Диагностика времени C40 по всей базе; исходник и рабочая копия не меняются."""
from collections import Counter
from pathlib import Path
import hashlib
import json

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
WORKING = ROOT / 'europe/data/working/02_aircraft_reference_enriched.parquet'
SOURCE = ROOT / 'europe/data/source/osn23_landings_merged.parquet.gzip'
REPORT = ROOT / 'europe/reports/preparation'
REVIEW = ROOT / 'europe/data/review'
COLUMNS = ['flight_id', 'icao24', 'callsign', 'registration', 'airport', 'landing_time',
           'GoA', 'n_attempts', 'attempt_times', 'ILS', 'AC_CLASS', 'AP_C_RWY',
           'C40_CROSS_TIME', 'C40_BEARING', 'C40_CROSS_LAT', 'C40_CROSS_LON']
FLAGS = ['after_first_attempt', 'after_landing', 'after_last_attempt',
         'equal_first_attempt', 'equal_landing']


def sha256(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def attempt_summary(batch):
    """Сравниваем первый элемент с минимумом списка, не предполагая его порядок."""
    values = batch.column('attempt_times')
    times = pc.list_flatten(values).to_numpy(zero_copy_only=False).astype('datetime64[ns]')
    parents = pc.list_parent_indices(values).to_numpy(zero_copy_only=False)
    first = np.full(batch.num_rows, np.datetime64('NaT', 'ns'))
    last = first.copy()
    if len(times):
        start = np.r_[True, parents[1:] != parents[:-1]]
        end = np.r_[parents[1:] != parents[:-1], True]
        first[parents[start]] = times[start]
        last[parents[end]] = times[end]
    valid = ~np.isnat(times)
    minimum = np.full(batch.num_rows, np.iinfo(np.int64).max, dtype=np.int64)
    np.minimum.at(minimum, parents[valid], times[valid].astype(np.int64))
    minimum[minimum == np.iinfo(np.int64).max] = np.iinfo(np.int64).min
    earliest = minimum.view('datetime64[ns]')
    return first, last, earliest


def distribution(values):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    return {'count': len(values), 'unit': 'minutes',
            'min': float(values.min()) if len(values) else None,
            'max': float(values.max()) if len(values) else None,
            'quantiles': {str(q): float(np.quantile(values, q)) for q in [.01, .5, .99]} if len(values) else {}}


def main():
    REPORT.mkdir(parents=True, exist_ok=True)
    REVIEW.mkdir(parents=True, exist_ok=True)
    expected = json.loads((REPORT / '04_aircraft_reference_applied.json').read_text(encoding='utf8'))
    working_before, source_before = sha256(WORKING), sha256(SOURCE)
    assert working_before == expected['output_sha256']
    assert source_before == expected['source_sha256']
    reader = pq.ParquetFile(WORKING)
    counts, events, groups = Counter(), Counter(), Counter()
    flagged_parts, raw_parts, durations = [], [], []
    rows = go_arounds = first_not_minimum = null_lists = null_elements = 0
    partial_c40_metadata = 0
    for batch in reader.iter_batches(batch_size=100_000, columns=COLUMNS):
        frame = batch.to_pandas(ignore_metadata=True)
        first, last, earliest = attempt_summary(batch)
        crossing = frame['C40_CROSS_TIME'].to_numpy(dtype='datetime64[ns]')
        landing = frame['landing_time'].to_numpy(dtype='datetime64[ns]')
        present, has_first, has_landing = ~np.isnat(crossing), ~np.isnat(first), ~np.isnat(landing)
        first_not_minimum += int((has_first & ~np.isnat(earliest) & (first != earliest)).sum())
        null_lists += batch.column('attempt_times').null_count
        null_elements += pc.list_flatten(batch.column('attempt_times')).null_count
        metadata = frame[['AC_CLASS', 'C40_CROSS_TIME', 'C40_BEARING', 'C40_CROSS_LAT', 'C40_CROSS_LON']].notna()
        partial_c40_metadata += int((metadata.any(axis=1) & ~metadata.all(axis=1)).sum())
        flags = {
            'after_first_attempt': present & has_first & (crossing > first),
            'after_landing': present & has_landing & (crossing > landing),
            'after_last_attempt': present & ~np.isnat(last) & (crossing > last),
            'equal_first_attempt': present & has_first & (crossing == first),
            'equal_landing': present & has_landing & (crossing == landing),
        }
        category = np.select([~present, ~has_first | ~has_landing, flags['after_landing'],
                              flags['after_first_attempt'], flags['equal_first_attempt']],
                             ['missing_c40', 'missing_comparison_time', 'after_landing',
                              'after_first_before_or_equal_landing', 'equal_first_attempt'],
                             default='before_first_attempt')
        frame['year'] = frame['landing_time'].dt.year
        frame['c40_time_category'] = category
        ga = frame['GoA'].fillna(False).to_numpy(dtype=bool)
        for name, mask in flags.items():
            counts[name] += int(mask.sum())
            events[name] += int(ga[mask].sum())
        for key, record in frame.groupby(['airport', 'year', 'c40_time_category'], dropna=False)['GoA'].agg(['size', 'sum']).iterrows():
            groups[(key, 'rows')] += int(record['size'])
            groups[(key, 'go_arounds')] += int(record['sum'])
        available = present & has_landing
        duration = (landing[available] - crossing[available]) / np.timedelta64(1, 'm')
        durations.append(duration)
        selected_mask = flags['after_first_attempt'] | flags['after_landing'] | flags['equal_first_attempt'] | flags['equal_landing']
        if selected_mask.any():
            # Типизированная проекция исходных значений хранится отдельно от диагностических CSV.
            raw = pa.Table.from_batches([batch]).filter(pa.array(selected_mask))
            positions = rows + np.flatnonzero(selected_mask)
            raw_parts.append(raw.append_column('source_row_position', pa.array(positions)))
            selected = frame.loc[selected_mask].copy()
            selected['source_row_position'] = positions
            selected['first_attempt_time'] = pd.to_datetime(first[selected_mask], utc=True)
            selected['last_attempt_time'] = pd.to_datetime(last[selected_mask], utc=True)
            selected['earliest_attempt_time'] = pd.to_datetime(earliest[selected_mask], utc=True)
            for name, mask in flags.items():
                selected[name] = mask[selected_mask]
            selected['landing_minus_c40_minutes'] = (selected['landing_time'] - selected['C40_CROSS_TIME']).dt.total_seconds() / 60
            selected['c40_minus_first_minutes'] = (selected['C40_CROSS_TIME'] - selected['first_attempt_time']).dt.total_seconds() / 60
            selected['last_attempt_minus_c40_minutes'] = (selected['last_attempt_time'] - selected['C40_CROSS_TIME']).dt.total_seconds() / 60
            flagged_parts.append(selected)
        rows += len(frame)
        go_arounds += int(ga.sum())
        if rows % 1_000_000 == 0:
            print('Проверено строк:', rows, flush=True)
    flagged = pd.concat(flagged_parts, ignore_index=True)
    # Независимая сверка с полным аудитом 01, а не с ограниченным списком примеров.
    audit_flags = pd.read_csv(ROOT / 'europe/reports/review_flag_summary.csv').set_index('reason')
    for name in ['after_first_attempt', 'after_landing']:
        reference = audit_flags.loc['C40_CROSS_TIME:' + name]
        assert counts[name] == int(reference['rows'])
        assert events[name] == int(reference['go_arounds'])
    assert rows == expected['rows_after'] and go_arounds == expected['go_arounds_after']
    keys = sorted({key for key, metric in groups})
    coverage = pd.DataFrame([{'airport': key[0], 'year': key[1], 'category': key[2],
                             'rows': groups[(key, 'rows')], 'go_arounds': groups[(key, 'go_arounds')]} for key in keys])
    # Проверяем ближайшие записи того же адреса: окно +/-24 ч диагностическое, не правило сопоставления.
    addresses = set(flagged['icao24'])
    anchors = {address: part for address, part in flagged.groupby('icao24')}
    related_parts = []
    offset = 0
    for batch in reader.iter_batches(batch_size=100_000, columns=COLUMNS):
        frame = batch.to_pandas(ignore_metadata=True)
        frame['source_row_position'] = offset + frame.index
        candidates = frame.loc[frame['icao24'].isin(addresses)]
        for address, part in candidates.groupby('icao24'):
            near = pd.Series(False, index=part.index)
            for anchor in anchors[address].itertuples(index=False):
                near |= (part['landing_time'] - anchor.landing_time).abs().le(pd.Timedelta(hours=24))
                near |= (part['landing_time'] - anchor.C40_CROSS_TIME).abs().le(pd.Timedelta(hours=24))
            if near.any():
                related_parts.append(part.loc[near])
        offset += len(frame)
    assert offset == rows
    related = pd.concat(related_parts, ignore_index=True)
    pairs = []
    for record in flagged.itertuples(index=False):
        other = related.loc[(related['icao24'] == record.icao24) &
                            (related['source_row_position'] != record.source_row_position)]
        other = other.loc[(other['landing_time'] - record.landing_time).abs().le(pd.Timedelta(hours=24)) |
                          (other['landing_time'] - record.C40_CROSS_TIME).abs().le(pd.Timedelta(hours=24))]
        for candidate in other.itertuples(index=False):
            same_time = pd.notna(candidate.C40_CROSS_TIME) and candidate.C40_CROSS_TIME == record.C40_CROSS_TIME
            same_airport = candidate.airport == record.airport
            pairs.append({'flagged_row_position': record.source_row_position,
                          'related_row_position': candidate.source_row_position,
                          'icao24': record.icao24, 'flagged_airport': record.airport,
                          'related_airport': candidate.airport,
                          'same_airport': same_airport, 'same_c40_time': same_time,
                          'same_c40_lat_lon': bool(same_time and candidate.C40_CROSS_LAT == record.C40_CROSS_LAT and candidate.C40_CROSS_LON == record.C40_CROSS_LON),
                          'same_callsign': candidate.callsign == record.callsign,
                          'related_landing_minus_c40_minutes': (candidate.landing_time - record.C40_CROSS_TIME).total_seconds() / 60,
                          'landing_difference_minutes': (candidate.landing_time - record.landing_time).total_seconds() / 60})
    pairs = pd.DataFrame(pairs)
    flag_summary = pd.DataFrame([{'flag': name, 'rows': counts[name], 'go_arounds': events[name]} for name in FLAGS])
    classes = flagged.groupby(['c40_time_category', 'GoA', 'n_attempts']).size().rename('rows').reset_index()
    all_durations = np.concatenate(durations)
    working_after, source_after = sha256(WORKING), sha256(SOURCE)
    assert working_after == working_before and source_after == source_before
    summary = {
        'checked_date': '2026-10-08', 'input': str(WORKING.relative_to(ROOT)),
        'input_sha256': working_before, 'input_unchanged': True,
        'source_sha256': source_before, 'source_unchanged': True,
        'rows_checked': rows, 'go_arounds': go_arounds, 'rows_changed': 0, 'rows_deleted': 0,
        'flags': flag_summary.to_dict(orient='records'), 'unique_review_rows': len(flagged),
        'first_attempt_not_earliest_rows': first_not_minimum,
        'null_attempt_lists': null_lists, 'null_attempt_elements': null_elements,
        'partial_AC_CLASS_C40_metadata_rows': partial_c40_metadata,
        'categories': coverage.groupby('category')[['rows', 'go_arounds']].sum().reset_index().to_dict(orient='records'),
        'landing_minus_c40': distribution(all_durations),
        'negative_landing_minus_c40': distribution(all_durations[all_durations < 0]),
        'c40_minus_first_flagged': distribution(flagged.loc[flagged['after_first_attempt'], 'c40_minus_first_minutes']),
        'related_rows_in_diagnostic_window': len(related), 'diagnostic_window_hours': 24,
        'flagged_rows_sharing_c40_with_other_same_airport_landing': int(pairs.loc[pairs['same_airport'] & pairs['same_c40_time'], 'flagged_row_position'].nunique()) if len(pairs) else 0,
        'matching_semantics': 'candidate associations only; no historical identity or correction established',
    }
    flag_summary.to_csv(REPORT / '06_c40_flag_summary.csv', index=False)
    coverage.to_csv(REPORT / '06_c40_airport_year_categories.csv', index=False)
    classes.to_csv(REPORT / '06_c40_flagged_attempts.csv', index=False)
    flagged.to_csv(REPORT / '06_c40_records.csv', index=False)
    related.to_csv(REPORT / '06_c40_related_records.csv', index=False)
    pairs.to_csv(REPORT / '06_c40_related_pairs.csv', index=False)
    # Проекция не содержит сохраненного pandas-индекса полного файла.
    raw_projection = pa.concat_tables(raw_parts).replace_schema_metadata(None)
    pq.write_table(raw_projection, REVIEW / '06_c40_raw_projection.parquet', compression='gzip')
    (REPORT / '06_c40_diagnostics.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf8')
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return flag_summary, coverage, flagged, pairs


if __name__ == '__main__':
    main()
