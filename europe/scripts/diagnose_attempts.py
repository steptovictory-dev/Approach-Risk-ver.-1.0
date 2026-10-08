"""Диагностика нескольких попыток; число попыток не устанавливает назначение полета."""
from collections import Counter
from pathlib import Path
import hashlib
import json
import runpy

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
INPUT = ROOT / 'europe/data/working/03_c40_diagnostic_flags.parquet'
SOURCE = ROOT / 'europe/data/source/osn23_landings_merged.parquet.gzip'
REPORT = ROOT / 'europe/reports/preparation'
REVIEW = ROOT / 'europe/data/review'


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    previous = json.loads((REPORT / '07_c40_flags_applied.json').read_text(encoding='utf8'))
    before, source_before = digest(INPUT), digest(SOURCE)
    assert before == previous['output_sha256'] and source_before == previous['source_sha256']
    timestamp_lists = runpy.run_path(str(ROOT / 'europe/scripts/profile_data.py'))['timestamp_lists']
    reader = pq.ParquetFile(INPUT)
    counts, events, structural = Counter(), Counter(), Counter()
    pieces = []
    rows = 0
    for batch in reader.iter_batches(batch_size=100_000):
        table = pa.Table.from_batches([batch])
        n = table['n_attempts'].combine_chunks().to_numpy(zero_copy_only=False)
        ga = table['GoA'].combine_chunks().to_numpy(zero_copy_only=False)
        assert table['n_attempts'].null_count == table['GoA'].null_count == 0
        lengths = pc.list_value_length(table['attempt_times']).to_numpy()
        ils_lengths = pc.list_value_length(table['ILS']).to_numpy()
        flat, parents, first, last, reverse, repeated, nulls = timestamp_lists(batch)
        landing = table['landing_time'].combine_chunks().to_numpy(zero_copy_only=False).astype('datetime64[ns]')
        flags = {
            'attempt_count_nonpositive': n < 1,
            'attempt_length_mismatch': lengths != n,
            'ILS_length_mismatch': ils_lengths != n,
            'GoA_differs_from_more_than_one_attempt': ga != (n > 1),
            'null_attempt_element': nulls,
            'descending_attempt_times': reverse,
            'repeated_adjacent_attempt_time': repeated,
            'last_attempt_after_landing': ~np.isnat(last) & (last > landing),
        }
        for name, mask in flags.items():
            structural[name] += int(mask.sum())
        for value in np.unique(n):
            mask = n == value
            counts[int(value)] += int(mask.sum())
            events[int(value)] += int(ga[mask].sum())
        selected = table.filter(pa.array(n > 1))
        if len(selected):
            positions = rows + np.flatnonzero(n > 1)
            pieces.append(selected.append_column('source_row_position', pa.array(positions)))
        rows += len(table)
        if rows % 1_000_000 == 0:
            print('Проверено строк:', rows, flush=True)
    raw = pa.concat_tables(pieces).replace_schema_metadata(None)
    frame = raw.to_pandas(ignore_metadata=True)
    details, intervals = [], []
    for record in frame.itertuples(index=False):
        times = pd.to_datetime(record.attempt_times, utc=True)
        runways = list(record.ILS)
        # DatetimeIndex может хранить микросекунды; явно приводим к ns перед арифметикой.
        gaps = np.diff(times.as_unit('ns').asi8) / (60 * 10**9)
        span = (times[-1] - times[0]).total_seconds() / 60
        assert np.isclose(gaps.sum(), span, rtol=1e-12, atol=1e-9)
        switches = sum(a != b for a, b in zip(runways[:-1], runways[1:]))
        details.append({
            'source_row_position': record.source_row_position, 'flight_id': record.flight_id,
            'icao24': record.icao24, 'callsign': record.callsign, 'registration': record.registration,
            'airport': record.airport, 'landing_time': record.landing_time,
            'year': record.landing_time.year, 'n_attempts': record.n_attempts, 'GoA': record.GoA,
            'typecode': record.typecode, 'icaoaircrafttype': record.icaoaircrafttype,
            'market_segment': record.market_segment,
            'first_attempt_time': times[0], 'last_attempt_time': times[-1],
            'first_to_last_attempt_minutes': span,
            'first_attempt_to_landing_minutes': (record.landing_time - times[0]).total_seconds() / 60,
            'last_attempt_to_landing_seconds': (record.landing_time - times[-1]).total_seconds(),
            'min_attempt_gap_minutes': float(gaps.min()), 'median_attempt_gap_minutes': float(np.median(gaps)),
            'max_attempt_gap_minutes': float(gaps.max()),
            'unique_runway_designations': len(set(runways)), 'runway_switches': switches,
            'c40_time_missing': record.c40_time_missing,
            'c40_after_first_attempt': record.c40_after_first_attempt,
            'c40_after_landing': record.c40_after_landing,
        })
        for index, gap in enumerate(gaps):
            intervals.append({'source_row_position': record.source_row_position, 'flight_id': record.flight_id,
                              'icao24': record.icao24, 'airport': record.airport,
                              'n_attempts': record.n_attempts, 'attempt_number_from': index + 1,
                              'attempt_number_to': index + 2, 'time_from': times[index], 'time_to': times[index+1],
                              'runway_from': runways[index], 'runway_to': runways[index+1],
                              'gap_minutes': float(gap), 'same_runway': runways[index] == runways[index+1]})
    details = pd.DataFrame(details)
    intervals = pd.DataFrame(intervals)
    high = details.loc[details['n_attempts'] > 2].copy()
    high_intervals = intervals.loc[intervals['n_attempts'] > 2].copy()
    by_count = details.groupby('n_attempts').agg(
        rows=('flight_id','size'), aircraft_addresses=('icao24','nunique'), airports=('airport','nunique'),
        span_min_minutes=('first_to_last_attempt_minutes','min'),
        span_median_minutes=('first_to_last_attempt_minutes','median'),
        span_max_minutes=('first_to_last_attempt_minutes','max'),
        rows_same_runway=('unique_runway_designations',lambda s: int((s==1).sum())),
        c40_missing_rows=('c40_time_missing','sum')).reset_index()
    airport = high.groupby('airport').agg(rows=('flight_id','size'), aircraft_addresses=('icao24','nunique'),
                                         max_attempts=('n_attempts','max'),
                                         median_span_minutes=('first_to_last_attempt_minutes','median')).reset_index()
    aircraft = high.groupby(['icao24','registration','typecode','icaoaircrafttype'],dropna=False).agg(
        rows=('flight_id','size'), max_attempts=('n_attempts','max'),
        first_landing=('landing_time','min'),last_landing=('landing_time','max')).reset_index()
    distribution = pd.DataFrame([{'n_attempts': n, 'rows': counts[n], 'go_arounds': events[n]} for n in sorted(counts)])
    baseline = pd.read_csv(ROOT / 'europe/reports/attempts_summary.csv')
    pd.testing.assert_frame_equal(distribution,baseline[['n_attempts','rows','go_arounds']],check_dtype=False)
    assert rows == previous['rows_after'] and len(frame) == sum(events.values()) == previous['go_arounds_after']
    assert len(high) == 1052 and high['GoA'].all()
    assert len(intervals) == int((details['n_attempts']-1).sum())
    assert not any(structural.values()), 'Обнаружено новое структурное противоречие; требуется отдельный разбор.'
    assert digest(INPUT) == before and digest(SOURCE) == source_before
    summary = {
        'input': str(INPUT.relative_to(ROOT)), 'input_sha256': before, 'source_sha256': source_before,
        'input_unchanged': True, 'source_unchanged': True, 'rows_checked': rows,
        'rows_changed': 0, 'rows_deleted': 0, 'diagnostic_definition': 'n_attempts > 2; not an aviation limit',
        'high_attempt_rows': len(high), 'two_attempt_rows': int((details['n_attempts']==2).sum()),
        'go_arounds': len(frame), 'high_attempt_percent_of_go_around_rows': 100*len(high)/len(frame),
        'max_attempts': int(high['n_attempts'].max()), 'high_attempt_airports': int(high['airport'].nunique()),
        'high_attempt_aircraft_addresses': int(high['icao24'].nunique()), 'structural_flags': dict(structural),
        'high_attempt_same_runway_rows': int((high['unique_runway_designations']==1).sum()),
        'high_attempt_runway_change_rows': int((high['unique_runway_designations']>1).sum()),
        'high_attempt_c40_missing_rows': int(high['c40_time_missing'].sum()),
        'high_attempt_c40_after_first_rows': int(high['c40_after_first_attempt'].sum()),
        'high_attempt_c40_after_landing_rows': int(high['c40_after_landing'].sum()),
        'high_attempt_span_minutes': {'min':float(high['first_to_last_attempt_minutes'].min()),
                                     'median':float(high['first_to_last_attempt_minutes'].median()),
                                     'max':float(high['first_to_last_attempt_minutes'].max())},
        'high_attempt_gap_minutes': {'min':float(high_intervals['gap_minutes'].min()),
                                    'median':float(high_intervals['gap_minutes'].median()),
                                    'max':float(high_intervals['gap_minutes'].max())},
        'gap_quantile_unit': 'intervals, not flights; flights with many attempts contribute more intervals',
        'two_attempt_gap_minutes_median': float(intervals.loc[intervals.n_attempts==2,'gap_minutes'].median()),
        'training_flight_identity': 'not established from metadata',
        'n_attempts_minus_one_is_confirmed_go_around_count': False,
        'flight_duration_interpretation': 'first-to-last attempt window, not total flight duration',
    }
    details.to_csv(REPORT/'09_multiple_attempt_records.csv',index=False)
    high.to_csv(REPORT/'09_high_attempt_records.csv',index=False)
    high_intervals.to_csv(REPORT/'09_high_attempt_intervals.csv',index=False)
    by_count.to_csv(REPORT/'09_attempt_count_comparison.csv',index=False)
    airport.to_csv(REPORT/'09_high_attempt_airports.csv',index=False)
    aircraft.to_csv(REPORT/'09_high_attempt_aircraft.csv',index=False)
    high.groupby(['airport','year']).size().rename('rows').reset_index().to_csv(REPORT/'09_high_attempt_airport_year.csv',index=False)
    frame.loc[frame.n_attempts>2].groupby('market_segment',dropna=False).size().rename('rows').reset_index().to_csv(REPORT/'09_high_attempt_market_segments.csv',index=False)
    selected = raw.filter(pc.greater(raw['n_attempts'],2))
    pq.write_table(selected, REVIEW/'09_high_attempt_records.parquet',compression='gzip')
    example_positions = high.nlargest(3,'n_attempts')['source_row_position'].tolist()
    example_positions += high.nsmallest(3,'min_attempt_gap_minutes')['source_row_position'].tolist()
    high_intervals.loc[high_intervals.source_row_position.isin(example_positions)].to_csv(REPORT/'09_high_attempt_examples.csv',index=False)
    (REPORT/'09_attempts_diagnostics.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2,allow_nan=False)+'\n',encoding='utf8')
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)
    return by_count, airport, high, high_intervals


if __name__ == '__main__':
    main()
