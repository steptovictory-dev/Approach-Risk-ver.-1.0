"""Полная диагностика повторяющихся flight_id без удаления или объединения записей."""
from pathlib import Path
import hashlib
import json

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
    audit = json.loads((ROOT / 'europe/reports/duplicate_check.json').read_text(encoding='utf8'))
    before, source_before = digest(INPUT), digest(SOURCE)
    assert before == previous['output_sha256'] and source_before == previous['source_sha256']
    reader = pq.ParquetFile(INPUT)
    ids = reader.read(columns=['flight_id'])['flight_id']
    assert ids.null_count == 0
    values = pc.value_counts(ids)
    repeated = pc.filter(values, pc.greater(values.field('counts'), 1))
    repeated_ids = repeated.field('values')
    del ids, values
    pieces = []
    rows = 0
    for batch in reader.iter_batches(batch_size=100_000):
        mask = pc.is_in(batch.column('flight_id'), value_set=repeated_ids)
        selected = pa.Table.from_batches([batch]).filter(mask)
        if len(selected):
            positions = rows + pc.indices_nonzero(mask).to_numpy(zero_copy_only=False)
            pieces.append(selected.append_column('source_row_position', pa.array(positions)))
        rows += len(batch)
        if rows % 1_000_000 == 0:
            print('Проверено строк:', rows, flush=True)
    raw = pa.concat_tables(pieces).replace_schema_metadata(None)
    frame = raw.to_pandas(ignore_metadata=True)
    frame['landing_date_utc'] = frame['landing_time'].dt.strftime('%Y-%m-%d')
    frame['landing_year'] = frame['landing_time'].dt.year
    frame['first_attempt_time'] = pd.to_datetime(frame['attempt_times'].map(lambda times: times[0]), utc=True)
    grouped = frame.groupby('flight_id', sort=True)
    groups = grouped.agg(
        rows=('flight_id', 'size'), go_arounds=('GoA', 'sum'),
        dates=('landing_date_utc', 'nunique'), years=('landing_year', 'nunique'),
        addresses=('icao24', lambda s: s.nunique(dropna=False)),
        airports=('airport', lambda s: s.nunique(dropna=False)),
        callsigns=('callsign', lambda s: s.nunique(dropna=False)),
        first_landing=('landing_time', 'min'), last_landing=('landing_time', 'max'),
        c40_after_landing_rows=('c40_after_landing', 'sum')).reset_index()
    groups['landing_span_days'] = (groups['last_landing'] - groups['first_landing']).dt.total_seconds() / 86400
    groups['date_alone_disambiguates_identifier'] = groups['dates'] == groups['rows']
    contexts = frame.groupby(['flight_id', 'landing_date_utc', 'icao24', 'airport'], dropna=False).size()
    same_context_ids = set(contexts[contexts > 1].index.get_level_values('flight_id'))
    groups['same_day_address_airport_repeat'] = groups['flight_id'].isin(same_context_ids)
    groups['category'] = 'different_dates_only'
    groups.loc[~groups['date_alone_disambiguates_identifier'], 'category'] = 'same_day_different_context'
    groups.loc[groups['same_day_address_airport_repeat'], 'category'] = 'same_day_same_address_airport'

    # Точное сравнение содержимого: JSON только для канонизации значений, не хеш-критерий равенства.
    existing_names = [name for name in reader.schema_arrow.names if name != '__index_level_0__']
    business_names = [name for name in pq.ParquetFile(SOURCE).schema_arrow.names if name != '__index_level_0__']
    records = raw.to_pylist()
    canonical_working = [json.dumps({name: record[name] for name in existing_names},
                                   sort_keys=True, default=str, allow_nan=False) for record in records]
    canonical_business = [json.dumps({name: record[name] for name in business_names},
                                    sort_keys=True, default=str, allow_nan=False) for record in records]
    working_duplicate_excess = len(records) - len(set(canonical_working))
    business_duplicate_excess = len(records) - len(set(canonical_business))
    core_duplicate_excess = int(frame.duplicated(['landing_time', 'icao24', 'airport']).sum())
    del records, canonical_working, canonical_business

    # Все пары внутри повторяющихся идентификаторов; максимум группы определяется по данным.
    columns = ['flight_id', 'source_row_position', 'icao24', 'airport', 'callsign',
               'landing_date_utc', 'landing_time', 'first_attempt_time', 'GoA', 'n_attempts']
    pairs = frame[columns].merge(frame[columns], on='flight_id', suffixes=('_a', '_b'))
    pairs = pairs.loc[pairs['source_row_position_a'] < pairs['source_row_position_b']].copy()
    for field in ['icao24', 'airport', 'callsign', 'landing_date_utc', 'landing_time']:
        pairs['same_' + field] = pairs[field + '_a'].eq(pairs[field + '_b'])
    pairs['landing_gap_minutes'] = (pairs['landing_time_b'] - pairs['landing_time_a']).abs().dt.total_seconds() / 60
    same_context = pairs['same_icao24'] & pairs['same_airport'] & pairs['same_landing_date_utc']
    lower = pairs[['first_attempt_time_a', 'first_attempt_time_b']].max(axis=1)
    upper = pairs[['landing_time_a', 'landing_time_b']].min(axis=1)
    pairs['attempt_to_landing_windows_overlap'] = same_context & lower.le(upper)
    near = pairs.loc[same_context].copy()
    categories = groups.groupby('category').agg(identifier_groups=('flight_id', 'size'),
                                                rows=('rows', 'sum'), go_arounds=('go_arounds', 'sum')).reset_index()
    size_distribution = groups.groupby('rows').agg(identifier_groups=('flight_id', 'size'),
                                                  go_arounds=('go_arounds', 'sum')).reset_index().rename(columns={'rows': 'rows_per_identifier'})
    airports = frame.groupby('airport').agg(rows=('flight_id', 'size'),
                                           repeated_identifiers=('flight_id', 'nunique'), go_arounds=('GoA', 'sum')).reset_index()
    excess = int((groups['rows'] - 1).sum())
    assert rows == previous['rows_after'] and len(groups) == audit['flight_id_repeated_values']
    assert excess == audit['flight_id_excess_rows']
    assert len(frame) == len(groups) + excess and int(pc.sum(repeated.field('counts')).as_py()) == len(frame)
    assert digest(INPUT) == before and digest(SOURCE) == source_before
    summary = {
        'input': str(INPUT.relative_to(ROOT)), 'input_sha256': before, 'source_sha256': source_before,
        'input_unchanged': True, 'source_unchanged': True, 'rows_checked': rows,
        'rows_changed': 0, 'rows_deleted': 0, 'repeated_identifiers': len(groups),
        'rows_with_repeated_identifier': len(frame), 'excess_identifier_rows': excess,
        'go_arounds_in_repeated_identifier_rows': int(frame['GoA'].sum()),
        'max_rows_per_identifier': int(groups['rows'].max()),
        'exact_working_duplicate_excess_in_repeated_groups': working_duplicate_excess,
        'normalized_source_business_duplicate_excess_in_repeated_groups': business_duplicate_excess,
        'core_key_duplicate_excess_in_repeated_groups': core_duplicate_excess,
        'groups_with_multiple_aircraft_addresses': int((groups['addresses'] > 1).sum()),
        'groups_with_multiple_airports': int((groups['airports'] > 1).sum()),
        'groups_with_multiple_callsigns_including_null': int((groups['callsigns'] > 1).sum()),
        'groups_with_at_least_one_go_around': int((groups['go_arounds'] > 0).sum()),
        'groups_with_mixed_go_around_labels': int(((groups['go_arounds'] > 0) & (groups['go_arounds'] < groups['rows'])).sum()),
        'groups_with_multiple_go_around_rows': int((groups['go_arounds'] > 1).sum()),
        'go_arounds_lost_if_deduplicating_flight_id_keep_first_in_file_order': int(frame.loc[frame.duplicated('flight_id', keep='first'), 'GoA'].sum()),
        'identifier_prefix_matches_callsign_rows_in_repeated_groups': int(frame['flight_id'].str.rsplit('_', n=1).str[0].eq(frame['callsign']).sum()),
        'min_landing_gap_across_all_identifier_pairs_minutes': float(pairs['landing_gap_minutes'].min()),
        'median_landing_gap_across_all_identifier_pairs_days': float(pairs['landing_gap_minutes'].median() / 1440),
        'categories': categories.to_dict(orient='records'),
        'same_day_same_address_airport_pairs': len(near),
        'overlapping_attempt_to_landing_window_pairs': int(near['attempt_to_landing_windows_overlap'].sum()),
        'same_context_min_landing_gap_minutes': float(near['landing_gap_minutes'].min()) if len(near) else None,
        'same_context_max_landing_gap_minutes': float(near['landing_gap_minutes'].max()) if len(near) else None,
        'source_algorithm_generating_ids': 'not_established; resemblance to traffic default is a hypothesis',
        'core_key': ['landing_time', 'icao24', 'airport'],
        'key_status': 'unique observed record key in audited file, not proof of physical flight identity',
        'row_position_status': 'zero-based file-local position, not a cross-version identifier',
    }
    groups.to_csv(REPORT / '08_flight_id_groups.csv', index=False)
    categories.to_csv(REPORT / '08_flight_id_categories.csv', index=False)
    size_distribution.to_csv(REPORT / '08_flight_id_group_sizes.csv', index=False)
    airports.to_csv(REPORT / '08_flight_id_airports.csv', index=False)
    near.to_csv(REPORT / '08_flight_id_same_context_pairs.csv', index=False)
    pq.write_table(raw, REVIEW / '08_repeated_flight_id_records.parquet', compression='gzip')
    # Несколько полных групп для просмотра, а не произвольная голова файла.
    sample_ids = groups.sort_values(['rows', 'flight_id'], ascending=[False, True]).head(3)['flight_id'].tolist()
    sample_ids += groups.loc[groups['same_day_address_airport_repeat'], 'flight_id'].head(3).tolist()
    examples = frame.loc[frame['flight_id'].isin(sample_ids), columns + ['c40_after_landing']].sort_values(['flight_id', 'landing_time'])
    examples.to_csv(REPORT / '08_flight_id_examples.csv', index=False)
    (REPORT / '08_flight_id_diagnostics.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf8')
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    return groups, categories, examples, near


if __name__ == '__main__':
    main()
