"""Apply the separately approved reference enrichment, preserving existing columns."""
from pathlib import Path
from collections import Counter
import hashlib
import json
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / 'europe/reports/preparation'
WORK = ROOT / 'europe/data/working'
INPUT = WORK / '01_empty_strings_normalized.parquet'
OUTPUT = WORK / '02_aircraft_reference_enriched.parquet'
TEMPORARY = OUTPUT.with_suffix('.parquet.part')
SOURCE = ROOT / 'europe/data/source/osn23_landings_merged.parquet.gzip'
REFERENCE = ROOT / 'docs/sources/files/aircraft_code_review/icao_aircraft_types.json'
URL = 'https://doc8643.icao.int/External/AircraftTypes'
CHECKED_DATE = '2026-10-08'
ADDED = ['aircraft_description_reference', 'aircraft_reference_status',
         'aircraft_reference_source', 'aircraft_reference_checked_date']


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    previous = json.loads((REPORT / '01_empty_strings_normalized.json').read_text(encoding='utf8'))
    input_hash = digest(INPUT)
    assert input_hash == previous['output_sha256']
    assert digest(SOURCE) == previous['source_sha256']
    manifest = json.loads((REPORT / 'aircraft_reference_sources.json').read_text(encoding='utf8'))
    snapshot = next(s for s in manifest['sources'] if s['name'] == REFERENCE.name)
    reference_hash = digest(REFERENCE)
    assert reference_hash == snapshot['sha256'] and snapshot['url'] == URL
    official = pd.DataFrame(json.loads(REFERENCE.read_text(encoding='utf8')))
    proposals = pd.read_csv(REPORT / '03_aircraft_code_proposals.csv', keep_default_na=False)
    approved = proposals.loc[proposals['status'].eq('direct_unique_reference_match')]
    assert len(approved) == 18 and approved['rows'].sum() == 5888
    mapping = dict(zip(approved['source_icaoaircrafttype'], approved['reference_description']))
    for code, description in mapping.items():
        matches = official.loc[official['Designator'].eq(code.upper()), 'Description']
        assert set(matches.dropna()) == {description}
    unresolved = set(proposals.loc[proposals['status'].eq('unresolved'), 'source_icaoaircrafttype'])
    reader = pq.ParquetFile(INPUT)
    schema = reader.schema_arrow
    assert not set(ADDED).intersection(schema.names)
    for name in ADDED:
        schema = schema.append(pa.field(name, pa.string()))
    counts = Counter()
    events = Counter()
    statuses = Counter()
    written = 0
    with pq.ParquetWriter(TEMPORARY, schema, compression='gzip', compression_level=1) as writer:
        for batch in reader.iter_batches(batch_size=100_000):
            table = pa.Table.from_batches([batch])
            codes = table['icaoaircrafttype'].to_pandas()
            descriptions = codes.map(mapping)
            matched = descriptions.notna()
            status = pd.Series('not_reviewed', index=codes.index)
            status.loc[codes.isna()] = 'source_description_missing'
            status.loc[codes.isin(unresolved)] = 'unresolved'
            status.loc[matched] = 'direct_unique_reference_match'
            provenance = pd.Series(None, index=codes.index, dtype='object')
            provenance.loc[matched] = URL
            dates = pd.Series(None, index=codes.index, dtype='object')
            dates.loc[matched] = CHECKED_DATE
            for name, values in zip(ADDED, [descriptions, status, provenance, dates], strict=True):
                table = table.append_column(name, pa.array(values, type=pa.string(), from_pandas=True))
            counts.update(codes.loc[matched].value_counts().to_dict())
            flags = table['GoA'].to_pandas()
            events.update(codes.loc[matched & flags].value_counts().to_dict())
            statuses.update(status.value_counts().to_dict())
            writer.write_table(table)
            written += len(table)

    # Compare every existing value; verify derived values and provenance independently.
    result_reader = pq.ParquetFile(TEMPORARY)
    checked = total_events = 0
    for original_batch, result_batch in zip(reader.iter_batches(batch_size=100_000),
                                             result_reader.iter_batches(batch_size=100_000), strict=True):
        original = pa.Table.from_batches([original_batch])
        result = pa.Table.from_batches([result_batch])
        assert result.select(original.column_names).equals(original)
        codes = original['icaoaircrafttype'].to_pandas()
        expected = pa.array(codes.map(mapping), type=pa.string(), from_pandas=True)
        assert result[ADDED[0]].combine_chunks().equals(expected)
        available = pc.is_valid(result[ADDED[0]])
        for name, value in [(ADDED[2], URL), (ADDED[3], CHECKED_DATE)]:
            expected_values = pc.if_else(available, pa.scalar(value), pa.scalar(None, pa.string()))
            assert result[name].equals(expected_values)
        checked += len(original)
        total_events += pc.sum(pc.cast(original['GoA'], pa.int64())).as_py()
    assert checked == written == previous['rows_after'] == result_reader.metadata.num_rows
    assert total_events == 26253
    assert sum(counts.values()) == 5888 and sum(events.values()) == 27
    assert statuses['unresolved'] == 519
    for row in approved.itertuples(index=False):
        assert counts[row.source_icaoaircrafttype] == row.rows
        assert events[row.source_icaoaircrafttype] == row.go_arounds
    assert digest(INPUT) == input_hash
    assert digest(SOURCE) == previous['source_sha256']
    result_reader.close()
    reader.close()
    TEMPORARY.replace(OUTPUT)
    changes = [{'source_icaoaircrafttype': code, 'reference_description': mapping[code],
                'rows_enriched': count, 'go_arounds': events[code]}
               for code, count in sorted(counts.items())]
    pd.DataFrame(changes).to_csv(REPORT / '04_aircraft_reference_applied.csv', index=False)
    journal = {
        'rule': 'approved additional reference description for 18 exact source values',
        'input': str(INPUT.relative_to(ROOT)), 'input_sha256': input_hash,
        'output': str(OUTPUT.relative_to(ROOT)), 'output_sha256': digest(OUTPUT),
        'source_sha256': previous['source_sha256'], 'source_unchanged': True,
        'input_unchanged': True, 'existing_columns_unchanged': True,
        'rows_before': written, 'rows_after': written, 'rows_deleted': 0,
        'go_arounds_before': total_events, 'go_arounds_after': total_events,
        'added_fields': ADDED, 'rows_enriched': sum(counts.values()),
        'status_counts': dict(statuses), 'changes': changes,
        'official_search_url': 'https://www.icao.int/operational-safety/doc-8643-aircraft-type-designators/search',
        'reference_url': URL, 'reference_method': 'POST', 'reference_checked_date': CHECKED_DATE,
        'reference_sha256': reference_hash, 'reference_rows': len(official),
        'historical_validity_2019_2023': 'not_established',
    }
    (REPORT / '04_aircraft_reference_applied.json').write_text(
        json.dumps(journal, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
    print('Rows preserved:', written, 'Rows enriched:', sum(counts.values()), flush=True)
    return pd.DataFrame(changes)


if __name__ == '__main__':
    main()
