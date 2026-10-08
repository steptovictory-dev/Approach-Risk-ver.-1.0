from pathlib import Path
import json
import hashlib
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / 'europe/reports/preparation'
SOURCE = ROOT / 'europe/data/working/02_aircraft_reference_enriched.parquet'
CODES = ['82R', 'G550', 'G-V', 'GL6T', 'X']


def digest():
    with SOURCE.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    before = digest()
    expected = json.loads((REPORT / '04_aircraft_reference_applied.json').read_text(encoding='utf8'))
    assert before == expected['output_sha256']
    columns = ['icao24', 'registration', 'typecode', 'icaoaircrafttype', 'landing_time', 'airport', 'GoA']
    reader = pq.ParquetFile(SOURCE)
    parts = []
    offset = 0
    for batch in reader.iter_batches(batch_size=100_000, columns=columns):
        frame = batch.to_pandas(ignore_metadata=True)
        mask = frame['icaoaircrafttype'].isin(CODES)
        selected = frame.loc[mask].copy()
        selected['source_row_position'] = offset + selected.index
        parts.append(selected)
        offset += len(frame)
    flagged = pd.concat(parts, ignore_index=True)
    assert len(flagged) == 519 and flagged['GoA'].sum() == 2
    addresses = set(flagged['icao24'])
    related = []
    for batch in reader.iter_batches(batch_size=100_000, columns=columns):
        frame = batch.to_pandas(ignore_metadata=True)
        related.append(frame.loc[frame['icao24'].isin(addresses)].copy())
    related = pd.concat(related, ignore_index=True)
    grouped = related.groupby(['icao24', 'registration', 'typecode', 'icaoaircrafttype'], dropna=False).agg(
        rows=('airport', 'size'), go_arounds=('GoA', 'sum'),
        first_landing=('landing_time', 'min'), last_landing=('landing_time', 'max'),
        airports=('airport', lambda s: ', '.join(sorted(set(s))))).reset_index()
    summary = flagged.groupby('icaoaircrafttype').agg(
        rows=('airport', 'size'), go_arounds=('GoA', 'sum'), aircraft_addresses=('icao24', 'nunique'),
        first_landing=('landing_time', 'min'), last_landing=('landing_time', 'max')).reset_index()
    flagged.to_csv(REPORT / '05_unresolved_aircraft_records.csv', index=False)
    grouped.to_csv(REPORT / '05_unresolved_aircraft_history.csv', index=False)
    summary.to_csv(REPORT / '05_unresolved_aircraft_summary.csv', index=False)
    assert digest() == before and offset == expected['rows_after']
    (REPORT / '05_unresolved_aircraft_diagnostics.json').write_text(json.dumps({
        'input_sha256': before, 'input_unchanged': True, 'rows_changed': 0,
        'flagged_rows': len(flagged), 'flagged_go_arounds': int(flagged['GoA'].sum()),
        'aircraft_addresses': len(addresses), 'related_rows': len(related),
        'identity_caution': 'icao24 does not independently establish historical aircraft identity',
    }, indent=2) + '\n', encoding='utf8')
    print(summary.to_string(index=False), flush=True)
    return summary, grouped


if __name__ == '__main__':
    main()
