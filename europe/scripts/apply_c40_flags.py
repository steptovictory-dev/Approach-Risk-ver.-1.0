"""Добавляем согласованные диагностические флаги C40, сохраняя все прежние значения."""
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
REPORT = ROOT / 'europe/reports/preparation'
WORK = ROOT / 'europe/data/working'
INPUT = WORK / '02_aircraft_reference_enriched.parquet'
OUTPUT = WORK / '03_c40_diagnostic_flags.parquet'
TEMPORARY = OUTPUT.with_suffix('.parquet.part')
SOURCE = ROOT / 'europe/data/source/osn23_landings_merged.parquet.gzip'
ADDED = ['c40_time_missing', 'c40_after_first_attempt', 'c40_after_landing']
BATCH_SIZE = 100_000


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    expected = json.loads((REPORT / '06_c40_diagnostics.json').read_text(encoding='utf8'))
    input_hash, source_hash = digest(INPUT), digest(SOURCE)
    assert input_hash == expected['input_sha256'], 'Рабочая копия отличается от проверенной в шаге 7.'
    assert source_hash == expected['source_sha256'], 'Исходный файл изменился.'
    reader = pq.ParquetFile(INPUT)
    schema = reader.schema_arrow
    assert not set(ADDED).intersection(schema.names)
    fields = [pa.field(name, pa.bool_(), nullable=False) for name in ADDED]
    for field in fields:
        schema = schema.append(field)
    written = events_before = 0
    counts, events, overlaps = Counter(), Counter(), Counter()
    with pq.ParquetWriter(TEMPORARY, schema, compression='gzip', compression_level=1) as writer:
        for batch in reader.iter_batches(batch_size=BATCH_SIZE):
            table = pa.Table.from_batches([batch])
            attempts = table['attempt_times']
            lengths = pc.list_value_length(attempts)
            assert lengths.null_count == 0 and pc.min(lengths).as_py() >= 1
            assert pc.list_flatten(attempts).null_count == 0
            crossing, landing = table['C40_CROSS_TIME'], table['landing_time']
            first = pc.cast(pc.list_element(attempts, 0), crossing.type)
            masks = [pc.is_null(crossing),
                     pc.fill_null(pc.greater(crossing, first), False),
                     pc.fill_null(pc.greater(crossing, landing), False)]
            assert landing.null_count == 0
            goa = table['GoA']
            assert goa.null_count == 0
            events_before += int(pc.sum(pc.cast(goa, pa.int64())).as_py())
            for field, mask in zip(fields, masks, strict=True):
                assert mask.null_count == 0
                table = table.append_column(field, mask)
                counts[field.name] += int(pc.sum(pc.cast(mask, pa.int64())).as_py())
                events[field.name] += int(pc.sum(pc.cast(pc.and_(mask, goa), pa.int64())).as_py())
            combination = (pc.cast(masks[0], pa.int8()).to_numpy() +
                           2 * pc.cast(masks[1], pa.int8()).to_numpy() +
                           4 * pc.cast(masks[2], pa.int8()).to_numpy())
            for value in np.unique(combination):
                selection = combination == value
                overlaps[(int(value), 'rows')] += int(selection.sum())
                overlaps[(int(value), 'go_arounds')] += int(goa.to_numpy()[selection].sum())
            writer.write_table(table)
            written += len(table)
            if written % 1_000_000 == 0:
                print('Добавлены флаги, строк:', written, flush=True)

    # Второй полный проход: сравнение всех старых значений и независимый расчет флагов через NumPy.
    attempt_summary = runpy.run_path(str(ROOT / 'europe/scripts/diagnose_c40.py'))['attempt_summary']
    candidate = pq.ParquetFile(TEMPORARY)
    assert candidate.schema_arrow.equals(schema, check_metadata=True)
    checked = events_after = 0
    for original_batch, result_batch in zip(reader.iter_batches(batch_size=BATCH_SIZE),
                                             candidate.iter_batches(batch_size=BATCH_SIZE), strict=True):
        original = pa.Table.from_batches([original_batch])
        result = pa.Table.from_batches([result_batch])
        assert result.select(original.column_names).equals(original, check_metadata=True)
        first, _, earliest = attempt_summary(original_batch)
        assert not np.isnat(first).any() and np.array_equal(first, earliest)
        crossing = original['C40_CROSS_TIME'].combine_chunks().to_numpy(zero_copy_only=False).astype('datetime64[ns]')
        landing = original['landing_time'].combine_chunks().to_numpy(zero_copy_only=False).astype('datetime64[ns]')
        expected_masks = [np.isnat(crossing),
                          ~np.isnat(crossing) & (crossing > first),
                          ~np.isnat(crossing) & (crossing > landing)]
        for name, mask in zip(ADDED, expected_masks, strict=True):
            assert result[name].null_count == 0
            assert np.array_equal(result[name].combine_chunks().to_numpy(zero_copy_only=False), mask)
        checked += len(original)
        events_after += int(pc.sum(pc.cast(result['GoA'], pa.int64())).as_py())
    assert checked == written == expected['rows_checked'] == candidate.metadata.num_rows
    assert events_before == events_after == expected['go_arounds']
    expected_flags = {record['flag']: record for record in expected['flags']}
    missing = next(row for row in expected['categories'] if row['category'] == 'missing_c40')
    assert counts[ADDED[0]] == missing['rows'] and events[ADDED[0]] == missing['go_arounds']
    for name, reference_name in zip(ADDED[1:], ['after_first_attempt', 'after_landing'], strict=True):
        reference = expected_flags[reference_name]
        assert counts[name] == reference['rows'] and events[name] == reference['go_arounds']
    assert set(key for key, metric in overlaps).issubset({0, 1, 2, 6})
    assert digest(INPUT) == input_hash and digest(SOURCE) == source_hash
    candidate.close()
    reader.close()
    output_hash = digest(TEMPORARY)
    TEMPORARY.replace(OUTPUT)

    changes = pd.DataFrame([{'field': name, 'true_rows': counts[name],
                             'false_rows': written - counts[name], 'null_rows': 0,
                             'go_arounds_with_true': events[name]} for name in ADDED])
    combinations = pd.DataFrame([{
        ADDED[0]: bool(key & 1), ADDED[1]: bool(key & 2), ADDED[2]: bool(key & 4),
        'rows': overlaps[(key, 'rows')], 'go_arounds': overlaps[(key, 'go_arounds')],
    } for key in sorted({key for key, metric in overlaps})])
    changes.to_csv(REPORT / '07_c40_flags_applied.csv', index=False)
    combinations.to_csv(REPORT / '07_c40_flag_combinations.csv', index=False)
    journal = {
        'rule': 'approved three C40 diagnostic flags; preserve every existing value and row',
        'input': str(INPUT.relative_to(ROOT)), 'input_sha256': input_hash,
        'output': str(OUTPUT.relative_to(ROOT)), 'output_sha256': output_hash,
        'source_sha256': source_hash, 'input_unchanged': True, 'source_unchanged': True,
        'existing_columns_unchanged': True, 'independent_per_row_flag_validation': True,
        'first_attempt_equals_earliest_verified': True,
        'rows_before': written, 'rows_after': written, 'rows_deleted': 0,
        'go_arounds_before': events_before, 'go_arounds_after': events_after,
        'columns_before': len(schema) - 3, 'columns_after': len(schema), 'added_fields': ADDED,
        'flag_counts': changes.to_dict(orient='records'),
        'flag_combinations': combinations.to_dict(orient='records'),
        'definitions': {
            ADDED[0]: 'C40_CROSS_TIME is physical null',
            ADDED[1]: 'C40_CROSS_TIME strictly exceeds first attempt_times element; null comparison gives False',
            ADDED[2]: 'C40_CROSS_TIME strictly exceeds landing_time; null comparison gives False',
        },
        'model_use': 'audit only; temporal flags use post-observation information, not early prediction inputs',
        'false_does_not_establish_valid_c40': True,
        'source_review': 'europe/reports/preparation/06_c40_review.md',
    }
    (REPORT / '07_c40_flags_applied.json').write_text(
        json.dumps(journal, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf8')
    print(changes.to_string(index=False), flush=True)
    print('Сохранены все записи и существующие поля:', written, flush=True)
    return changes, combinations


if __name__ == '__main__':
    main()
