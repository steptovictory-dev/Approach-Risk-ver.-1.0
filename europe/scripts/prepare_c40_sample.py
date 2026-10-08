"""Предварительный расчет и применение отдельного правила доступности C40."""
from contextlib import ExitStack
from pathlib import Path
import argparse
import json
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from audit_feature_availability import ROOT, REPORT, digest
from audit_c40_selection import INPUT

DESTINATION = ROOT / 'europe/data/analytical'
INCLUDED = DESTINATION / '04_c40_available.parquet'
OUTSIDE = DESTINATION / '04_c40_outside_experiment.parquet'


def classify(table):
    # Взаимоисключающие причины помогают сохранить точные знаменатели
    valid = np.ones(len(table), dtype=bool)
    for field in ['C40_CROSS_TIME', 'C40_BEARING', 'C40_CROSS_LAT', 'C40_CROSS_LON']:
        valid &= pc.is_valid(table[field]).to_numpy(zero_copy_only=False)
    finite = np.ones(len(table), dtype=bool)
    for field in ['C40_BEARING', 'C40_CROSS_LAT', 'C40_CROSS_LON']:
        finite &= np.isfinite(table[field].to_numpy(zero_copy_only=False))
    c40 = table['C40_CROSS_TIME'].to_numpy(zero_copy_only=False).astype('datetime64[ns]')
    first = pc.list_element(table['attempt_times'], 0).to_numpy(zero_copy_only=False).astype('datetime64[ns]')
    reason = np.full(len(table), 'included', dtype=object)
    reason[~valid] = 'missing_c40_field'
    reason[valid & ~finite] = 'nonfinite_c40_value'
    reason[valid & finite & np.isnat(first)] = 'missing_first_attempt_time'
    reason[valid & finite & ~np.isnat(first) & (c40 >= first)] = 'c40_not_before_first_attempt'
    return reason


def main(apply=False):
    # Без apply создаем только расчет влияния; база остается неизменной
    previous = json.loads((REPORT / '04_09_selection_audit.json').read_text(encoding='utf8'))
    before = digest(INPUT)
    assert before == previous['input_sha256']
    reader = pq.ParquetFile(INPUT)
    pieces = []
    included_rows = outside_rows = 0
    with ExitStack() as stack:
        if apply:
            DESTINATION.mkdir(parents=True, exist_ok=True)
            if INCLUDED.exists() or OUTSIDE.exists():
                raise FileExistsError('Аналитические копии уже существуют; перезапись не выполняется')
            inside_temp = INCLUDED.with_suffix('.parquet.part')
            outside_temp = OUTSIDE.with_suffix('.parquet.part')
            writer_in = stack.enter_context(pq.ParquetWriter(inside_temp, reader.schema_arrow, compression='gzip', compression_level=1))
            writer_out = stack.enter_context(pq.ParquetWriter(outside_temp, reader.schema_arrow, compression='gzip', compression_level=1))
        for batch in reader.iter_batches(batch_size=100_000):
            table = pa.Table.from_batches([batch])
            reason = classify(table)
            mask = reason == 'included'
            frame = pd.DataFrame({'airport': table['airport'].to_pylist(), 'reason': reason,
                                  'GoA': table['GoA'].to_numpy(zero_copy_only=False)})
            pieces.append(frame.groupby(['airport','reason']).agg(rows=('GoA','size'),go_arounds=('GoA','sum')).reset_index())
            included_rows += int(mask.sum())
            outside_rows += int((~mask).sum())
            if apply:
                writer_in.write_table(table.filter(pa.array(mask)))
                writer_out.write_table(table.filter(pa.array(~mask)))
    impact = pd.concat(pieces).groupby(['airport','reason'])[['rows','go_arounds']].sum().reset_index()
    assert impact.rows.sum() == previous['rows_checked']
    assert impact.go_arounds.sum() == previous['go_arounds']
    assert included_rows == previous['eligible_rows'] and outside_rows == previous['excluded_rows']
    assert int(impact.loc[impact.reason=='included','go_arounds'].sum()) == previous['eligible_go_arounds']
    assert digest(INPUT) == before
    report = dict(input=str(INPUT.relative_to(ROOT)), input_sha256=before, input_unchanged=True,
                  rule='all four C40 fields present; numeric values finite; C40 strictly before first attempt',
                  applied=apply, rows_included=included_rows, rows_outside_experiment=outside_rows,
                  source_rows_deleted=0, values_changed=0, physical_columns_preserved=len(reader.schema_arrow),
                  trajectory_matching_verified=False,
                  positive_included=previous['eligible_go_arounds'], positive_outside_experiment=previous['excluded_go_arounds'],
                  exclusion_reason='Incomplete C40 data or C40 not strictly before first attempt; reconstructing availability requires a separate costly trajectory review outside the scope of this experiment',
                  decision_date='2026-10-08', authorized_by='user',
                  does_not_establish_all_excluded_records_as_incorrect=True)
    if apply:
        # Сверяем каждую записанную строку с соответствующим исходным подмножеством
        candidates = {True: pq.ParquetFile(inside_temp), False: pq.ParquetFile(outside_temp)}
        assert all(p.schema_arrow.equals(reader.schema_arrow) for p in candidates.values())
        expected = {flag: p.iter_batches(batch_size=100_000) for flag,p in candidates.items()}
        def chunks(flag):
            for batch in reader.iter_batches(batch_size=100_000):
                table = pa.Table.from_batches([batch])
                mask = classify(table) == 'included'
                selected = table.filter(pa.array(mask if flag else ~mask))
                if len(selected):
                    yield selected
        # Сравнение потоков учитывает различное разбиение Parquet на блоки
        for flag in [True, False]:
            source = iter(chunks(flag))
            current = next(source, None)
            position = 0
            checked = 0
            for actual_batch in expected[flag]:
                actual = pa.Table.from_batches([actual_batch])
                offset = 0
                while offset < len(actual):
                    assert current is not None
                    length = min(len(current)-position, len(actual)-offset)
                    assert actual.slice(offset,length).equals(current.slice(position,length))
                    checked += length
                    offset += length
                    position += length
                    if position == len(current):
                        current = next(source,None)
                        position = 0
            assert current is None
            assert checked == (included_rows if flag else outside_rows)
        assert digest(INPUT) == before
        for candidate in candidates.values():
            candidate.close()
        inside_temp.replace(INCLUDED)
        outside_temp.replace(OUTSIDE)
        report.update(included_file=str(INCLUDED.relative_to(ROOT)), outside_file=str(OUTSIDE.relative_to(ROOT)),
                      included_sha256=digest(INCLUDED), outside_sha256=digest(OUTSIDE),
                      exact_per_row_comparison_passed=True)
    REPORT.mkdir(parents=True, exist_ok=True)
    suffix = 'applied' if apply else 'preview'
    impact.to_csv(REPORT / f'04_10_c40_{suffix}_impact.csv',index=False)
    (REPORT / f'04_10_c40_{suffix}.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps(report,ensure_ascii=False,indent=2))
    return impact, report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apply',action='store_true')
    args = parser.parse_args()
    main(apply=args.apply)
