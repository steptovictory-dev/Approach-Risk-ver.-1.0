"""Исключение обеих строк 147 пар по согласованному решению пользователя."""
import json
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from audit_feature_availability import ROOT, INPUT, REPORT, digest

OUTPUT=ROOT/'europe/data/working/04_c40_ambiguous_pairs_excluded.parquet'


def main():
    before=digest(INPUT)
    journal=json.loads((REPORT/'04_05_pair_audit.json').read_text(encoding='utf8'))
    assert before==journal['input_sha256']
    pairs=pd.read_csv(REPORT/'04_05_pair_review.csv')
    positions=np.sort(np.r_[pairs.early_position.to_numpy(),pairs.late_position.to_numpy()])
    assert len(pairs)==147 and len(positions)==len(set(positions))==294
    excluded=pq.read_table(ROOT/'europe/data/review/04_05_c40_pair_full_records.parquet')
    assert excluded.num_rows==294 and set(excluded['source_row_position'].to_pylist())==set(positions)
    reader=pq.ParquetFile(INPUT)
    if OUTPUT.exists():
        raise FileExistsError('Новая копия уже существует; автоматическая перезапись не выполняется')
    temporary=OUTPUT.with_suffix('.parquet.part')
    offset=written=removed=removed_positive=retained_positive=0
    # Пишем отдельную копию, сохраняя порядок и все 31 исходное поле
    with pq.ParquetWriter(temporary,reader.schema_arrow,compression='gzip',compression_level=1) as writer:
        for batch in reader.iter_batches(batch_size=100_000):
            table=pa.Table.from_batches([batch])
            index=np.arange(offset,offset+len(table))
            mask=np.isin(index,positions)
            removed_table=table.filter(pa.array(mask))
            if len(removed_table):
                expected=excluded.filter(pa.array(np.isin(excluded['source_row_position'].to_numpy(),index))).drop(['source_row_position'])
                assert removed_table.equals(expected)
            kept=table.filter(pa.array(~mask))
            writer.write_table(kept)
            removed+=len(removed_table)
            removed_positive+=int(removed_table['GoA'].to_numpy(zero_copy_only=False).sum())
            retained_positive+=int(kept['GoA'].to_numpy(zero_copy_only=False).sum())
            written+=len(kept)
            offset+=len(table)
    assert offset==7984254 and written==7983960 and removed==294
    assert removed_positive==3 and retained_positive==26250
    # Независимо сверяем поток записанного файла с каждой сохраненной исходной строкой
    def expected_blocks():
        position=0
        for batch in reader.iter_batches(batch_size=100_000):
            table=pa.Table.from_batches([batch])
            mask=~np.isin(np.arange(position,position+len(table)),positions)
            position+=len(table)
            selected=table.filter(pa.array(mask))
            if len(selected): yield selected
    expected=iter(expected_blocks())
    current=next(expected,None)
    source_offset=checked=0
    candidate=pq.ParquetFile(temporary)
    assert candidate.schema_arrow.equals(reader.schema_arrow)
    for batch in candidate.iter_batches(batch_size=100_000):
        actual=pa.Table.from_batches([batch])
        cursor=0
        while cursor<len(actual):
            assert current is not None
            size=min(len(current)-source_offset,len(actual)-cursor)
            assert actual.slice(cursor,size).equals(current.slice(source_offset,size))
            cursor+=size
            source_offset+=size
            checked+=size
            if source_offset==len(current):
                current=next(expected,None)
                source_offset=0
    assert current is None and checked==written
    candidate.close()
    assert digest(INPUT)==before
    temporary.replace(OUTPUT)
    result=dict(rule='exclude both rows of all 147 reviewed ambiguous C40 pairs',
        decision_date='2026-10-08',authorized_by='user',input=str(INPUT.relative_to(ROOT)),input_sha256=before,
        input_unchanged=True,output=str(OUTPUT.relative_to(ROOT)),output_sha256=digest(OUTPUT),
        rows_before=offset,rows_after=written,rows_excluded=removed,
        go_arounds_before=retained_positive+removed_positive,go_arounds_after=retained_positive,
        go_arounds_excluded=removed_positive,physical_columns=31,values_corrected=0,
        preserved_rows_exact_comparison_passed=True,
        excluded_records_archive='europe/data/review/04_05_c40_pair_full_records.parquet',
        reason='research exclusion of unresolved pairs; does not establish all rows as incorrect',
        missing_c40_filter_applied=False,all_other_flights_retained=True)
    (REPORT/'04_08_pairs_excluded.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps(result,ensure_ascii=False,indent=2))
    return result
