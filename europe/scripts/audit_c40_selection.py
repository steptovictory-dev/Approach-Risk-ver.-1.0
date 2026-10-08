"""Последствия строгого отбора по доступности C40 на актуальной копии."""
import json
from collections import Counter
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from audit_feature_availability import ROOT, REPORT, digest

INPUT = ROOT / 'europe/data/working/04_c40_ambiguous_pairs_excluded.parquet'


def main():
    # Проверяем происхождение актуального входа после исключения пар
    journal = json.loads((REPORT / '04_08_pairs_excluded.json').read_text(encoding='utf8'))
    before = digest(INPUT)
    assert before == journal['output_sha256']
    counts, positives = Counter(), Counter()
    parts = {field: [] for field in ['airport', 'year', 'market_segment']}
    columns = ['airport', 'landing_time', 'market_segment', 'GoA', 'attempt_times',
               'C40_CROSS_TIME', 'C40_BEARING', 'C40_CROSS_LAT', 'C40_CROSS_LON']
    for batch in pq.ParquetFile(INPUT).iter_batches(batch_size=100_000, columns=columns):
        table = pa.Table.from_batches([batch])
        # Вычисляем время непосредственно из исходных полей, а не из старых флагов
        attempts = table['attempt_times']
        lengths = pc.fill_null(pc.list_value_length(attempts), 0).to_numpy()
        assert (lengths > 0).all(), 'Обнаружены пустые списки попыток'
        first = pc.list_element(attempts, 0).to_numpy().astype('datetime64[ns]')
        c40 = table['C40_CROSS_TIME'].to_numpy().astype('datetime64[ns]')
        fields = columns[-4:]
        complete = np.logical_and.reduce([pc.is_valid(table[f]).to_numpy() for f in fields])
        finite = np.logical_and.reduce([np.isfinite(table[f].to_numpy()) for f in fields[1:]])
        first_valid = ~np.isnat(first)
        # Причины взаимоисключающие: учитываем первое невыполненное условие
        status = np.select([~complete, complete & ~finite,
                            complete & finite & ~first_valid,
                            complete & finite & first_valid & (c40 == first),
                            complete & finite & first_valid & (c40 > first)],
                           ['incomplete_c40', 'nonfinite_c40', 'missing_first_attempt',
                            'c40_equal_first_attempt', 'c40_after_first_attempt'], default='eligible')
        frame = table.select(['airport', 'landing_time', 'market_segment', 'GoA']).to_pandas(ignore_metadata=True)
        frame['year'] = frame.landing_time.dt.year.astype('Int64')
        frame['selection_status'] = status
        for name, group in frame.groupby('selection_status'):
            counts[name] += len(group)
            positives[name] += int(group.GoA.sum())
        for field in parts:
            parts[field].append(frame.groupby([field, 'selection_status'], dropna=False).agg(
                rows=('GoA', 'size'), go_arounds=('GoA', 'sum')).reset_index())
    # Сверяем полный объем с предыдущим шагом и неизменность файла
    rows, ga = sum(counts.values()), sum(positives.values())
    assert rows == journal['rows_after'] and ga == journal['go_arounds_after']
    assert digest(INPUT) == before
    statuses = ['eligible', 'incomplete_c40', 'nonfinite_c40', 'missing_first_attempt',
                'c40_equal_first_attempt', 'c40_after_first_attempt']
    summary = pd.DataFrame([dict(selection_status=s, rows=counts[s], go_arounds=positives[s],
        row_percent=100*counts[s]/rows, positive_percent=100*positives[s]/ga,
        ga_rate_percent=100*positives[s]/counts[s] if counts[s] else None) for s in statuses])
    summary.to_csv(REPORT / '04_09_selection_summary.csv', index=False)
    groups = {}
    for field, pieces in parts.items():
        grouped = pd.concat(pieces).groupby([field, 'selection_status'], dropna=False)[['rows', 'go_arounds']].sum().reset_index()
        assert int(grouped.rows.sum()) == rows and int(grouped.go_arounds.sum()) == ga
        total = grouped.groupby(field, dropna=False)[['rows', 'go_arounds']].sum()
        kept = grouped[grouped.selection_status == 'eligible'].set_index(field)[['rows', 'go_arounds']].reindex(total.index, fill_value=0)
        result = total.rename(columns={'rows':'rows_before', 'go_arounds':'go_arounds_before'}).join(
            kept.rename(columns={'rows':'rows_eligible', 'go_arounds':'go_arounds_eligible'}))
        result['rows_excluded'] = result.rows_before-result.rows_eligible
        result['go_arounds_excluded'] = result.go_arounds_before-result.go_arounds_eligible
        result['row_retention_percent'] = 100*result.rows_eligible/result.rows_before
        result['positive_retention_percent'] = 100*result.go_arounds_eligible/result.go_arounds_before.replace(0,np.nan)
        result['row_share_before_percent'] = 100*result.rows_before/rows
        result['row_share_eligible_percent'] = 100*result.rows_eligible/counts['eligible']
        result['share_change_pp'] = result.row_share_eligible_percent-result.row_share_before_percent
        result['ga_rate_before_percent'] = 100*result.go_arounds_before/result.rows_before
        result['ga_rate_eligible_percent'] = 100*result.go_arounds_eligible/result.rows_eligible.replace(0,np.nan)
        groups[field] = result.reset_index()
        groups[field].to_csv(REPORT / f'04_09_selection_by_{field}.csv', index=False)
        grouped.to_csv(REPORT / f'04_09_reasons_by_{field}.csv', index=False)
    audit = dict(input=str(INPUT.relative_to(ROOT)), input_sha256=before, input_unchanged=True,
        rows_checked=rows, go_arounds=ga, eligible_rows=counts['eligible'], eligible_go_arounds=positives['eligible'],
        excluded_rows=rows-counts['eligible'], excluded_go_arounds=ga-positives['eligible'],
        rule='All four C40 fields present; numeric values finite; C40 strictly before first attempt',
        reason_priority=statuses[1:], rows_changed=0, rows_deleted=0, filter_applied=False,
        trajectory_matching_verified=False, market_segment_is_verified_flight_purpose=False,
        year_basis='UTC year of final landing; retrospective audit only',
        limitation='Availability does not establish correct C40 matching, coordinate plausibility or label validity')
    (REPORT / '04_09_selection_audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    return summary, groups, audit
