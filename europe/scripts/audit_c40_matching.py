"""Полная проверка повторного присоединения C40 без исправления записей."""
import json
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from audit_feature_availability import INPUT, ROOT, REPORT, digest


def main():
    before = digest(INPUT)
    previous = json.loads((REPORT / '04_02_feature_audit.json').read_text(encoding='utf8'))
    assert before == previous['input_sha256']
    pieces = []
    offset = missing_key_rows = 0
    columns = ['icao24','airport','landing_time','C40_CROSS_TIME','attempt_times','GoA',
               'C40_CROSS_LAT','C40_CROSS_LON','C40_BEARING']
    for batch in pq.ParquetFile(INPUT).iter_batches(batch_size=100_000,columns=columns):
        table = pa.Table.from_batches([batch])
        valid = pc.is_valid(table['C40_CROSS_TIME']).to_numpy(zero_copy_only=False)
        selected = table.filter(pa.array(valid))
        frame = selected.drop(['attempt_times']).to_pandas(ignore_metadata=True)
        frame['first_attempt_time'] = pd.to_datetime(pc.list_element(selected['attempt_times'],0).to_numpy(zero_copy_only=False),utc=True)
        frame['source_row_position'] = offset + np.flatnonzero(valid)
        missing_key_rows += int(frame[['icao24','airport']].isna().any(axis=1).sum())
        pieces.append(frame)
        offset += len(table)
    assert offset == previous['rows_checked']
    frame = pd.concat(pieces,ignore_index=True)
    frame['icao24'] = frame.icao24.astype('category')
    frame['airport'] = frame.airport.astype('category')
    keys = ['icao24','airport','C40_CROSS_TIME']
    # Одно время одного борта и аэропорта не должно автоматически связывать разные посадки
    sizes = frame.groupby(keys,observed=True,dropna=False).size()
    repeated = sizes[sizes>1].rename('rows').reset_index()
    candidates = frame.merge(repeated[keys],on=keys,how='inner',validate='many_to_one')
    candidates['c40_after_first_attempt'] = candidates.C40_CROSS_TIME > candidates.first_attempt_time
    candidates['c40_after_landing'] = candidates.C40_CROSS_TIME > candidates.landing_time
    candidates['landing_utc_date'] = candidates.landing_time.dt.strftime('%Y-%m-%d')
    grouped = candidates.groupby(keys,observed=True,dropna=False).agg(
        rows=('GoA','size'),go_arounds=('GoA','sum'),
        distinct_landing_times=('landing_time','nunique'),
        landing_dates=('landing_utc_date','nunique'),
        first_landing=('landing_time','min'),last_landing=('landing_time','max'),
        c40_after_first_rows=('c40_after_first_attempt','sum'),
        c40_after_landing_rows=('c40_after_landing','sum'),
        distinct_latitudes=('C40_CROSS_LAT','nunique'),
        distinct_longitudes=('C40_CROSS_LON','nunique'),
        distinct_bearings=('C40_BEARING','nunique')).reset_index()
    grouped['landing_span_minutes'] = (grouped.last_landing-grouped.first_landing).dt.total_seconds()/60
    # Проверяем неоднозначность контекста сопоставления в один UTC-день среди строк с C40
    frame['landing_utc_date'] = frame.landing_time.dt.strftime('%Y-%m-%d')
    context = frame.groupby(['icao24','airport','landing_utc_date'],observed=True,dropna=False).agg(
        rows=('GoA','size'),distinct_c40_times=('C40_CROSS_TIME','nunique'),go_arounds=('GoA','sum')).reset_index()
    ambiguous_context = context.loc[context.rows>1]
    assert len(candidates) == int(repeated.rows.sum())
    assert candidates.source_row_position.is_unique
    assert digest(INPUT) == before
    grouped.to_csv(REPORT / '04_04_repeated_c40_groups.csv',index=False)
    candidates.sort_values(keys+['landing_time']).to_csv(REPORT / '04_04_repeated_c40_records.csv',index=False)
    ambiguous_context.to_csv(REPORT / '04_04_same_day_contexts.csv',index=False)
    review = ROOT / 'europe/data/review/04_04_repeated_c40_projection.parquet'
    # Сохраняем типизированную диагностическую проекцию, не очищенную базу
    pq.write_table(pa.Table.from_pandas(candidates,preserve_index=False),review,compression='gzip')
    summary = dict(input_sha256=before,input_unchanged=True,rows_checked=offset,
                   rows_with_c40=len(frame),rows_with_missing_context_keys=missing_key_rows,
                   repeated_c40_groups=len(grouped),rows_sharing_c40=len(candidates),
                   positive_rows_sharing_c40=int(candidates.GoA.sum()),
                   groups_with_different_landing_times=int((grouped.distinct_landing_times>1).sum()),
                   groups_spanning_multiple_landing_dates=int((grouped.landing_dates>1).sum()),
                   repeated_rows_c40_after_first=int(candidates.c40_after_first_attempt.sum()),
                   repeated_rows_c40_after_landing=int(candidates.c40_after_landing.sum()),
                   same_day_multiple_landing_contexts_with_c40=len(ambiguous_context),
                   rows_in_same_day_multiple_landing_contexts=int(ambiguous_context.rows.sum()),
                   rows_changed=0,rows_deleted=0,selection_applied=False,
                   trajectory_matching_verified=False,
                   scope='Metadata audit; no independent trajectories or historical EU-NM records',
                   context_scope='Same-day contexts counted only among rows with known C40')
    (REPORT / '04_04_matching_audit.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps(summary,ensure_ascii=False,indent=2))
    return grouped,candidates,ambiguous_context
