"""Аудит покрытия C40 и выбор групп пилота без отбора по частоте GoA."""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / 'europe/weather'
INPUT = ROOT / 'europe/data/analytical/06_05_cases_excluded.parquet'


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    before = digest(INPUT)
    columns = ['airport', 'GoA', 'C40_CROSS_TIME', 'landing_time',
               'c40_after_first_attempt', 'c40_after_landing']
    parts = []
    totals = dict(rows=0, go_arounds=0, c40_missing=0)
    for batch in pq.ParquetFile(INPUT).iter_batches(columns=columns, batch_size=200_000):
        frame = batch.to_pandas()
        totals['rows'] += len(frame)
        totals['go_arounds'] += int(frame.GoA.sum())
        totals['c40_missing'] += int(frame.C40_CROSS_TIME.isna().sum())
        # Месяц посадки используется только для описания покрытия исходной базы.
        frame['month'] = frame.landing_time.dt.strftime('%Y-%m')
        frame['c40_present'] = frame.C40_CROSS_TIME.notna()
        frame['c40_early'] = (frame.c40_present & ~frame.c40_after_first_attempt
                              & ~frame.c40_after_landing)
        parts.append(frame.groupby(['airport', 'month'], dropna=False).agg(
            rows=('GoA', 'size'), go_arounds=('GoA', 'sum'),
            c40_present=('c40_present', 'sum'), c40_early=('c40_early', 'sum')))
    groups = pd.concat(parts).groupby(level=[0, 1], dropna=False).sum().reset_index()
    airports = groups.groupby('airport', dropna=False)[
        ['rows', 'go_arounds', 'c40_present', 'c40_early']].sum()
    airports['c40_early_percent'] = 100 * airports.c40_early / airports.rows
    airports = airports.sort_values(['c40_early', 'rows'], ascending=False)
    # EGCC — заранее согласованный пример; еще два аэропорта по числу ранних C40.
    selected = ['EGCC'] + [a for a in airports.index if a != 'EGCC'][:2]
    assert before == digest(INPUT), 'Вход изменился во время аудита'
    report = BASE / 'reports'
    report.mkdir(parents=True, exist_ok=True)
    groups.to_csv(report / '01_airport_month_coverage.csv', index=False)
    airports.to_csv(report / '01_airport_coverage.csv')
    journal = dict(input=str(INPUT.relative_to(ROOT)), input_sha256=before,
                   checked_utc=datetime.now(timezone.utc).isoformat(), **totals,
                   selected_airports=selected,
                   selection='EGCC plus top two other airports by early C40 count; no GoA ranking',
                   rows_changed=0, rows_deleted=0, input_unchanged=True)
    (report / '01_input_audit.json').write_text(
        json.dumps(journal, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
    print(json.dumps(journal, ensure_ascii=False, indent=2))
    print(airports.head(10).to_string())


if __name__ == '__main__':
    main()
