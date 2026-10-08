"""Покрытие C40 по группам и графики без фильтрации наблюдений."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import matplotlib.pyplot as plt
from audit_feature_availability import INPUT, REPORT, ROOT, digest

FIGURES = ROOT / 'europe/figures/04_analytical_sample'


def main():
    # Проверяем вход и агрегируем блоками, не загружая полную таблицу в память
    before = digest(INPUT)
    journal = json.loads((REPORT / '04_02_feature_audit.json').read_text(encoding='utf8'))
    assert before == journal['input_sha256']
    parts = {'airport': [], 'month': [], 'market_segment': []}
    gaps = []
    rows = events = 0
    columns = ['airport', 'landing_time', 'market_segment', 'GoA', 'C40_CROSS_TIME',
               'attempt_times', 'c40_time_missing', 'c40_after_first_attempt']
    for batch in pq.ParquetFile(INPUT).iter_batches(batch_size=100_000, columns=columns):
        table = pa.Table.from_batches([batch])
        frame = table.select([c for c in columns if c != 'attempt_times']).to_pandas(ignore_metadata=True)
        frame['month'] = frame.landing_time.dt.strftime('%Y-%m')
        frame['c40_group'] = np.where(frame.c40_time_missing, 'missing',
                                     np.where(frame.c40_after_first_attempt, 'after_first_attempt', 'present_not_after_first'))
        for field in parts:
            parts[field].append(frame.groupby([field, 'c40_group'], dropna=False).agg(
                rows=('GoA', 'size'), go_arounds=('GoA', 'sum')).reset_index())
        first = pc.list_element(table['attempt_times'], 0).to_numpy(zero_copy_only=False).astype('datetime64[ns]')
        c40 = table['C40_CROSS_TIME'].to_numpy(zero_copy_only=False).astype('datetime64[ns]')
        valid = ~np.isnat(c40)
        gaps.append((first[valid] - c40[valid]) / np.timedelta64(1, 'm'))
        rows += len(frame)
        events += int(frame.GoA.sum())
    assert rows == journal['rows_checked'] and events == journal['go_arounds']
    output = {}
    for field, pieces in parts.items():
        grouped = pd.concat(pieces).groupby([field, 'c40_group'], dropna=False)[['rows','go_arounds']].sum().reset_index()
        grouped['ga_percent'] = 100*grouped.go_arounds/grouped.rows
        totals = grouped.groupby(field, dropna=False)[['rows', 'go_arounds']].transform('sum')
        grouped['within_group_row_percent'] = 100*grouped.rows/totals.rows
        grouped['within_group_positive_percent'] = np.where(totals.go_arounds > 0,100*grouped.go_arounds/totals.go_arounds,np.nan)
        assert grouped.rows.sum() == rows and grouped.go_arounds.sum() == events
        grouped.to_csv(REPORT / f'04_03_c40_by_{field}.csv', index=False)
        output[field] = grouped
    gap = np.concatenate(gaps)
    assert np.isfinite(gap).all()
    assert int((gap < 0).sum()) == 87
    assert digest(INPUT) == before
    summary = dict(input_sha256=before, input_unchanged=True, rows_checked=rows,
                   go_arounds=events, rows_changed=0, rows_deleted=0,
                   present_c40_rows=len(gap), c40_equal_to_first_attempt_rows=int((gap==0).sum()),
                   c40_before_first_attempt_rows=int((gap>0).sum()),
                   first_attempt_minus_c40_minutes=dict(zip(['min','p05','median','p95','max'],
                                                          np.quantile(gap,[0,.05,.5,.95,1]).tolist())),
                   grouping_time='UTC month of final landing, retrospective grouping only',
                   market_segment_is_verified_flight_purpose=False,
                   trajectory_matching_verified=False)
    (REPORT / '04_03_c40_audit.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps(summary,ensure_ascii=False,indent=2))
    return output


def save(fig, name):
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIGURES / f'{name}.png', dpi=160)
    fig.savefig(FIGURES / f'{name}.svg')
    return fig


def missing_chart(data, field, name, title):
    # Пересчитываем долю пропусков с полным знаменателем для каждой группы
    total = data.groupby(field,dropna=False).rows.sum()
    absent = data.loc[data.c40_group=='missing'].groupby(field,dropna=False).rows.sum().reindex(total.index,fill_value=0)
    values = (100*absent/total).sort_values()
    labels = [str(x) if pd.notna(x) else 'Сегмент не указан' for x in values.index]
    fig, ax = plt.subplots(figsize=(10, max(4, .32*len(values))))
    bars = ax.barh(labels, values, color='#477ca8')
    ax.bar_label(bars, labels=[f'{x:.1f}%' for x in values],padding=3,fontsize=8)
    ax.set(xlim=(0,112),xlabel='Доля строк без времени C40, %',title=title)
    ax.grid(axis='x',alpha=.2)
    return save(fig,name)


def monthly_chart(data):
    # Месяц конечной посадки используется для аудита покрытия, а не как будущий признак
    total = data.groupby('month').rows.sum()
    positive = data.groupby('month').go_arounds.sum()
    absent = data.loc[data.c40_group=='missing'].set_index('month').reindex(total.index)
    fig, ax = plt.subplots(figsize=(12,5))
    x = np.arange(len(total))
    ax.plot(x,100*absent.rows.fillna(0)/total,label='Доля строк без C40',color='#477ca8')
    ax.plot(x,100*absent.go_arounds.fillna(0)/positive.replace(0,np.nan),
            label='Доля меток GoA=True без C40',color='#cc7941')
    positions = np.arange(0,len(total),3)
    ax.set_xticks(positions,total.index[positions],rotation=55,ha='right')
    ax.set(ylim=(0,105),ylabel='Доля, %',title='Покрытие C40 по месяцам конечной посадки (UTC)')
    ax.legend()
    ax.grid(alpha=.2)
    return save(fig,'04_03_monthly_missing')
