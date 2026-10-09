"""07_03: февраль 2020 EGCC, обе метки и конкретные записи без исключений."""
import json
import sys
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import matplotlib.pyplot as plt
import nbformat
from audit_airports import ROOT, INPUT, SOURCE, wilson
from audit_feature_availability import digest

REPORT = ROOT / 'europe/reports/airports/EGCC'
FIGURES = ROOT / 'europe/figures/07_airports/EGCC'


def aggregate(frame, keys):
    return wilson(frame.groupby(keys, dropna=False).GoA.agg(rows='size', go_arounds='sum').reset_index())


def main():
    expected = json.loads((REPORT/'07_02_audit.json').read_text(encoding='utf8'))
    before, source_before = digest(INPUT), digest(SOURCE)
    assert before == expected['input_sha256'] and source_before == expected['source_sha256']
    fields = ['flight_id','icao24','airport','landing_time','GoA','n_attempts',
              'C40_CROSS_TIME','typecode','registration','market_segment','attempt_times']
    frame = pq.read_table(INPUT, columns=fields, filters=[('airport','=','EGCC')]).to_pandas(ignore_metadata=True)
    raw = pq.read_table(SOURCE, columns=fields, filters=[('airport','=','EGCC')]).to_pandas(ignore_metadata=True)
    for data in [frame, raw]:
        data['month'] = data.landing_time.dt.strftime('%Y-%m')
        data['date'] = data.landing_time.dt.strftime('%Y-%m-%d')
        data['first_attempt'] = data.attempt_times.map(lambda x: pd.to_datetime(x[0], utc=True) if len(x) else pd.NaT)
        data['attempt_date'] = data.first_attempt.dt.strftime('%Y-%m-%d')
    feb = frame[frame.month.eq('2020-02')].copy()
    assert len(feb)==4301 and int(feb.GoA.sum())==45
    days = pd.date_range('2020-02-01','2020-02-29').strftime('%Y-%m-%d')
    daily = aggregate(feb,['date']).set_index('date').reindex(days)
    daily[['rows','go_arounds']] = daily[['rows','go_arounds']].fillna(0).astype(int)
    daily = wilson(daily.reset_index(names='date'))
    attempt_daily = aggregate(feb,['attempt_date'])
    monthly = aggregate(frame,['month'])
    source_monthly = aggregate(raw,['month'])
    comparison = monthly.merge(source_monthly,on='month',suffixes=('_current','_source'),how='outer')
    comparison = comparison[comparison.month.isin(['2020-01','2020-02','2020-03']) | comparison.month.str.endswith('-02')]
    fleet=[]
    for field in ['icao24','typecode','market_segment','registration']:
        group=aggregate(feb,[field]);group.insert(0,'field',field);group=group.rename(columns={field:'value'})
        fleet.append(group)
    composition=pd.concat(fleet,ignore_index=True)
    # Сравнение типов использует отдельные знаменатели каждого периода и обе метки.
    type_period=aggregate(frame[frame.month.isin(['2020-01','2020-02','2020-03'])],['month','typecode'])
    positive=feb[feb.GoA].copy()
    key=['icao24','airport','landing_time']
    positive['duplicate_landing_key']=feb.duplicated(key,keep=False).loc[positive.index]
    positive['duplicate_flight_id']=feb.duplicated('flight_id',keep=False).loc[positive.index]
    positive['attempts_json']=positive.attempt_times.map(lambda x:json.dumps([str(v) for v in x]))
    positive['minutes_first_attempt_to_landing']=(positive.landing_time-positive.first_attempt).dt.total_seconds()/60
    positive['crosses_utc_day']=positive.date.ne(positive.attempt_date)
    records=positive.drop(columns='attempt_times').sort_values('first_attempt')
    # Проверка ближайшего следующего сегмента того же борта охватывает весь EGCC.
    ordered=frame.sort_values(['icao24','landing_time']).copy()
    ordered['next_landing_time']=ordered.groupby('icao24',dropna=False).landing_time.shift(-1)
    ordered['next_flight_id']=ordered.groupby('icao24',dropna=False).flight_id.shift(-1)
    ordered['minutes_to_next_landing']=(ordered.next_landing_time-ordered.landing_time).dt.total_seconds()/60
    neighbors=ordered.loc[positive.index].drop(columns='attempt_times')
    calendar=pd.read_csv(ROOT/'europe/reports/airports/airport_calendar.csv')
    parts=[]
    for batch in pq.ParquetFile(SOURCE).iter_batches(columns=['airport','landing_time','GoA'],batch_size=100000):
        chunk=batch.to_pandas(ignore_metadata=True)
        chunk=chunk[chunk.landing_time.dt.strftime('%Y-%m').eq('2020-02')].copy()
        chunk['date']=chunk.landing_time.dt.strftime('%Y-%m-%d')
        parts.append(chunk.groupby(['airport','date']).GoA.agg(rows='size',go_arounds='sum').reset_index())
    source_calendar=pd.concat(parts).groupby(['airport','date'])[['rows','go_arounds']].sum().reset_index()
    cross=calendar[calendar.date.between('2020-02-01','2020-02-29')].copy()
    cross=wilson(cross)
    cross=cross.merge(source_calendar[['airport','date','rows','go_arounds']],on=['airport','date'],suffixes=('','_source'),validate='one_to_one')
    cross['retained_percent']=100*cross.rows/cross.rows_source.replace(0,np.nan)
    peak=daily.sort_values(['go_arounds','rows'],ascending=[False,False]).head(5)
    dates=peak.date.tolist()
    cross_peak=cross[cross.date.isin(dates)].sort_values(['date','go_arounds'],ascending=[True,False])
    for data,name in [(daily,'daily'),(attempt_daily,'first_attempt_daily'),(comparison,'month_comparison'),
                      (composition,'composition'),(type_period,'type_period'),(records,'positive_records'),
                      (neighbors,'next_segments'),(cross,'other_airports_daily'),(cross_peak,'peak_day_comparison')]:
        data.to_csv(REPORT/f'07_03_{name}.csv',index=False)
    plt.rcParams.update({'font.family':'DejaVu Sans','svg.fonttype':'none','pdf.fonttype':42})
    fig,axes=plt.subplots(3,1,figsize=(12,10),sharex=True)
    x=np.arange(len(daily))
    axes[0].bar(x,daily.rows,color='#537c99');axes[0].set_ylabel('Все записи')
    axes[1].bar(x,daily.go_arounds,color='#b45335');axes[1].set_ylabel('GoA=True')
    axes[2].errorbar(x,daily.ga_percent,yerr=np.vstack([daily.ga_percent-daily.wilson_lower_percent,daily.wilson_upper_percent-daily.ga_percent]),fmt='o',capsize=2)
    axes[2].set_ylabel('GoA, %');axes[2].set_xticks(x,[d[-2:] for d in daily.date]);axes[2].set_xlabel('День февраля 2020, посадка UTC')
    fig.suptitle('EGCC: число наблюдений и отмеченных уходов в феврале 2020')
    for ax in axes:ax.grid(axis='y',alpha=.15)
    fig.tight_layout(rect=(0,.04,1,.96));fig.text(.01,.01,'95% Уилсон: точечные интервалы; зависимость записей и качество разметки не учтены.',fontsize=9)
    for ext in ['png','svg','pdf']:fig.savefig(FIGURES/f'07_03_february_daily.{ext}',dpi=300,bbox_inches='tight')
    plt.close(fig)
    summary=dict(input_sha256=before,source_sha256=source_before,rows=len(feb),go_arounds=int(feb.GoA.sum()),
        unique_positive_aircraft=int(positive.icao24.nunique()),unique_positive_types=int(positive.typecode.nunique()),
        duplicate_positive_keys=int(positive.duplicate_landing_key.sum()),duplicate_positive_flight_ids=int(positive.duplicate_flight_id.sum()),
        positive_crosses_utc_day=int(positive.crosses_utc_day.sum()),top_five_days=dates,top_five_go_arounds=int(peak.go_arounds.sum()),
        rows_deleted=0,values_changed=0,weather_status='pending',causality_established=False)
    assert digest(INPUT)==before and digest(SOURCE)==source_before
    (REPORT/'07_03_audit.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    text='# 07_03 – EGCC: всплеск февраля 2020\n\n'
    text+=f'Актуальная выборка: {len(feb)} записей, {int(feb.GoA.sum())} GoA=True ({100*feb.GoA.mean():.3f}%). Единица учета – запись посадочного сегмента. Обе метки включены в знаменатели.\n\n'
    text+='## Распределение по дням\n\n| День посадки UTC | n | GoA | Доля, % |\n| --- | ---: | ---: | ---: |\n'
    for r in daily.itertuples():text+=f'| {r.date} | {r.rows} | {r.go_arounds} | {r.ga_percent:.3f} |\n'
    text+=f'\nНа пять дней с наибольшим числом GoA приходится {int(peak.go_arounds.sum())} из 45 записей. Это описательное ранжирование после просмотра данных. Даты первой попытки отдельно сохранены; переход суток: {summary["positive_crosses_utc_day"]} записей.\n\n'
    text+='## Борта, типы и повторные записи\n\n'
    text+=f'Положительных бортов: {summary["unique_positive_aircraft"]}; известных кодов типа: {summary["unique_positive_types"]}. Повтор составного ключа среди положительных: {summary["duplicate_positive_keys"]}; повтор flight_id: {summary["duplicate_positive_flight_ids"]}. Отсутствие точного повтора не подтверждает уникальность физического полета.\n\n'
    for field in ['icao24','typecode']:
        text+=f'Наибольшие группы {field}:\n\n| Значение | n всех | GoA | Доля, % |\n| --- | ---: | ---: | ---: |\n'
        for r in composition[composition.field.eq(field)].sort_values('go_arounds',ascending=False).head(10).itertuples():text+=f'| {r.value} | {r.rows} | {r.go_arounds} | {r.ga_percent:.3f} |\n'
        text+='\n'
    text+='## Сравнение периодов\n\n| Месяц | n текущей | GoA текущей | Доля, % | n исходника | GoA исходника |\n| --- | ---: | ---: | ---: | ---: | ---: |\n'
    for r in comparison.sort_values('month').itertuples():text+=f'| {r.month} | {r.rows_current} | {r.go_arounds_current} | {r.ga_percent_current:.3f} | {r.rows_source} | {r.go_arounds_source} |\n'
    text+='\nПоздние феврали без C40 представлены исходником; они не сравниваются как одинаковая модельная выборка. Пропуски не исправлены. Состав типов января–марта и ежедневные n/k остальных аэропортов сохранены отдельно, вместе с охватом исходника.\n\n'
    text+='## Конкретные записи и границы вывода\n\nВсе 45 положительных записей сохранены с исходными идентификаторами и временами попыток в `07_03_positive_records.csv`. Следующие сегменты тех же бортов: `07_03_next_segments.csv`. Это кандидаты для проверки, а не подтвержденные ошибки. Независимая проверка разметки по траекториям не выполнена.\n\nПогодная проверка выполняется отдельно по архивным METAR, с сохранением исходных сообщений. Причина всплеска пока не установлена. Новые исключения отсутствуют; SHA256 входа и исходника до и после совпадают.\n'
    (REPORT/'07_03_review.md').write_text(text,encoding='utf8')
    notebook=nbformat.v4.new_notebook(cells=[nbformat.v4.new_markdown_cell(text),nbformat.v4.new_code_cell("from pathlib import Path\nimport pandas as pd\nroot = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p/'europe/scripts/audit_egcc_february.py').exists())\nreport = root/'europe/reports/airports/EGCC'\npd.read_csv(report/'07_03_positive_records.csv')"),nbformat.v4.new_markdown_cell('![Дневная динамика](../../../figures/07_airports/EGCC/07_03_february_daily.png)')],metadata={'kernelspec':{'name':'python3','display_name':'Python 3','language':'python'}})
    nbformat.validate(notebook)
    nbformat.write(notebook,ROOT/'europe/notebooks/07_airports/EGCC/07_03_egcc_february.ipynb')
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)
    print(peak.to_string(index=False),flush=True)
    print(comparison.to_string(index=False),flush=True)


if __name__=='__main__':
    if hasattr(sys.stdout,'reconfigure'):sys.stdout.reconfigure(encoding='utf8')
    main()
