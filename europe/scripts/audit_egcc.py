"""07_02 EGCC: происхождение временного провала и состав обеих меток."""
import json
import sys
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import matplotlib.pyplot as plt
from audit_airports import ROOT, INPUT, SOURCE, wilson, save
import audit_airports
from audit_feature_availability import digest

REPORT=ROOT/'europe/reports/airports/EGCC'
FIGURES=ROOT/'europe/figures/07_airports/EGCC'
CUT=pd.Timestamp('2021-03-01',tz='UTC')
KEY=['icao24','airport','landing_time']


def extract(path,columns):
    # Arrow читает только нужные поля; фильтр ограничивает диагностический результат EGCC.
    return pq.read_table(path,columns=columns,filters=[('airport','=', 'EGCC')]).to_pandas(ignore_metadata=True)


def monthly(frame,group):
    frame=frame.copy();frame['month']=frame.landing_time.dt.strftime('%Y-%m')
    result=frame.groupby(['month',*group],dropna=False).GoA.agg(rows='size',go_arounds='sum').reset_index()
    return wilson(result)


def main():
    REPORT.mkdir(parents=True,exist_ok=True);FIGURES.mkdir(parents=True,exist_ok=True)
    overview=json.loads((ROOT/'europe/reports/airports/07_01_audit.json').read_text(encoding='utf8'))
    before=digest(INPUT);source_before=digest(SOURCE)
    assert before==overview['input_sha256'] and source_before==overview['source_sha256']
    fields=['flight_id',*KEY,'GoA','n_attempts','C40_CROSS_TIME','typecode','market_segment']
    print('Чтение EGCC: исходник и актуальный вход',flush=True)
    raw=extract(SOURCE,[*fields,'attempt_times'])
    current=extract(INPUT,fields)
    assert len(raw)==211299 and int(raw.GoA.sum())==791
    assert len(current)==87435 and int(current.GoA.sum())==306
    assert not raw.duplicated(KEY).any() and not current.duplicated(KEY).any()
    first=raw.attempt_times.map(lambda values:pd.to_datetime(values[0],utc=True))
    raw['c40_status']=np.select([raw.C40_CROSS_TIME.isna(),raw.C40_CROSS_TIME.ge(first)],
        ['missing','not_strictly_before_first'],default='strictly_before_first')
    matched=raw.merge(current[KEY].assign(retained=True),on=KEY,how='left',validate='one_to_one')
    matched['retained']=matched.retained.eq(True)
    assert int(matched.retained.sum())==len(current)
    dropped_valid=matched[~matched.retained & matched.c40_status.eq('strictly_before_first')]
    pairs=extract(ROOT/'europe/data/review/04_05_c40_pair_full_records.parquet',KEY)
    assert len(dropped_valid)==6
    assert len(dropped_valid.merge(pairs,on=KEY,validate='one_to_one'))==6
    assert not dropped_valid.GoA.any()
    matched['selection_reason']=np.where(matched.retained,'retained',
        np.where(matched.c40_status.eq('strictly_before_first'),'ambiguous_pair_04_08',matched.c40_status))
    matched['period']=np.where(matched.landing_time.lt(CUT),'before_2021_03','from_2021_03')
    period=wilson(matched.groupby(['period','selection_reason']).GoA.agg(rows='size',go_arounds='sum').reset_index())
    period.to_csv(REPORT/'07_02_period_reasons.csv',index=False)
    reasons=monthly(matched,['selection_reason']);reasons.to_csv(REPORT/'07_02_monthly_reasons.csv',index=False)
    raw_month=monthly(raw,[]);cur_month=monthly(current,[])
    months=pd.date_range('2019-01-01','2023-05-01',freq='MS').strftime('%Y-%m')
    def full(frame):
        frame=frame.set_index('month').reindex(months);frame.index.name='month'
        frame[['rows','go_arounds']]=frame[['rows','go_arounds']].fillna(0).astype('int64')
        return wilson(frame.reset_index())
    raw_month=full(raw_month);cur_month=full(cur_month)
    compare=raw_month.merge(cur_month,on='month',suffixes=('_source','_current'),validate='one_to_one')
    compare.to_csv(REPORT/'07_02_monthly_comparison.csv',index=False)
    dropped_valid.drop(columns='attempt_times').to_csv(REPORT/'07_02_prior_pair_exclusions.csv',index=False)
    fleet=[]
    for label,frame in [('source_before_cut',raw[raw.landing_time.lt(CUT)]),('source_after_cut',raw[raw.landing_time.ge(CUT)]),('current',current)]:
        for field in ['typecode','market_segment']:
            f=frame.copy();f['value']=f[field].fillna('(пропуск)')
            g=f.groupby(['GoA','value']).size().reset_index(name='rows');g['field']=field;g['population']=label
            totals=frame.groupby('GoA').size();g['within_label_percent']=100*g.rows/g.GoA.map(totals)
            fleet.append(g)
    pd.concat(fleet,ignore_index=True).to_csv(REPORT/'07_02_composition.csv',index=False)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
        'axes.spines.right':False,'svg.fonttype':'none','pdf.fonttype':42})
    audit_airports.FIGURES=FIGURES
    x=np.arange(len(months));ticks=list(range(0,len(months),6));cut_index=list(months).index('2021-03')
    fig,axes=plt.subplots(2,1,figsize=(12,8),sharex=True)
    axes[0].plot(x,compare.rows_source,color='#537c99',label='Исходник')
    axes[0].plot(x,compare.rows_current,color='#b45335',label='Текущая выборка')
    axes[0].set_ylabel('Число записей');axes[0].set_title('EGCC: исходное покрытие и доступность C40',fontweight='bold');axes[0].legend()
    for label,color in [(False,'#537c99'),(True,'#b45335')]:
        r=matched[matched.GoA.eq(label)].copy();r['month']=r.landing_time.dt.strftime('%Y-%m')
        values=r.groupby('month').retained.mean().reindex(months)
        axes[1].plot(x,100*values,color=color,label=f'GoA={label}')
    axes[1].set(ylabel='Сохранено исходной метки, %',ylim=(-3,103));axes[1].legend()
    for ax in axes:ax.axvline(cut_index-.5,color='#697680',linestyle='--');ax.grid(alpha=.15)
    axes[1].set_xticks(ticks,[months[i] for i in ticks],rotation=45,ha='right')
    save(fig,'07_02_egcc_coverage','Пунктир: март 2021. Нулевой охват при наличии исходных записей не означает отсутствие полетов.')
    fig,ax=plt.subplots(figsize=(12,5))
    for label,color in [('source','#537c99'),('current','#b45335')]:
        ax.plot(x,compare[f'ga_percent_{label}'],color=color,label='Исходник' if label=='source' else 'Текущая выборка')
        ax.fill_between(x,compare[f'wilson_lower_percent_{label}'],compare[f'wilson_upper_percent_{label}'],color=color,alpha=.14)
    ax.axvline(cut_index-.5,color='#697680',linestyle='--');ax.set(title='EGCC: месячная доля GoA и 95% интервалы Уилсона',ylabel='Доля GoA=True, %')
    ax.set_xticks(ticks,[months[i] for i in ticks],rotation=45,ha='right');ax.legend();ax.grid(alpha=.15)
    save(fig,'07_03_egcc_rates','Нет сохраненных строк → разрыв линии, а не нулевая доля. Интервалы точечные; зависимость полетов не устранена.')
    status=reasons.pivot(index='month',columns='selection_reason',values='rows').reindex(months).fillna(0)
    fig,ax=plt.subplots(figsize=(12,5));bottom=np.zeros(len(months))
    for name,title,color in [('retained','Сохранены','#537c99'),('missing','Нет C40','#b45335'),('ambiguous_pair_04_08','Пары 04_08','#8b8197'),('not_strictly_before_first','C40 не раньше попытки','#be9b54')]:
        if name in status:
            ax.bar(x,status[name],bottom=bottom,label=title,color=color,width=.85);bottom+=status[name].to_numpy()
    ax.set(title='EGCC: причины отсутствия в эксперименте по месяцам',ylabel='Число исходных записей')
    ax.set_xticks(ticks,[months[i] for i in ticks],rotation=45,ha='right');ax.legend();ax.grid(axis='y',alpha=.15)
    save(fig,'07_04_egcc_reasons','Столбцы полностью разделяют исходные строки. Шесть строк исключены раньше как неоднозначные пары; остальные потери объясняются C40.')
    after=matched[matched.landing_time.ge(CUT)];pre=matched[matched.landing_time.lt(CUT)]
    assert after.C40_CROSS_TIME.isna().all() and not after.retained.any()
    calendar=pd.read_csv(ROOT/'europe/reports/airports/airport_calendar.csv')
    calendar=calendar[calendar.airport.eq('EGCC')].copy()
    source_daily=raw.groupby(raw.landing_time.dt.strftime('%Y-%m-%d')).size()
    calendar['source_rows']=calendar.date.map(source_daily).fillna(0).astype('int64')
    calendar['new_empty_after_selection']=calendar.rows.eq(0)&calendar.source_rows.gt(0)
    calendar.to_csv(REPORT/'07_02_calendar_difference.csv',index=False)
    tail=calendar[calendar.date.ge('2021-03-01')]
    summary=dict(checked_date='2026-10-09',airport='EGCC',input_sha256=before,source_sha256=source_before,
        source_rows=len(raw),source_go_arounds=int(raw.GoA.sum()),current_rows=len(current),current_go_arounds=int(current.GoA.sum()),
        cutoff_utc=CUT.isoformat(),source_after_cut_rows=len(after),source_after_cut_go_arounds=int(after.GoA.sum()),
        all_c40_missing_after_cut=True,prior_ambiguous_pair_exclusions=6,prior_excluded_positive=0,
        source_before_cut_rows=len(pre),source_before_cut_go_arounds=int(pre.GoA.sum()),
        latest_retained_landing_utc=current.landing_time.max().isoformat(),
        source_after_cut_first_utc=after.landing_time.min().isoformat(),source_after_cut_last_utc=after.landing_time.max().isoformat(),
        tail_calendar_days=len(tail),tail_days_with_source_records=int(tail.source_rows.gt(0).sum()),
        tail_days_without_source_records=int(tail.source_rows.eq(0).sum()),
        rows_deleted=0,values_changed=0,c40_missing_cause_established=False,airport_exclusion_applied=False)
    assert digest(INPUT)==before and digest(SOURCE)==source_before
    (REPORT/'07_02_audit.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    text=f'''# 07_02 – EGCC: временное покрытие и C40

Проверены все {len(raw):,} исходных записей EGCC (791 GoA=True) и все {len(current):,} сохраненных (306 GoA=True). Источники проверены по SHA256 до и после; новых исключений и исправлений нет.

## Что установлено

С 01.03.2021 в исходнике есть **{len(after):,} записей**, включая **{int(after.GoA.sum()):,} GoA=True**, но у всех отсутствует C40_CROSS_TIME. Ни одна не попала в текущий C40-эксперимент. Последняя сохраненная посадка: {summary['latest_retained_landing_utc']}.

Длинный разрыв до общей границы 30.05.2023 составляет {len(tail)} дней: в {summary['tail_days_with_source_records']} есть исходные записи, но нет сохраненных из-за C40; в {summary['tail_days_without_source_records']} дней записей нет уже в исходнике. Последняя исходная запись EGCC: {summary['source_after_cut_last_utc']}. Отсутствие записей само по себе не доказывает прекращение рейсов или уходов.

До границы марта 2021 исходник содержит {len(pre):,} строк, {int(pre.GoA.sum()):,} GoA=True. Шесть строк с допустимым C40 не сохранены из-за прежнего исключения неоднозначных пар 04_08; все GoA=False. Совпадение этих строк с архивом пар проверено по составному ключу icao24 + airport + landing_time.

## Причины и сравнение классов

| Период | Причина | n | GoA=True | Доля, % |
| --- | --- | ---: | ---: | ---: |
'''
    for row in period.itertuples():text+=f'| {row.period} | {row.selection_reason} | {row.rows:,} | {row.go_arounds:,} | {row.ga_percent:.3f} |\n'
    text+='''
Общий охват 41,4% складывается прежде всего из потери целого позднего периода. Нельзя интерпретировать отличие общих долей GoA исходника и текущей выборки как чистое влияние фильтра: сравниваются разные календарные составы. Таблица composition.csv разделяет исходный ранний период, поздний период и текущую выборку, отдельно по обоим классам.

## Границы вывода и следующий вопрос

Механизм отсутствия C40 не установлен: возможен разрыв обогащения или сопоставления источников, но это гипотеза. Не объявлять поздние полеты ошибочными и не восстанавливать C40 по посадке. Исходные поздние записи уже сохранены вне эксперимента этапа 04. Ни неисправность борта, ни прекращение полетов из этих данных не следуют.

EGCC пока сохраняется. Следующее решение – как использовать его неполный период в сравнении аэропортов и временном разбиении модели. Удаление всего аэропорта не требуется автоматически. Февраль 2020 с 45 отмеченными уходами в 4 301 сохраненной записи остается отдельным вопросом; погодная причина этим разбором не проверена.

## Артефакты

`07_02_period_reasons.csv`, `07_02_monthly_reasons.csv`, `07_02_monthly_comparison.csv`, `07_02_prior_pair_exclusions.csv`, `07_02_composition.csv`, `07_02_calendar_difference.csv`, `07_02_audit.json`. Графики – `europe/figures/07_airports/EGCC/`, PNG 300 dpi, SVG и PDF. Пропуски типов и сегментов в таблице состава показаны как категория, исходные значения не переписаны.
'''
    decision_path=REPORT/'07_02_user_decision.json'
    if decision_path.exists():
        decision=json.loads(decision_path.read_text(encoding='utf8'))
        assert decision['input_sha256']==before, 'Решение относится к другой версии входа'
        text=text.replace('EGCC пока сохраняется. Следующее решение – как использовать его неполный период в сравнении аэропортов и временном разбиении модели. Удаление всего аэропорта не требуется автоматически.',
                          'EGCC сохранен по принятому решению пользователя; ограничения и план чувствительности приведены ниже.')
        text+='\n## Принятое решение пользователя\n\nСохранить доступный период EGCC. При моделировании сравнить обучение с EGCC и без него на одинаковом тесте остальных аэропортов. Частоты дополнительно сравнить за общий календарный период; поздние записи без C40 используются описательно. Решение по составу закрыто, моделирование и чувствительность еще не выполнены.\n\n[Решение и план](07_02_decision_and_sensitivity.md). Журнал: `07_02_user_decision.json`.\n'
    (REPORT/'07_02_review.md').write_text(text,encoding='utf8')
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)
    print(period.to_string(index=False),flush=True)


if __name__=='__main__':
    if hasattr(sys.stdout,'reconfigure'):sys.stdout.reconfigure(encoding='utf8')
    main()
