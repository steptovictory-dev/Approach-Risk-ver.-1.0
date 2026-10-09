"""07_01: полный описательный аудит аэропортов без изменения выборки."""
from collections import Counter
from datetime import datetime, timezone
import json
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from audit_feature_availability import ROOT, digest

INPUT = ROOT / 'europe/data/analytical/06_05_cases_excluded.parquet'
SOURCE = ROOT / 'europe/data/source/osn23_landings_merged.parquet.gzip'
REPORT = ROOT / 'europe/reports/airports'
FIGURES = ROOT / 'europe/figures/07_airports'
FIELDS = ['registration', 'typecode', 'icaoaircrafttype', 'aircraft_description_restored',
          'market_segment', 'AC_CLASS', 'AP_C_RWY']
Z = 1.959963984540054


def wilson(table):
    """Точечные биномиальные интервалы; не поправка на множественные сравнения."""
    table = table.copy()
    n = table.rows.to_numpy(dtype=float)
    k = table.go_arounds.to_numpy(dtype=float)
    p = np.divide(k, n, out=np.full_like(n, np.nan), where=n > 0)
    safe = np.where(n > 0, n, 1)
    denominator = 1 + Z**2 / safe
    center = (p + Z**2 / (2 * safe)) / denominator
    half = Z * np.sqrt(p * (1-p) / safe + Z**2 / (4 * safe**2)) / denominator
    table['goa_false'] = table.rows - table.go_arounds
    table['ga_percent'] = 100 * p
    table['wilson_lower_percent'] = 100 * np.where(k==0,0,np.maximum(0, center-half))
    table['wilson_upper_percent'] = 100 * np.where((k==n)&(n>0),1,np.minimum(1, center+half))
    table['max_deviation_pp'] = np.maximum(table.ga_percent-table.wilson_lower_percent,
                                         table.wilson_upper_percent-table.ga_percent)
    return table


def collapse(parts):
    frame = pd.concat(parts)
    return frame.groupby(level=list(range(frame.index.nlevels))).sum().reset_index()


def scan(path, detailed=False):
    """Потоковый проход охватывает обе метки; сырые пропуски не исправляются."""
    columns = ['airport', 'landing_time', 'GoA']
    if detailed:
        columns += ['icao24', 'n_attempts', 'C40_CROSS_TIME', 'C40_BEARING',
                    'C40_CROSS_LAT', 'C40_CROSS_LON', *FIELDS]
    months, days, missing, categories = [], [], [], []
    aircraft = Counter()
    for batch in pq.ParquetFile(path).iter_batches(batch_size=100_000, columns=columns):
        frame = batch.to_pandas(ignore_metadata=True)
        assert frame.airport.notna().all() and frame.GoA.notna().all()
        counts = pd.DataFrame(dict(airport=frame.airport,
            month=frame.landing_time.dt.strftime('%Y-%m'), rows=1,
            go_arounds=frame.GoA.astype('int64')))
        months.append(counts.groupby(['airport', 'month']).sum(numeric_only=True))
        counts['date'] = frame.landing_time.dt.strftime('%Y-%m-%d')
        days.append(counts.groupby(['airport', 'date'])[['rows', 'go_arounds']].sum())
        if not detailed:
            continue
        flags = pd.DataFrame(dict(airport=frame.airport, GoA=frame.GoA, rows=1))
        for field in FIELDS:
            flags[field] = (frame[field].isna() | frame[field].eq('').fillna(False)).astype('int64')
        # Производная классификация заполняет только часть исходных пропусков.
        usable = frame.icaoaircrafttype.where(frame.icaoaircrafttype.notna(), frame.aircraft_description_restored)
        flags['classification_unavailable'] = usable.isna().astype('int64')
        flags['more_than_two_attempts'] = frame.n_attempts.gt(2).astype('int64')
        flags['c40_not_before_landing'] = frame.C40_CROSS_TIME.ge(frame.landing_time).astype('int64')
        flags['c40_coordinate_out_of_range'] = (~frame.C40_CROSS_LAT.between(-90,90) |
                                               ~frame.C40_CROSS_LON.between(-180,180)).astype('int64')
        missing.append(flags.groupby(['airport', 'GoA']).sum(numeric_only=True))
        for field in ['market_segment', 'typecode']:
            group = pd.DataFrame(dict(airport=frame.airport, GoA=frame.GoA,
                field=field, value=frame[field].fillna('(пропуск)'), rows=1))
            categories.append(group.groupby(['airport','GoA','field','value']).sum(numeric_only=True))
        for key, value in frame.groupby(['airport','icao24'], dropna=False).size().items():
            aircraft[key] += int(value)
    return collapse(months), collapse(days), (collapse(missing) if detailed else None), (
        collapse(categories) if detailed else None), aircraft


def save(fig, name, caption, layout=True):
    if layout:
        fig.tight_layout(rect=(0,.075,1,.96))
    fig.text(.01, .012, caption, fontsize=8, color='#4d5966', va='bottom')
    for extension in ['png', 'svg', 'pdf']:
        fig.savefig(FIGURES / f'{name}.{extension}', dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def plots(airports, monthly, coverage, missing):
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
        'axes.spines.right':False,'axes.titleweight':'bold','svg.fonttype':'none',
        'pdf.fonttype':42,'axes.labelcolor':'#243746','text.color':'#243746'})
    order = airports.sort_values('rows', ascending=True).airport.tolist()
    a = airports.set_index('airport').loc[order]
    fig, ax = plt.subplots(figsize=(9,8))
    ax.barh(order, a.goa_false, color='#537c99', label='GoA=False')
    ax.barh(order, a.go_arounds, left=a.goa_false, color='#b45335', label='GoA=True')
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x,_:f'{x/1000:,.0f}'.replace(',',' ') ))
    ax.set(xlabel='Число записей, тыс.', title='Состав аналитической выборки по аэропортам')
    ax.legend(loc='lower right');ax.grid(axis='x',alpha=.15);ax.set_axisbelow(True)
    save(fig,'07_01_airport_volume','Все аэропорты; обе метки. Числа относятся к сохраненным наблюдениям, а не ко всем операциям аэропорта.')

    a = airports.sort_values('ga_percent').reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(9,8)); y=np.arange(len(a))
    ax.errorbar(a.ga_percent,y,xerr=np.vstack([a.ga_percent-a.wilson_lower_percent,
        a.wilson_upper_percent-a.ga_percent]),fmt='o',color='#245b78',capsize=3)
    ax.set_yticks(y, [f'{row.airport}  (k={row.go_arounds:,}; n={row.rows:,})'.replace(',',' ') for row in a.itertuples()])
    overall=100*a.go_arounds.sum()/a.rows.sum();ax.axvline(overall,color='#b45335',linestyle='--',label=f'Общая доля {overall:.3f}%')
    ax.set(xlabel='Доля GoA=True, %',title='Доли отмеченных уходов и 95% интервалы Уилсона')
    ax.grid(axis='x',alpha=.2);ax.legend(loc='lower right')
    save(fig,'07_02_airport_rates','Точечные интервалы: условная независимость и общая вероятность внутри группы. Рейтинг не является оценкой безопасности.')

    months=sorted(monthly.month.unique()); order=airports.sort_values('rows',ascending=False).airport.tolist()
    full=pd.MultiIndex.from_product([order,months],names=['airport','month'])
    m=monthly.set_index(['airport','month']).reindex(full,fill_value=0).reset_index()
    fig,ax=plt.subplots(figsize=(13,7))
    values=m.pivot(index='airport',columns='month',values='rows').loc[order,months]
    im=ax.imshow(np.log10(values+1),aspect='auto',cmap='Blues',interpolation='nearest')
    ax.set_yticks(range(len(order)),order);ticks=list(range(0,len(months),6));ax.set_xticks(ticks,[months[i] for i in ticks],rotation=45,ha='right')
    ax.set(title='Временное покрытие: аэропорт × месяц',xlabel='Месяц UTC')
    fig.colorbar(im,ax=ax,label='log10(число записей + 1)')
    save(fig,'07_03_monthly_coverage','Светлая ячейка означает малое число или отсутствие записей в выборке; причину отсутствия этот график не устанавливает.')

    fig,ax=plt.subplots(figsize=(13,8))
    vals=wilson(m).pivot(index='airport',columns='month',values='ga_percent').loc[order,months]
    # Нулевой знаменатель отображается как отсутствие данных, не как нулевая доля.
    im=ax.imshow(np.ma.masked_invalid(vals),aspect='auto',cmap='YlOrBr',interpolation='nearest')
    ax.set_yticks(range(len(order)),order);ax.set_xticks(ticks,[months[i] for i in ticks],rotation=45,ha='right')
    ax.set(title='Доля GoA=True по аэропортам и месяцам',xlabel='Месяц UTC');fig.colorbar(im,ax=ax,label='Доля GoA=True, %')
    save(fig,'07_04_monthly_rates','Белое поле: нет записей. Различия могут отражать состав и разметку; числа n и k доступны в таблице airport_month.csv.')

    coverage_order=coverage.sort_values('retained_rows_percent',ascending=False).airport.tolist()
    c=coverage.set_index('airport').loc[coverage_order]
    fig,ax=plt.subplots(figsize=(10,8));y=np.arange(len(c))
    ax.barh(y-.18,c.retained_goa_percent,height=.34,color='#b45335',label='GoA=True')
    ax.barh(y+.18,c.retained_non_goa_percent,height=.34,color='#537c99',label='GoA=False')
    ax.set_yticks(y,coverage_order);ax.set(xlim=(0,100),xlabel='Сохранено от исходной группы, %',title='Охват после отбора C40 и решений 04–06');ax.legend()
    for i,row in enumerate(c.itertuples()):
        if row.rows==0:ax.text(1,i,'0 записей после отбора',va='center',fontsize=9)
    save(fig,'07_05_selection_coverage','Знаменатель: исходный европейский Parquet. График не показывает полноту реального воздушного движения.')

    rows=missing.copy();labels=['Код типа','Исходная классификация','Классификация после 05','Сегмент','Регистрация','AC_CLASS','AP_C_RWY']
    fields=['typecode','icaoaircrafttype','classification_unavailable','market_segment','registration','AC_CLASS','AP_C_RWY']
    fig,axes=plt.subplots(1,2,figsize=(14,8),sharey=True)
    for ax,label in zip(axes,[False,True]):
        d=rows[rows.GoA.eq(label)].set_index('airport').reindex(order)
        matrix=np.column_stack([100*d[field]/d.rows for field in fields])
        im=ax.imshow(np.ma.masked_invalid(matrix),aspect='auto',vmin=0,vmax=100,cmap='YlOrBr')
        ax.set_xticks(range(len(fields)),labels,rotation=45,ha='right');ax.set_yticks(range(len(order)),order);ax.set_title(f'GoA={label}')
    fig.suptitle('Пропуски справочных полей по аэропортам',fontweight='bold')
    fig.subplots_adjust(left=.07,right=.87,bottom=.30,top=.89,wspace=.16)
    color_axis=fig.add_axes([.90,.30,.015,.59]);fig.colorbar(im,cax=color_axis,label='Пропуски, %')
    save(fig,'07_06_missing_by_label','Пропуск = null или точная пустая строка. Производная классификация дополняет исходную.',layout=False)

    fig,axes=plt.subplots(5,4,figsize=(15,14),sharex=True)
    for ax,airport in zip(axes.flat,order):
        d=wilson(m[m.airport.eq(airport)]).set_index('month').loc[months]
        x=np.arange(len(months));ax.plot(x,d.ga_percent,color='#245b78',linewidth=1.3)
        ax.fill_between(x,d.wilson_lower_percent,d.wilson_upper_percent,color='#537c99',alpha=.2)
        if airport=='LFPO':
            # Два малых месячных знаменателя дают широкие интервалы: сохраняем их полностью.
            ax.set_yscale('symlog',linthresh=1)
            ax.text(.03,.95,'LFPO: шкала symlog; до 1% линейная',transform=ax.transAxes,fontsize=7,va='top')
        ax.set_title(airport,fontsize=11);ax.set_xticks(list(range(0,len(months),12)),[months[i][:4] for i in range(0,len(months),12)])
        ax.set_ylabel('GoA, %');ax.grid(alpha=.15)
    for ax in list(axes.flat)[len(order):]:ax.set_visible(False)
    fig.suptitle('Месячная динамика каждого аэропорта',fontweight='bold',y=.995)
    save(fig,'07_07_airport_time_series','Линия: доля GoA; полоса: 95% Уилсон. Шкалы Y различаются; LFPO: symlog из-за n=2 и 11 в апреле–мае 2020. Разрыв: нет данных.')


def main():
    REPORT.mkdir(parents=True,exist_ok=True);FIGURES.mkdir(parents=True,exist_ok=True)
    expected=json.loads((ROOT/'europe/reports/attempts/06_05_exclusion_applied.json').read_text(encoding='utf8'))
    source_expected=json.loads((ROOT/'europe/reports/profile_summary.json').read_text(encoding='utf8'))
    before=digest(INPUT);source_before=digest(SOURCE)
    assert before==expected['output_sha256'] and source_before==source_expected['source_sha256']
    print('Сканирование актуального входа: обе метки',flush=True)
    month,day,missing,categories,aircraft=scan(INPUT,True)
    print('Сканирование исходного входа для оценки охвата',flush=True)
    smonth,sday,_,_,_=scan(SOURCE)
    airport=wilson(month.groupby('airport')[['rows','go_arounds']].sum().reset_index())
    assert airport.rows.sum()==expected['rows_after'] and airport.go_arounds.sum()==expected['go_arounds_after']
    assert smonth.rows.sum()==source_expected['rows']
    coverage=airport.merge(smonth.groupby('airport')[['rows','go_arounds']].sum().reset_index().rename(
        columns={'rows':'source_rows','go_arounds':'source_go_arounds'}),on='airport',how='outer',validate='one_to_one')
    for field in ['rows','go_arounds','goa_false']:
        coverage[field]=coverage[field].fillna(0).astype('int64')
    coverage['retained_goa_percent']=100*coverage.go_arounds/coverage.source_go_arounds.replace(0,np.nan)
    coverage['retained_non_goa_percent']=100*coverage.goa_false/(coverage.source_rows-coverage.source_go_arounds).replace(0,np.nan)
    coverage['retained_rows_percent']=100*coverage.rows/coverage.source_rows
    coverage['source_ga_percent']=100*coverage.source_go_arounds/coverage.source_rows
    dates=pd.date_range(min(sday.date),max(sday.date),freq='D').strftime('%Y-%m-%d')
    full=pd.MultiIndex.from_product([sorted(coverage.airport),dates],names=['airport','date'])
    calendar=day.set_index(['airport','date']).reindex(full,fill_value=0).reset_index()
    source_calendar=sday.set_index(['airport','date']).reindex(full,fill_value=0).reset_index()
    gap=calendar.assign(without_records=calendar.rows.eq(0)).groupby('airport').agg(
        days_with_records=('rows',lambda x:int(x.gt(0).sum())),days_without_records=('without_records','sum'))
    source_gap=source_calendar.assign(without_records=source_calendar.rows.eq(0)).groupby('airport').agg(
        source_days_without_records=('without_records','sum'))
    airport=airport.merge(gap,on='airport').merge(source_gap,on='airport')
    unique=Counter(key[0] for key in aircraft)
    airport['unique_icao24']=airport.airport.map(unique)
    airport['active_months']=airport.airport.map(month.groupby('airport').month.nunique())
    airport=airport.sort_values('rows',ascending=False)
    shared=calendar.assign(empty=calendar.rows.eq(0)).groupby('date').agg(
        airports_without_records=('empty','sum'),rows=('rows','sum'),go_arounds=('go_arounds','sum')).reset_index()
    shared['source_rows']=shared.date.map(source_calendar.groupby('date').rows.sum())
    # Отдельный описательный список: не правила исключения и не тест безопасности.
    monthly=wilson(month)
    shortlist=monthly.sort_values('ga_percent',ascending=False).head(15)
    for table,name in [(airport,'airport_summary'),(monthly,'airport_month'),(wilson(smonth),'source_airport_month'),
        (coverage,'selection_coverage'),(missing,'missing_by_label'),(categories,'categories_by_label'),
        (calendar,'airport_calendar'),(source_calendar,'source_airport_calendar'),(shared,'shared_daily_gaps'),(shortlist,'highest_monthly_rates')]:
        table.to_csv(REPORT/f'{name}.csv',index=False)
    plots(airport,monthly,coverage,missing)
    assert digest(INPUT)==before and digest(SOURCE)==source_before
    audit=dict(checked_date='2026-10-09',generated_utc=datetime.now(timezone.utc).isoformat(),
        input=str(INPUT.relative_to(ROOT)),input_sha256=before,source=str(SOURCE.relative_to(ROOT)),
        source_sha256=source_before,rows=int(airport.rows.sum()),go_arounds=int(airport.go_arounds.sum()),
        goa_false=int(airport.goa_false.sum()),airports=len(airport),source_airports=len(coverage),
        airports_without_retained_records=coverage.loc[coverage.rows.eq(0),'airport'].tolist(),source_rows=int(smonth.rows.sum()),
        first_date=min(sday.date),last_date=max(sday.date),input_unchanged=True,source_unchanged=True,
        rows_deleted=0,values_changed=0,labels_audited=[False,True],wilson_z=Z,
        confidence_level=.95,intervals_pointwise=True,independence_not_verified=True,
        automatic_airport_exclusion=False,old_precision_cutoff_applied=False,
        figures=sorted(p.name for p in FIGURES.glob('*.png')),
        method_source='https://www.itl.nist.gov/div898/handbook/prc/section2/prc241.htm')
    (REPORT/'07_01_audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    text='# 07_01 – Аэропорты: полный описательный аудит\n\n'
    text+=f'Проверены {audit["rows"]:,} строк: {audit["go_arounds"]:,} GoA=True и {audit["goa_false"]:,} GoA=False; {audit["airports"]} аэропортов. Обе метки охвачены полностью. Вход и исходник проверены по SHA256 до и после; изменения и исключения отсутствуют.\n\n'
    text+='## Сводка по аэропортам\n\n| Аэропорт | n | k GoA | Доля, % | 95% Уилсон, % | Активных месяцев | Дней без записей | Сохранено исходных строк, % |\n| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: |\n'
    for row in airport.itertuples():
        retained=coverage.set_index('airport').loc[row.airport,'retained_rows_percent']
        text+=f'| {row.airport} | {row.rows:,} | {row.go_arounds:,} | {row.ga_percent:.3f} | {row.wilson_lower_percent:.3f}–{row.wilson_upper_percent:.3f} | {row.active_months} | {row.days_without_records} | {retained:.1f} |\n'
    text+='\nВ исходнике '+str(len(coverage))+' аэропортов. Без сохраненных наблюдений после отбора: '+', '.join(audit['airports_without_retained_records'])+'. Нулевой охват не означает нулевую частоту GoA; частота пустой группы не вычисляется.\n'
    text+='\n## Интерпретация и дальнейший разбор\n\n'
    top=airport.sort_values('ga_percent',ascending=False).head(3)
    text+='Самые высокие наблюдаемые доли: '+', '.join(f'{row.airport} ({row.ga_percent:.3f}%, k={row.go_arounds}, n={row.rows})' for row in top.itertuples())+'. Это описательная характеристика разметки, а не рейтинг безопасности.\n\n'
    low=coverage.sort_values('retained_rows_percent').head(3)
    text+='Наименьший охват исходных строк: '+', '.join(f'{row.airport} ({row.retained_rows_percent:.1f}%)' for row in low.itertuples())+'. Дальше нужно разделить влияние отбора по C40, исходного покрытия и состава полетов.\n\n'
    text+='`highest_monthly_rates.csv` содержит 15 месяцев с наибольшей долей, с n, k и интервалами; это ранжирование без автоматического порога удаления. Неполные крайние годы и общие провалы покрытия проверяются по календарным таблицам, а не трактуются как изменение авиационного риска.\n\n'
    text+='## Метод и ограничения\n\nЕдиница учета – сохраненная запись посадочного сегмента, не гарантированно уникальный физический полет. Интервал Уилсона 95% рассчитан по [формуле NIST](https://www.itl.nist.gov/div898/handbook/prc/section2/prc241.htm). Интервалы точечные, не одновременные; зависимость наблюдений одного борта и календарных групп не устранена. Они описывают биномиальную неопределенность при условных предположениях, а не погрешность детектора и не репрезентативность реального движения. Архивный допуск 0,2 п.п. не применен как правило к новой Европе.\n\nGoA=False означает отсутствие отмеченного ухода, а не независимое подтверждение нормального профиля. Реальное число операций аэропорта неизвестно; охват измеряется относительно исходного Parquet. Изменение доли между исходником и выборкой не устанавливает причину. Типы и сегменты описаны отдельно по обеим меткам; пропуски не заполнены. Производная классификация дополняет исходную только там, где выполнены правила этапа 05.\n\nLSGG остается с принятой оговоркой; план чувствительности из этапа 06 сохраняется. Никакие новые аэропорты и строки не исключены. Углубленный разбор отдельных аэропортов – следующий шаг 07_02, после обсуждения обнаруженных групп. Подготовка идентификатора и окончательный допуск признаков идут после аэропортового разбора.\n'
    text+='\nДля LFPO на рисунке месячной динамики используется явно обозначенная шкала symlog (линейная до 1%); она сохраняет полные широкие интервалы малых месячных групп. Малый знаменатель не доказывает изменение риска.\n'
    (REPORT/'07_01_review.md').write_text(text,encoding='utf8')
    print(airport[['airport','rows','go_arounds','ga_percent','days_without_records']].to_string(index=False),flush=True)
    print(json.dumps(audit,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':
    if hasattr(sys.stdout,'reconfigure'):sys.stdout.reconfigure(encoding='utf8')
    main()
