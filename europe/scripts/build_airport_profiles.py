"""Карточки всех аэропортов на проверенных агрегатах 07_01."""
import json
import pandas as pd
from audit_feature_availability import ROOT, digest


def main():
    report=ROOT/'europe/reports/airports'
    audit=json.loads((report/'07_01_audit.json').read_text(encoding='utf8'))
    assert digest(ROOT/audit['input'])==audit['input_sha256']
    airports=pd.read_csv(report/'airport_summary.csv').set_index('airport')
    coverage=pd.read_csv(report/'selection_coverage.csv').set_index('airport')
    missing=pd.read_csv(report/'missing_by_label.csv')
    categories=pd.read_csv(report/'categories_by_label.csv')
    months=pd.read_csv(report/'airport_month.csv')
    calendar=pd.read_csv(report/'airport_calendar.csv')
    gaps=[]
    for airport,frame in calendar.groupby('airport'):
        frame=frame.sort_values('date');empty=frame.rows.eq(0)
        block=empty.ne(empty.shift()).cumsum()
        for _,part in frame[empty].groupby(block[empty]):
            gaps.append(dict(airport=airport,first_date=part.date.iloc[0],last_date=part.date.iloc[-1],days=len(part)))
    gap_table=pd.DataFrame(gaps).sort_values(['days','airport'],ascending=[False,True])
    gap_table.to_csv(report/'continuous_gaps.csv',index=False)
    text='# Карточки аэропортов – 07_01\n\nВсе 20 аэропортов исходника рассмотрены, включая LTFJ без сохраненных строк. Период календаря общий: 01.01.2019–30.05.2023. Дни без записей не доказывают отсутствие движения. Таблицы состава показывают наиболее частые категории, доли рассчитаны внутри соответствующей метки; это не скорректированное сравнение риска. Новые исключения не применены.\n'
    for airport,c in coverage.iterrows():
        text+=f'\n## {airport}\n\nИсходник: n={int(c.source_rows):,}, k={int(c.source_go_arounds):,}. Сохранено {c.retained_rows_percent:.1f}% строк; GoA=True – {c.retained_goa_percent:.1f}%, GoA=False – {c.retained_non_goa_percent:.1f}%.\n'
        if airport not in airports.index:
            text+='\nВ аналитической выборке нет наблюдений; доля GoA не вычисляется. Это последствие принятого правила C40. В отчете 04_03 все строки LTFJ относятся к группе missing. Дополнительное исключение 07 не требуется.\n'
            continue
        a=airports.loc[airport]
        text+=f'\nТекущий состав: n={int(a.rows):,}, k={int(a.go_arounds):,}; доля {a.ga_percent:.3f}%, 95% Уилсон {a.wilson_lower_percent:.3f}–{a.wilson_upper_percent:.3f}%. Активных месяцев: {int(a.active_months)}; дней без записей: {int(a.days_without_records)} против {int(a.source_days_without_records)} в исходнике. ICAO24-адресов: {int(a.unique_icao24):,}; адрес не гарантирует неизменную идентичность физического борта.\n'
        g=gap_table[gap_table.airport.eq(airport)].head(3)
        text+='\nТри наиболее длинных непрерывных провала: '+('; '.join(f'{row.first_date}–{row.last_date} ({row.days} дней)' for row in g.itertuples()) or 'нет')+'.\n'
        m=months[months.airport.eq(airport)].sort_values('ga_percent',ascending=False).iloc[0]
        text+=f'\nМаксимальная месячная доля: {m.month}, {m.ga_percent:.3f}%, k={int(m.go_arounds)}, n={int(m.rows)}; точечный интервал {m.wilson_lower_percent:.3f}–{m.wilson_upper_percent:.3f}%. Месяц выбран после просмотра, подтверждающий тест не выполнен.\n'
        text+='\n| Метка | Число строк | Пропуск typecode, % | Классификация недоступна после 05, % | Пропуск сегмента, % |\n| --- | ---: | ---: | ---: | ---: |\n'
        for row in missing[missing.airport.eq(airport)].itertuples():
            text+=f'| GoA={row.GoA} | {row.rows} | {100*row.typecode/row.rows:.2f} | {100*row.classification_unavailable/row.rows:.2f} | {100*row.market_segment/row.rows:.2f} |\n'
        text+='\n| Метка | Поле | Самые частые категории: число (доля внутри метки) |\n| --- | --- | --- |\n'
        for label in [False,True]:
            denominator=int(a.go_arounds if label else a.goa_false)
            for field in ['typecode','market_segment']:
                group=categories[categories.airport.eq(airport)&categories.GoA.eq(label)&categories.field.eq(field)].sort_values('rows',ascending=False).head(5)
                values='; '.join(f'{row.value}: {row.rows} ({100*row.rows/denominator:.1f}%)' for row in group.itertuples())
                text+=f'| GoA={label} | {field} | {values} |\n'
        if airport=='LSGG':text+='\nСтатус LSGG: сохранен с оговоркой, проверка чувствительности при моделировании. Новая карточка не отменяет решение 06.\n'
    (report/'airport_profiles.md').write_text(text,encoding='utf8')
    print('Карточки 20 аэропортов и непрерывные провалы покрытия сохранены.')


if __name__=='__main__':main()
