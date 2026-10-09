"""Проверка артефактов и краткий разбор конкретных записей 07_03."""
import json
import pandas as pd
import nbformat
from audit_feature_availability import ROOT


def main():
    report=ROOT/'europe/reports/airports/EGCC'
    records=pd.read_csv(report/'07_03_positive_records.csv',dtype={'icao24':str})
    events=pd.read_csv(report/'07_03_weather_at_events.csv')
    cross=pd.read_csv(report/'07_03_other_airports_daily.csv')
    daily=pd.read_csv(report/'07_03_daily.csv')
    assert len(records)==45 and records.flight_id.nunique()==45
    assert daily.rows.sum()==4301 and daily.go_arounds.sum()==45
    assert events.metar_utc.notna().all() and events.metar_age_minutes.between(0,60).all()
    assert events.event_utc.ge(events.metar_utc).all()
    # Сопоставление по всем дням с нулями; максимум выбирается описательно.
    maxima=cross.loc[cross.groupby('airport').go_arounds.idxmax()].sort_values('go_arounds',ascending=False)
    maxima.to_csv(report/'07_03_other_airport_maxima.csv',index=False)
    intervals=[]
    for row in records.itertuples():
        times=pd.to_datetime(json.loads(row.attempts_json),utc=True)
        for i in range(1,len(times)):
            intervals.append(dict(flight_id=row.flight_id,icao24=row.icao24,date=row.date,
                from_attempt=i,to_attempt=i+1,minutes=(times[i]-times[i-1]).total_seconds()/60))
    gaps=pd.DataFrame(intervals);gaps.to_csv(report/'07_03_attempt_intervals.csv',index=False)
    assert gaps.minutes.gt(0).all()
    text='\n## Проверка конкретных записей и погодного контекста\n\n'
    text+=f'Записи с n_attempts: {records.n_attempts.value_counts().sort_index().to_dict()}. Для каждого интервала сохранены исходный flight_id и длительность. Минимальный интервал между попытками: {gaps.minutes.min():.2f} мин; максимальный: {gaps.minutes.max():.2f} мин. Короткий интервал сам по себе не является правилом удаления.\n\n'
    text+='Самые короткие интервалы – первоочередные записи для просмотра:\n\n| flight_id | Борт | День посадки | Интервал, мин |\n| --- | --- | --- | ---: |\n'
    for r in gaps.sort_values('minutes').head(5).itertuples():text+=f'| {r.flight_id} | {r.icao24} | {r.date} | {r.minutes:.2f} |\n'
    text+='\n[Погодный разбор](07_03_weather_review.md): 1391 архивных сообщений; для всех попыток и посадок найден предшествующий METAR не старше часа.\n\n'
    for day in ['2020-02-10','2020-02-11']:
        d=events[events.event.str.startswith('attempt_') & events.event_utc.str.startswith(day)]
        text+=f'{day}, METAR у отмеченных попыток: ветер {d.wind_kt.min():.0f}–{d.wind_kt.max():.0f} kt; известные порывы {d.gust_kt.min():.0f}–{d.gust_kt.max():.0f} kt. Пропуск порыва не означает его нулевое значение.\n\n'
    text+='## Совпадение дней между аэропортами\n\n| Аэропорт | День максимального числа GoA в феврале | n | GoA | Доля, % |\n| --- | --- | ---: | ---: | ---: |\n'
    for r in maxima.head(10).itertuples():text+=f'| {r.airport} | {r.date} | {r.rows} | {r.go_arounds} | {r.ga_percent:.3f} |\n'
    text+='\nПри равенстве выбран первый день. Максимумы числа GoA не являются максимумами вероятности. Полная дневная таблица содержит обе метки и исходный охват; совпадение календарных пиков не доказывает общей причины.\n\n'
    text+='Всплеск сосредоточен в отдельных днях, присутствует и до отбора C40 и охватывает несколько бортов и типов. Ветер у попыток дает проверяемый погодный контекст. Причина каждого ухода и корректность детектора остаются открытыми без независимых траекторий. Новые исключения не применены.\n'
    path=report/'07_03_review.md';base=path.read_text(encoding='utf8').split('\n## Проверка конкретных записей и погодного контекста')[0]
    path.write_text(base+text,encoding='utf8')
    nbpath=ROOT/'europe/notebooks/07_airports/EGCC/07_03_egcc_february.ipynb'
    nb=nbformat.read(nbpath,as_version=4);nb.cells[0].source=base+text;nbformat.validate(nb);nbformat.write(nb,nbpath)
    print(text)


if __name__=='__main__':main()
