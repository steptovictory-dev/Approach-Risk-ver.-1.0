"""Сводка METAR и оформление решения CFL12 в ноутбуке 06_04."""
import json
import re
import nbformat
import pandas as pd
from audit_feature_availability import ROOT, digest

REPORT=ROOT/"europe/reports/attempts"


def wind(raw):
    # Первый основной ветер; группы тренда не подменяют наблюдение
    match=re.search(r"(?:^|\s)(\d{3}|VRB)(\d{2,3})(?:G(\d{2,3}))?KT(?:\s|$)",raw)
    assert match,raw
    return dict(wind_direction_group=match[1],wind_speed_kt=int(match[2]),wind_gust_kt=int(match[3]) if match[3] else None)


def main():
    manifest=json.loads((REPORT/"06_04_weather_manifest.json").read_text(encoding="utf8"))
    events=pd.read_csv(REPORT/"06_04_weather_at_attempts.csv",dtype=str)
    summaries=[]
    for source in manifest["cases"]:
        path=ROOT/source["file"];assert digest(path)==source["sha256"]
        daily=pd.read_csv(path,comment="#",dtype=str,keep_default_na=False)
        daily_wind=pd.DataFrame([wind(raw) for raw in daily.metar])
        part=events[events.flight_id.eq(source["flight_id"])]
        used=part.drop_duplicates(["metar_utc","metar"])
        during=pd.DataFrame([wind(raw) for raw in used.metar])
        summaries.append(dict(flight_id=source["flight_id"],airport=source["airport"],date_utc=source["date_utc"],daily_reports=len(daily),daily_max_wind_speed_kt=int(daily_wind.wind_speed_kt.max()),daily_max_reported_gust_kt=daily_wind.wind_gust_kt.max(),event_metars=len(used),event_wind_speed_min_kt=int(during.wind_speed_kt.min()),event_wind_speed_max_kt=int(during.wind_speed_kt.max()),event_max_reported_gust_kt=during.wind_gust_kt.max(),event_max_metar_age_minutes=float(pd.to_numeric(part.age_minutes).max()),all_event_reports_CAVOK=bool(used.metar.str.contains(r"\bCAVOK\b",regex=True).all())))
    summary=pd.DataFrame(summaries);summary.to_csv(REPORT/"06_04_weather_summary.csv",index=False)
    text="""# 06_04 – Исключение CFL12 и погода остальных шести случаев

Проверка 09.10.2026. CFL12_27984, ICAO24 4ab179, ESSA, landing_time 12.11.2022 10:07:17 UTC исключен по прямому решению пользователя как нетипичный для исследовательской популяции. Историческая задача полета независимо не установлена. Полная строка сохранена отдельно; GoA не переписывается.

Актуальная копия – `europe/data/analytical/06_04_cfl12_excluded.parquet`: 6 331 492 строки, 18 356 GoA=True. Проверены схема, порядок и все значения оставшихся строк, а также неизменность входа по SHA256. Теперь больше двух попыток у 147 записей: 141 с тремя, пять с четырьмя, одна с пятью. Журнал – `06_04_cfl12_exclusion_applied.json`.

## Источник и привязка времени

Из [Iowa Environmental Mesonet](https://mesonet.agron.iastate.edu/request/download.phtml) получены архивные METAR за полные UTC-дни каждого из шести случаев. Это вторичный архив переданных сообщений, а не прямое получение архивной справки национальной метеослужбы. Сам IEM указывает на ограниченный контроль качества. Исходные ответы сохранены целиком, URL и SHA256 – в `06_04_weather_manifest.json`.

Для каждой попытки и landing_time выбран последний METAR не позже события. Между сообщениями погода не интерполируется. Максимальный возраст выбранного сообщения – 28,1 минуты; фактическое состояние в конкретную секунду остается неизвестным. Станция и день/время проверены по тексту каждого METAR. EDDB использован как код станции из исторического сообщения 2020 года; современное название аэропорта не подменяет исторический идентификатор.

Отдельно сохранено окно от часа до первой попытки до часа после landing_time. Позднейшее сообщение не приписывается более раннему событию. Полные UTC-дни нужны для контекста; суточный максимум не считается ветром при попытке. Прогнозные группы TEMPO/BECMG отделяются от наблюдаемой части сообщения.

## Условия непосредственно у попыток

| Случай и дата UTC | Ветер в предшествующих METAR | Видимость и облака / CAVOK | Вывод |
| --- | --- | --- | --- |
| KLM1386, EHAM, 09.02.2020 | 200°, 36–41 kt, порывы 48–60 kt | 9999, затем 8000 м; BKN019; морось в 15:25 | Сильный порывистый ветер – существенный погодный контекст; причина каждого ухода не доказана |
| KLM28Q, EHAM, 31.01.2022 | 310°, 34 kt, порывы 44–46 kt | 8000 м, дождь; BKN032/BKN031 | Аналогично: порывистый ветер подтвержден сообщениями |
| AOJ92R, EDDB, 22.04.2020 | 060–100°, 10–12 kt | CAVOK | В использованных METAR нет сильного ветра или сообщения об ухудшении видимости; четыре отметки не объяснены |
| ENF04, LIMC, 26.02.2021 | VRB, 2–3 kt | CAVOK | Погодная гипотеза по METAR не поддерживается; специальное назначение остается гипотезой |
| FHXRG, LSGG, 20.03.2021 | 020–030°, 13–14 kt | CAVOK | METAR не показывает разворота ветра, объясняющего последовательность 04 → 22 → 04 → 04 |
| ECKJQ, LEMD, 02.07.2021 | 210–240°, 12–15 kt; температура 34–35°C | CAVOK | Высокая температура отражена; признаки сильного ветра/осадков в использованных сообщениях не найдены. Причина четырех отметок неизвестна |

kt – узлы. BKN019/031/032 – исходные коды слоя с высотой 1 900/3 100/3 200 ft; эти значения не сравниваются с минимумами конкретного захода. 9999 сохранено как код сообщения, а не точное измерение 9 999 метров. CAVOK не доказывает отсутствие всех локальных опасных явлений, сдвига ветра или турбулентности. Ограничения самолета, фактическая ВПП и минимумы не проверялись; превышение эксплуатационного ограничения не заявляется.

### Два важных ограничения по времени

- EHAM 09.02.2020 15:25: `20041G60KT`, видимость 8000, `-DZ BKN019`; `TEMPO 4000` относится к тренду, а не доказывает наблюдаемую видимость 4000 при посадке.
- LSGG порыв `G24KT` есть в сообщении 21:20 UTC, через 55 минут 37 секунд после landing_time 20:24:23. В METAR, выбранных для попыток, порывы не сообщены. Поздний порыв не используется для объяснения ранних попыток.

## Решение по оставшимся строкам

Остальные шесть строк сохраняются. Для двух KLM получен погодный контекст, совместимый с возможными повторными заходами, но GoA и количество попыток не считаются независимо подтвержденными. Для AOJ92R/FHXRG погода по сообщениям не объяснила последовательности. Для ENF04/ECKJQ проверка погоды не установила причину и не заменила проверку назначения полета. Других исключений пользователь не запрашивал.

Погода используется для аудита после события, не присоединена как признак модели. Для раннего прогноза потребуются отдельные правила времени выпуска/доступности METAR; observation timestamp сам по себе не доказывает доступность сообщения в ту же секунду.

## Артефакты

`06_04_weather_summary.csv` – сутки и условия по выбранным сообщениям; `06_04_weather_at_attempts.csv` – привязка каждой попытки и посадки к METAR; `06_04_weather_windows.csv` – расширенные окна; `06_04_weather_manifest.json` – происхождение и контрольные суммы. Все исходные поля и единицы описаны в `06_04_weather_dictionary.md`. Исторический семистрочный отчет и диагностическая копия сохранены как состояние до исключения CFL12.
"""
    (REPORT/"06_04_weather_review.md").write_text(text,encoding="utf8")
    nbpath=ROOT/"europe/notebooks/06_attempts/multiple_attempts/06_04_high_attempt_cases.ipynb"
    nb=nbformat.read(nbpath,as_version=4)
    marker="## Решение по CFL12 и исторические METAR – 09.10.2026"
    nb.cells=[cell for cell in nb.cells if not(cell.cell_type=="markdown" and cell.source.startswith(marker)) and cell.get("metadata",{}).get("weather_06_04") is not True]
    nb.cells.append(nbformat.v4.new_markdown_cell(marker+"\n\nПервоначальный разбор семи строк выше сохранен как история. CFL12 исключен; остальные шесть сохраняются. Далее – погодная проверка."))
    cell=nbformat.v4.new_code_cell('import pandas as pd\ndisplay(pd.read_csv(ROOT / "europe/reports/attempts/06_04_weather_summary.csv"))\ndisplay(pd.read_csv(ROOT / "europe/reports/attempts/06_04_weather_at_attempts.csv"))\ndisplay(Markdown((ROOT / "europe/reports/attempts/06_04_weather_review.md").read_text(encoding="utf8")))');cell.metadata["weather_06_04"]=True;nb.cells.append(cell)
    nbformat.validate(nb);nbformat.write(nb,nbpath)
    print(summary.to_string(index=False))


if __name__=="__main__":main()
