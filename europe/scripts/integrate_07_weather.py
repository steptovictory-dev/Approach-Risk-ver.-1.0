"""Интегрирует межаэропортовый погодный разбор в общий ноутбук папки 07."""
import nbformat
import pandas as pd
from audit_feature_availability import ROOT


def main():
    folder=ROOT/'europe/notebooks/07_airports'
    report=ROOT/'europe/reports/airports/february_2020_weather'
    data=pd.read_csv(report/'07_03_weather_comparison.csv',keep_default_na=False)
    headers=['Дата UTC','Аэропорт','GoA / все записи','GoA, %','Ветер, kt','Порывы, kt','Видимость, км','Явления METAR','Сопоставлено / попытки']
    table='| '+' | '.join(headers)+' |\n| '+' | '.join(['---']*len(headers))+' |\n'
    for r in data.itertuples():
        table+='| '+' | '.join([r.date,r.airport,f'{r.go_arounds} / {r.rows}',f'{r.ga_percent:.3f}',r.wind_kt,r.gust_kt,r.visibility_km,r.phenomena,f'{r.matched} / {r.attempts}'])+' |\n'
    intro='''# 07_04 – Погода в дни всплесков GoA: общий разбор

Рабочая точка для просмотра погодной таблицы, обсуждения результатов и принятия решения в папке **07_airports**. Предыдущий локальный разбор: [07_03 – EGCC](EGCC/07_03_egcc_february.ipynb).

Выбраны пять дней EGCC с наибольшим числом GoA и дни максимального числа GoA десяти аэропортов в феврале 2020. Выбор описательный, после просмотра данных. Знаменатели включают GoA=True и GoA=False.
'''
    method='''## Как читать таблицу

Дата относится к посадке UTC. Погода взята у предшествующих попыток ухода, включая переход суток: последний METAR не позже попытки, возраст не более 60 минут. Последняя попытка завершает сегмент и не включена в таблицу погодных условий у уходов. Одна запись GoA=True может содержать несколько отмеченных уходов; поэтому число попыток может превышать число записей.

Диапазоны объединяют сообщения у разных попыток, а не суточные экстремумы. Порывы показаны только там, где они переданы; «—» не означает нулевой порыв. Ветер приведен к kt, исходные единицы и сообщения сохранены. Видимость 9999/CAVOK показана как ≥10 км. METAR может содержать прогнозную часть; перечисление явлений из всего сообщения служит контекстом и не гарантирует их наблюдение непосредственно у попытки.

RA – дождь; DZ – морось; SHRA – ливневый дождь; SN – снег; SHSN – снежный ливень; TS – гроза; GS – снежная крупа или мелкий град. «−» – слабая интенсивность, «+» – сильная.
'''
    analysis='''## Анализ перед решением

1. EGCC: 10–11 февраля дают 18 из 45 GoA февраля, при 165 и 133 наблюдениях. Всплеск присутствует также в исходнике до отбора C40.
2. Дни крупных пиков других аэропортов группируются вокруг 9–10 февраля; у попыток встречаются сильный ветер и порывы. Это совместный календарный и погодный контекст, не доказательство причины каждого ухода.
3. В EGCC 24 февраля у попыток порывы не переданы, хотя суточный максимум порыва был выше. Нельзя заменять условия у попытки суточным максимумом.
4. Таблица охватывает выбранные положительные события. Для оценки связи погоды с GoA нужно сопоставить погоду также с GoA=False и контролировать состав, время и покрытие. Эта таблица сама по себе не оценивает погодный эффект.
5. Точные повторы ключей EGCC не обнаружены; независимая проверка траекторий и корректности детектора еще не выполнена.

Обсуждаем: достаточен ли погодный контекст для завершения описательного разбора, какие конкретные записи требуют проверки и нужен ли отдельный анализ условий у обоих классов.
'''
    decision='''## Решение

**Статус: ожидает обсуждения с пользователем.** Новое решение по погодному разбору не принято. Новые исключения и изменения данных не применены. Прежнее решение сохранить доступный период EGCC остается действующим.

После обсуждения здесь фиксируются дата, принятое решение, основания и оставшиеся ограничения. Высокая доля GoA, погода или необычный график сами по себе не являются основанием исключать аэропорт.
'''
    code="""from pathlib import Path
import pandas as pd

root = next(p for p in [Path.cwd(), *Path.cwd().parents]
            if (p / 'europe/reports/airports/february_2020_weather/07_03_weather_comparison.csv').exists())
report = root / 'europe/reports/airports/february_2020_weather'
weather_table = pd.read_csv(report / '07_03_weather_comparison.csv', keep_default_na=False)
weather_table
"""
    details="""# Конкретные попытки и исходные METAR для проверки выбранной группы.
events = pd.read_csv(report / '07_03_weather_comparison_events.csv', dtype={'icao24': str})
airport = 'EGCC'
landing_date = '2020-02-10'
events.loc[(events.airport == airport) & (events.landing_date == landing_date),
           ['flight_id', 'icao24', 'attempt', 'event_utc', 'metar_utc',
            'age_minutes', 'wind_kt', 'gust_kt', 'metar']]
"""
    sources='''## Данные и воспроизведение

Таблица встроена выше; открывать другие папки для обсуждения не требуется. Служебные CSV и журнал происхождения остаются в `europe/reports/airports/february_2020_weather`, исходные сообщения – в `docs/sources/files/07_03_weather`. Источник METAR: вторичный архив Iowa Environmental Mesonet. [Журнал запросов и SHA256](../../reports/airports/february_2020_weather/07_03_weather_comparison_manifest.json).

Из корня проекта: `.\\.venv\\Scripts\\python.exe -u -X utf8 europe/scripts/build_07_03_weather_table.py`, затем `.\\.venv\\Scripts\\python.exe -u -X utf8 europe/scripts/integrate_07_weather.py`. Скрипт интеграции не перезаписывает существующий ноутбук, чтобы сохранить обсуждение и принятое решение.
'''
    target=folder/'07_04_weather_comparison.ipynb'
    nb=nbformat.v4.new_notebook(cells=[nbformat.v4.new_markdown_cell(intro),nbformat.v4.new_markdown_cell(table),nbformat.v4.new_markdown_cell(method),nbformat.v4.new_markdown_cell(analysis),nbformat.v4.new_markdown_cell(decision),nbformat.v4.new_code_cell(code),nbformat.v4.new_code_cell(details),nbformat.v4.new_markdown_cell(sources)],metadata={'kernelspec':{'name':'python3','display_name':'Python 3','language':'python'}})
    nbformat.validate(nb)
    if not target.exists():nbformat.write(nb,target)
    readme=folder/'README.md';text=readme.read_text(encoding='utf8')
    text=text.replace('- [Погода в дни всплесков нескольких аэропортов](../../reports/airports/february_2020_weather/07_03_weather_table.html) – общая таблица, выше локальной папки EGCC.', '- [07_04 – Погода в дни всплесков: таблица, анализ и решение](07_04_weather_comparison.ipynb) – общий рабочий ноутбук папки 07.')
    readme.write_text(text,encoding='utf8')
    local=folder/'EGCC/README.md';text=local.read_text(encoding='utf8')
    text=text.replace('Межаэропортовая [таблица погоды](../../../reports/airports/february_2020_weather/07_03_weather_table.html) размещена на общем уровне раздела 07.', 'Межаэропортовый [07_04 – погодный разбор и решение](../07_04_weather_comparison.ipynb) находится на общем уровне папки 07.')
    local.write_text(text,encoding='utf8')
    nbformat.validate(nbformat.read(target,as_version=4))
    assert len(data)==14 and data.matched.sum()==180
    print(target)


if __name__=='__main__':main()
