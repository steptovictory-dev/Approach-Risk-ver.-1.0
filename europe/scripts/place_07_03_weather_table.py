"""Размещает межаэропортовую таблицу на общем уровне и проверяет навигацию."""
from pathlib import Path
from audit_feature_availability import ROOT


def main():
    base=ROOT/'europe/reports/airports'
    target=base/'february_2020_weather';target.mkdir(exist_ok=True)
    names=['07_03_weather_comparison.csv','07_03_weather_comparison_events.csv',
           '07_03_weather_comparison_manifest.json','07_03_weather_table.md','07_03_weather_table.html']
    for name in names:
        source=base/'EGCC'/name;destination=target/name
        assert source.resolve().is_relative_to(ROOT.resolve()) and destination.resolve().is_relative_to(ROOT.resolve())
        if source.exists():
            assert not destination.exists()
            source.rename(destination)
    links={base/'README.md':'\n\n## Погода в дни всплесков февраля 2020\n\nМежаэропортовое сравнение размещено на общем уровне: [оформленная таблица](february_2020_weather/07_03_weather_table.html), [таблица Markdown](february_2020_weather/07_03_weather_table.md). Сырые сообщения, сопоставление каждой попытки и журнал происхождения сохранены. Локальный разбор EGCC остается в `EGCC/`.\n',
        ROOT/'europe/notebooks/07_airports/README.md':'\n\n- [07_03 – EGCC: всплеск февраля 2020](EGCC/07_03_egcc_february.ipynb).\n- [Погода в дни всплесков нескольких аэропортов](../../reports/airports/february_2020_weather/07_03_weather_table.html) – общая таблица, выше локальной папки EGCC.\n',
        ROOT/'europe/notebooks/07_airports/EGCC/README.md':'\n\n[07_03 – Всплеск февраля 2020](07_03_egcc_february.ipynb). [Отчет](../../../reports/airports/EGCC/07_03_review.md). Межаэропортовая [таблица погоды](../../../reports/airports/february_2020_weather/07_03_weather_table.html) размещена на общем уровне раздела 07.\n'}
    for path,note in links.items():
        current=path.read_text(encoding='utf8')
        if note.strip() not in current:path.write_text(current+note,encoding='utf8')
    assert all((target/name).exists() for name in names)
    import pandas as pd
    rows=pd.read_csv(target/names[0]);events=pd.read_csv(target/names[1])
    assert len(rows)==14 and rows.matched.sum()==len(events) and events.metar_utc.notna().all()
    print(f'Проверено: {len(rows)} групп, {len(events)} попыток, все сопоставлены. Таблица: {target}')


if __name__=='__main__':main()
