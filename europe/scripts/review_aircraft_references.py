from pathlib import Path
import json
import hashlib
import pandas as pd

root = Path(__file__).resolve().parents[2]
references = root / 'docs/sources/files/aircraft_code_review'
reports = root / 'europe/reports/preparation'
official = pd.DataFrame(json.loads((references / 'icao_aircraft_types.json').read_text(encoding='utf8')))
values = pd.read_csv(reports / '02_aircraft_code_values.csv', keep_default_na=False)
rows = []
evidence = []
url = 'https://doc8643.icao.int/External/AircraftTypes'
for record in values.itertuples(index=False):
    key = record.icaoaircrafttype.upper()
    matches = official.loc[official['Designator'].eq(key)]
    descriptions = sorted(matches['Description'].dropna().unique())
    confirmed = len(descriptions) == 1 and bool(descriptions[0])
    rows.append({
        'source_icaoaircrafttype': record.icaoaircrafttype,
        'rows': int(record.rows), 'go_arounds': int(record.go_arounds),
        'lookup_designator': key,
        'reference_description': descriptions[0] if confirmed else '',
        'status': 'direct_unique_reference_match' if confirmed else 'unresolved',
        'proposed_action': 'derive_description_keep_raw' if confirmed else 'keep_raw_pending_investigation',
        'applied': False,
        'reference_url': url, 'reference_checked_date': '2026-10-08',
        'historical_validity_2019_2023': 'not_established',
    })
    for match in matches[['Designator', 'Description', 'ModelFullName', 'ManufacturerCode']].drop_duplicates().to_dict(orient='records'):
        evidence.append(match)
review = pd.DataFrame(rows)
assert len(review) == 23 and review['rows'].sum() == 6407 and review['go_arounds'].sum() == 29
review.to_csv(reports / '03_aircraft_code_proposals.csv', index=False)
pd.DataFrame(evidence).to_csv(reports / '03_aircraft_reference_matches.csv', index=False)
manifest_path = reports / 'aircraft_reference_sources.json'
manifest = json.loads(manifest_path.read_text(encoding='utf8'))
path = references / 'icao_aircraft_types.json'
manifest['sources'] = [s for s in manifest['sources'] if s['name'] != path.name]
manifest['sources'].append({'name': path.name, 'url': url, 'method': 'POST',
                            'bytes': path.stat().st_size,
                            'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                            'status': 'retrieved', 'reference_rows': len(official)})
manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf8')
table = review[['source_icaoaircrafttype', 'rows', 'go_arounds', 'reference_description', 'status']]
lines = ['| Значение источника | Записи | GoA=True | Описание ICAO | Статус |',
         '| --- | ---: | ---: | --- | --- |']
for row in table.itertuples(index=False):
    lines.append(f'| `{row.source_icaoaircrafttype}` | {row.rows} | {row.go_arounds} | '
                 f'{row.reference_description or "–"} | {"Сопоставлено" if row.status == "direct_unique_reference_match" else "Не разрешено"} |')
text = '''# Проверка неоднозначных кодов самолетов

Дата проверки – 08.10.2026. Рабочая база не изменена. Таблица ниже содержит предложения, а не выполненные исправления.

## Происхождение полей

[Описание Zenodo](https://zenodo.org/records/10210178) не раскрывает отдельно `typecode` и `icaoaircrafttype`. [Ноутбук авторов](https://github.com/figuetbe/OSN23-GoA/blob/main/OSN_paper_code.ipynb) читает готовый Parquet и отбирает записи по значениям `icaoaircrafttype`, например `L2J` и `L3J`. Код формирования справочного обогащения в этом ноутбуке отсутствует. Поэтому механизм смешения полей и версия справочника, примененная авторами, не установлены.

Страница [OpenSky Aircraft Database](https://opensky-network.org/data/aircraft) описывает сочетание официальных реестров, Basestation и данных сообщества. Это контекст возможных источников метаданных, а не доказательство конкретного пути обогащения этой таблицы.

## Справочное сопоставление

Использована публичная таблица, загружаемая [официальным поиском ICAO Doc 8643](https://www.icao.int/operational-safety/doc-8643-aircraft-type-designators/search): endpoint `https://doc8643.icao.int/External/AircraftTypes`, POST. Снимок содержит 7 261 справочную строку. В ней отдельно представлены `Designator` (код типа) и `Description` (описание, например `L2J`). Несколько моделей одного designator допустимы; предложение формируется только при единственном Description для этого designator.

Сопоставление выполнено по исходному `icaoaircrafttype`, приведенному к верхнему регистру только для поиска. Источник не изменялся. Текущий снимок подтверждает справочное соответствие, но не подтверждает состояние справочника в каждый день периода 2019–2023 и фактический тип конкретного борта. Исторический [FAA JO 7360.1D](https://www.faa.gov/documentLibrary/media/Order/Order_7360.1D_Aircraft_Type_Designators_FINAL.pdf), действующий с 24.05.2018, использован как дополнительная проверка; его обозначение двигателя и класса не переносится в поле источника автоматически.

''' + '\n'.join(lines) + '''

18 значений прямо сопоставились с одним описанием: 5 888 записей, 27 отмеченных уходов. Пять значений не имеют прямого совпадения в полученном снимке: 519 записей, 2 отмеченных ухода. Отсутствие совпадения не доказывает ошибку или несуществование исторического кода.

## Неразрешенные случаи

- `82R` / `typecode=737`: 457 записей. По этим полям нельзя однозначно выбрать модификацию. Не предлагаем подменять код на B738 или другой код по догадке.
- `G550`: 37 записей. Текущая таблица ICAO связывает модель Gulfstream G550 с GLF5, но исходное G550 не является прямым совпадением designator. Возможное ручное соответствие требует отдельного согласования и проверки борта.
- `G-V`: 8 записей. В справочнике есть семейство GLF5, но прямое совпадение отсутствует. Название модели не приравнивается к подтвержденному коду конкретного борта.
- `GL6T`: 12 записей. Прямого совпадения в текущей таблице ICAO нет. Историческое соответствие этим поиском не установлено; не считаем автоматически опечаткой GLEX.
- `X`: 5 записей. Значение неопределенно; не заменяем его на категорию конкретного самолета.

## Предлагаемое правило для согласования

Для 18 прямо сопоставленных значений добавить производное описание по справочнику, например `aircraft_description_reference`, и признак происхождения сопоставления. Исходные `icaoaircrafttype` и `typecode` сохранить. Для пяти неразрешенных значений производное описание оставить отсутствующим и пометить как неразрешенное. Удаление записей, перезапись исходных полей и заполнение typecode этим правилом не предлагаются.

В этой таблице FA8X и FA7X сопоставляются с L3J, E300 – с L1P, AT76 – с L2T. Поэтому единое исправление всех значений на L2J было бы содержательно неверным.

## Наблюдение для диссертации

Авторский ноутбук использует отбор по классам, позывным и числу попыток. Это показывает, что его аналитическая выборка отличается от исходной таблицы. Восстановление классификации может менять результат отбора по классам, даже если число строк исходной базы не меняется. Сравнение оценок при разных правилах справочного восстановления и отборе представляет отдельный вопрос чувствительности. Авторские фильтры в нашем проекте не применены; утверждение об учебном назначении по числу попыток не принимается как установленный факт.

Файлы справочников и авторского кода сохранены локально в `docs/sources/files/aircraft_code_review/`; URL, размеры и SHA256 записаны в `aircraft_reference_sources.json`. Предложения находятся в `03_aircraft_code_proposals.csv`; отобранные справочные строки – в `03_aircraft_reference_matches.csv`.
'''
(reports / '03_aircraft_code_review.md').write_text(text, encoding='utf8')
print(review.groupby('status')[['rows', 'go_arounds']].sum().to_string())
