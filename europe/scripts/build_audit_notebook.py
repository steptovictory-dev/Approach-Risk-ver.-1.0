from pathlib import Path
import nbformat

REGION=Path(__file__).resolve().parents[1]
nb=nbformat.v4.new_notebook()
cells=[]
def md(text): cells.append(nbformat.v4.new_markdown_cell(text.strip()))
def code(text): cells.append(nbformat.v4.new_code_cell(text.strip()))

md('''# 01 – Аудит базы данных Европы

Цель – установить состав, временное покрытие и ограничения исходной таблицы перед подготовкой собственной аналитической базы.

Источник: [Large aircraft landing dataset at 20 large europen airports](https://zenodo.org/records/10210178), Benoit Figuet, DOI `10.5281/zenodo.10210178`, версия v1 от 27.11.2023. Лицензия – CC-BY-4.0 по сохраненным метаданным Zenodo. ADS-B наблюдения получены авторами из OpenSky, обогащение – из EUROCONTROL Network Manager.

Одна строка содержит метаданные наблюдения полета с итоговой посадкой и может описывать несколько попыток захода. `GoA` означает наличие хотя бы одного отмеченного ухода, а не число уходов, аварийность или нарушение. Полные траектории и погодные наблюдения в файле отсутствуют.

Аудит проверяет весь исходный Parquet блоками по 100 000 строк. Никакие строки не исключаются, значения не исправляются. Отдельные диагностические выгрузки сохраняют исходные значения и номера строк. Графики и очистка выполняются следующими этапами.''')
code('''# Проверяем окружение и находим корень независимо от папки запуска Jupyter
from pathlib import Path
import importlib
import json
import sys

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from IPython.display import display, Markdown

ROOT = next((p for p in [Path.cwd(), *Path.cwd().parents]
             if (p / 'europe/data/source/osn23_landings_merged.parquet.gzip').is_file()
             and (p / '.git').exists()), None)
if ROOT is None:
    raise FileNotFoundError('Запустите Jupyter из проекта или папки europe/notebooks')
REGION = ROOT / 'europe'
SOURCE = REGION / 'data/source/osn23_landings_merged.parquet.gzip'
REPORTS = REGION / 'reports'
sys.path.insert(0, str(REGION / 'scripts'))
import profile_data
profile_data = importlib.reload(profile_data)

pd.set_option('display.max_rows', 70)
pd.set_option('display.max_columns', 12)
pd.set_option('display.max_colwidth', 90)
print('Python:', sys.version.split()[0])
print('pandas:', pd.__version__)
print('pyarrow:', pa.__version__)
print('Источник:', SOURCE.relative_to(ROOT).as_posix())
print('Размер файла, МБ:', round(SOURCE.stat().st_size / 1_000_000, 1))''')
md('''## Схема и первый просмотр

Сначала показываем фактические типы и десять исходных записей. Эти десять строк служат только примером структуры; все дальнейшие показатели рассчитываются по всему файлу.

[Русский словарь полей](../docs/data_dictionary.md) отделяет описание авторов от интерпретаций, которые еще требуют проверки. В схеме есть служебный индекс pandas; он не считается содержательным признаком или уникальным идентификатором.''')
code('''# Читаем физическую схему, сохраняя все поля исходника
parquet = pq.ParquetFile(SOURCE)
schema = pd.DataFrame([{'Поле': f.name, 'Тип Arrow': str(f.type),
                       'Назначение': 'Служебный индекс' if f.name == '__index_level_0__' else 'Данные'}
                      for f in parquet.schema_arrow])
display(schema)
print('Физических столбцов:', len(parquet.schema_arrow))
print('Содержательных полей:', len(schema) - int((schema['Поле'] == '__index_level_0__').sum()))
print('Строк по метаданным Parquet:', parquet.metadata.num_rows)''')
code('''# Выводим небольшой пример без преобразования списков в модельные признаки
sample = next(parquet.iter_batches(batch_size=10)).to_pandas(ignore_metadata=True)
display(sample[['flight_id', 'icao24', 'airport', 'landing_time', 'GoA',
                'n_attempts', 'attempt_times', 'ILS']])''')
md('''## Полная проверка исходника

Следующая ячейка запускает полный аудит и заново формирует отчеты в `europe/reports`. Скрипт проверяет SHA256 по реестру источников до чтения и повторно после завершения. Для воспроизведения всех таблиц нужно выполнить ячейки по порядку.

[Код полного аудита](../scripts/profile_data.py) включает проверки вложенных списков, хронологии событий и точное сравнение кандидатов повторов. `data/review` содержит диагностические копии строк, а не очищенную базу.''')
code('''# Выполняем полный проход; исходный Parquet открывается только для чтения
summary = profile_data.profile_source()
assert summary['source_unchanged']
assert summary['rows_deleted'] == 0 and summary['values_corrected'] == 0

def report(name):
    return pd.read_csv(REPORTS / name, keep_default_na=False)

LABELS = {'rows':'Кол-во записей', 'go_arounds':'С уходом', 'ga_percent':'Доля с уходом, %',
          'country':'Страна', 'airport':'Аэропорт', 'city':'Город', 'month':'Месяц', 'year':'Год',
          'n_attempts':'Кол-во попыток', 'reason':'Диагностическая причина', 'percent':'Доля записей, %'}

def show(frame):
    display(frame.rename(columns=LABELS).style.format(
        {LABELS.get(c,c):'{:.1f}' for c in ['ga_percent','percent'] if c in frame.columns}))

overview = pd.DataFrame([
    ['Кол-во записей', f"{summary['rows']:,}"],
    ['Кол-во содержательных полей', summary['business_columns']],
    ['Аэропортов', summary['airports']],
    ['Стран', summary['countries']],
    ['Записей с уходом', f"{summary['go_arounds']:,}"],
    ['Доля с уходом, %', f"{summary['ga_percent']:.4f}"],
    ['Начало периода UTC', summary['min_landing_time']],
    ['Конец периода UTC', summary['max_landing_time']],
], columns=['Показатель', 'Значение'])
display(overview)''')
md('''## Географический состав

Показываем все страны и все аэропорты исходного файла. Города добавлены как подписи для чтения, а не как исходные признаки. Европейский охват здесь соответствует выборке авторов; принадлежность к этой выборке не доказывает единообразие процедур и условий наблюдения.''')
code('''# Состав по странам; суммы должны совпасть с полным набором
countries = report('country_summary.csv').sort_values('rows', ascending=False)
assert int(countries['rows'].sum()) == summary['rows']
assert int(countries['go_arounds'].sum()) == summary['go_arounds']
show(countries)''')
code('''# Все аэропорты с городами и частотами отмеченных событий
airports = report('airport_summary.csv').sort_values('rows', ascending=False)
assert int(airports['rows'].sum()) == summary['rows']
assert int(airports['go_arounds'].sum()) == summary['go_arounds']
show(airports)
print('Аэропортов с несколькими значениями страны:', int((airports.groupby('airport')['country'].nunique() > 1).sum()))''')
md('''Частота отмеченных уходов условна относительно состава и покрытия этого источника. Сравнение аэропортов по одной доле не является оценкой их безопасности. Отбор по критерию точности и исключения из прежнего эксперимента здесь не применяются.''')
md('''## Временное покрытие

Группируем записи по времени итоговой посадки в UTC. Частичный 2023 год нельзя сравнивать с полными годами без учета сезонности. Наличие наблюдений в месяце не означает полноту всех его дней и полетов.

Описание Zenodo указывает период до июня 2023 года. Ниже проверяется фактический период локального файла; при расхождении сохраняем оба утверждения и не дополняем данные догадками.''')
code('''# Показываем каждый представленный год
annual = report('year_summary.csv')
show(annual)
assert int(annual['rows'].sum()) == summary['rows']''')
code('''# Показываем все месяцы фактического периода
monthly = report('month_summary.csv')
show(monthly)
assert int(monthly['rows'].sum()) == summary['rows']
print('Фактический последний месяц:', monthly['month'].max())''')
code('''# Проверяем дни наблюдений отдельно для каждого аэропорта
coverage = report('coverage_summary.csv')
display(coverage.rename(columns={'airport':'Аэропорт', 'observed_days':'Дней с записями',
    'calendar_days':'Дней в общем периоде', 'days_without_records':'Дней без записей',
    'observed_day_percent':'Доля дней с записями, %', 'first_calendar_day':'Начало общего периода',
    'last_calendar_day':'Конец общего периода'}).style.format({'Доля дней с записями, %':'{:.1f}'}))
print('Подробные таблицы: airport_month_summary.csv и airport_calendar.csv.')''')
md('''День без записей означает отсутствие наблюдений в файле. Причиной могут быть отбор авторов, доступность ADS-B, изменение работы аэропорта или другие обстоятельства. По этой таблице нельзя установить причину или фактическую интенсивность движения.''')
md('''## Поля и представления пропусков

Раздельно считаем физический null, пустую строку, строку из пробелов и текстовые маркеры отсутствия. Для списков дополнительно проверяем пустые списки и null внутри. Не заменяем эти значения нулями и не объединяем их автоматически.''')
code('''# Пропуски по каждому физическому полю
missing = report('missingness.csv')
display(missing.rename(columns={'column':'Поле', 'arrow_type':'Тип Arrow', 'null_count':'Null',
    'null_percent':'Null, %', 'empty_string_count':'Пустая строка',
    'whitespace_only_count':'Только пробелы', 'text_missing_marker_count':'Текстовый маркер',
    'unique_nonnull_strings':'Уникальных строк без null', 'role':'Роль'}).style.format({'Null, %':'{:.1f}'}))''')
code('''# Диапазоны числовых полей, без исправления значений
numeric = report('numeric_ranges.csv')
display(numeric.rename(columns={'column':'Поле', 'count':'Конечных значений',
    'nonfinite':'Неконечных непустых значений', 'min':'Минимум', 'max':'Максимум'}))''')
md('''Диапазоны широты, долготы и направления используются для диагностики. Они не подтверждают правильность конкретной точки входа. Метеоданные, высота и скорость самолета в схеме отсутствуют; проверки температуры и давления из старого эксперимента сюда не переносятся.''')
md('''## Категории воздушных судов и сегменты перевозок

В `icaoaircrafttype` обнаружены значения разного формата: например, `L2J` рядом с `A332` и `a320`. В диагностике выделяются непустые значения вне преобладающего шаблона «буква + цифра + буква» и совпадения с `typecode` без учета регистра. Это проверка структуры поля, а не нормативное правило и не автоматическое исправление.

Физический null в таблицах обозначен как `<NULL>` только для отображения. Пустая строка сохраняется как отдельная категория. Статус справочников, историческая актуальность и доступность этих сведений в момент будущего прогноза требуют проверки.''')
code('''# Все представленные сегменты перевозок
show(report('market_segment_summary.csv'))''')
code('''# Классы воздушных судов и составные классификационные коды
show(report('AC_CLASS_summary.csv'))
show(report('icaoaircrafttype_summary.csv'))''')
code('''# Типы самолетов: первые 20 по объему, полный перечень сохранен в отчете
types = report('typecode_summary.csv')
show(types.head(20))
print('Категорий typecode, включая представления пропусков:', len(types))''')
md('''## Разметка и попытки захода

Проверяем распределение `n_attempts`, длины `attempt_times` и `ILS`, а также согласованность с `GoA`. Совпадение метки с условием `n_attempts > 1` показывает внутреннюю согласованность полей, но не является независимой проверкой событий.

Больше двух попыток – отдельная диагностическая группа. Она не объявляется учебными полетами и не исключается из базы.''')
code('''# Полное распределение количества попыток
attempts = report('attempts_summary.csv')
show(attempts)
assert int(attempts['rows'].sum()) == summary['rows']''')
code('''# Длины списков попыток и обозначений направлений
lengths = report('list_lengths.csv')
display(lengths.rename(columns={'field':'Поле', 'list_length':'Длина списка', 'rows':'Кол-во записей'}))''')
md('''## Повторы идентификаторов и записей

Проверяем `flight_id` отдельно от составного ключа `landing_time + icao24 + airport`. Хеши используются только для отбора кандидатов. Итоговые совпадения устанавливаются по точным значениям ключа и всех содержательных полей, включая списки. Служебный индекс pandas исключен из определения содержательного дубля.

Совпадение идентификатора, ключа или даже всех полей требует отдельного решения об обработке; этот аудит не удаляет записи.''')
code('''# Выводим точные результаты проверки повторов
with (REPORTS / 'duplicate_check.json').open(encoding='utf8') as stream:
    duplicates = json.load(stream)
display(pd.DataFrame([
    ['Повторяющихся flight_id', duplicates['flight_id_repeated_values']],
    ['Дополнительных строк с повтором flight_id', duplicates['flight_id_excess_rows']],
    ['Повторяющихся составных ключей', duplicates['repeated_core_keys']],
    ['Строк в группах повторяющегося составного ключа', duplicates['rows_in_repeated_core_keys']],
    ['Дополнительных строк с повтором составного ключа', duplicates['excess_core_key_rows']],
    ['Дополнительных полных дублей без служебного индекса', duplicates['full_duplicate_rows_excluding_storage_index']],
], columns=['Показатель', 'Кол-во']))
if duplicates['exact_core_key_examples']:
    display(pd.DataFrame(duplicates['exact_core_key_examples'])[
        ['source_row','flight_id','icao24','airport','landing_time','n_attempts','GoA']])''')
code('''# Примеры повторяющихся flight_id; совпадение имени не означает один и тот же полет
identifier_examples = pd.DataFrame(duplicates['repeated_identifier_examples'])
if not identifier_examples.empty:
    display(identifier_examples[['flight_id','icao24','airport','landing_time']].head(15))''')
md('''## Время входа в ASMA и пригодность для прогнозирования

`C40_CROSS_TIME` заявлено как время входа в область радиусом 40 NM. Проверяем наличие координат, положение этого времени относительно первой попытки и итоговой посадки. Разность времени посадки и входа приводится только как диагностический показатель.

Отсутствие флага не доказывает доступность сведений в реальном времени. До обучения потребуется определить момент прогноза, источник признаков и правила работы с пропусками.''')
code('''# Сводка времени между входом в ASMA и итоговой посадкой
with (REPORTS / 'c40_to_landing_duration.json').open(encoding='utf8') as stream:
    c40_duration = json.load(stream)
display(pd.DataFrame([
    ['Записей с обоими временами', c40_duration['rows_with_both_times']],
    ['Минимум, мин', c40_duration['min']],
    ['1-й процентиль, мин', c40_duration['quantiles'].get('0.01')],
    ['Медиана, мин', c40_duration['quantiles'].get('0.5')],
    ['99-й процентиль, мин', c40_duration['quantiles'].get('0.99')],
    ['Максимум, мин', c40_duration['max']],
], columns=['Показатель','Значение']))''')
md('''## Диагностические флаги и примеры

Проверки охватывают обязательные значения, формат адресов и аэропортов, согласованность количества попыток и списков, порядок времен, время входа в ASMA и диапазоны координат. Формат обозначения ВПП проверяется отдельно от исторической применимости процедуры.

Флаг – основание для разбора. Он не доказывает ошибку, нарушение, учебное назначение полета или опасность захода. Причины могут пересекаться, поэтому суммы строк по флагам не равны числу уникальных отмеченных записей.''')
code('''# Только ненулевые группы, полный перечень сохранен в CSV
flags = report('review_flag_summary.csv')
show(flags.loc[flags['rows'] > 0])
print('Уникальных записей с хотя бы одним флагом:', f"{summary['unique_flagged_rows']:,}")
print('Среди них с уходом:', f"{summary['flagged_go_arounds']:,}")''')
code('''# По одному примеру на причину; все исходные поля сохранены в диагностическом Parquet
with (REPORTS / 'review_examples.json').open(encoding='utf8') as stream:
    examples = pd.DataFrame(json.load(stream))
if not examples.empty:
    display(examples.drop_duplicates('review_reason')[
        ['review_reason','source_row','flight_id','airport','landing_time','n_attempts','GoA']])
print('Номер source_row начинается с нуля и соответствует порядку строк Parquet.')''')
md('''## Сохранность и вывод аудита

Проверка исходника выполняется независимо от диагностических отчетов. Очищенный файл не создается. Сохраненные результаты относятся к исходному набору; будущие правила обработки должны иметь отдельное обоснование и версию.''')
code('''# Подтверждаем сохранность исходного файла и согласованность итогов
assert profile_data.sha256(SOURCE) == summary['source_sha256']
assert summary['source_unchanged']
assert summary['rows_deleted'] == 0
assert summary['values_corrected'] == 0
for name in ['country_summary.csv','airport_summary.csv','month_summary.csv','year_summary.csv','attempts_summary.csv']:
    table = report(name)
    assert int(table['rows'].sum()) == summary['rows'], name
    assert int(table['go_arounds'].sum()) == summary['go_arounds'], name
print('Исходный Parquet сохранен без изменений. Суммы агрегатов сходятся.')

display(Markdown(
    f"Проверены **{summary['rows']:,} записей**, **{summary['airports']} аэропортов**, "
    f"**{summary['countries']} стран**. Записей с отмеченным уходом: **{summary['go_arounds']:,}** "
    f"(**{summary['ga_percent']:.4f}%**). Фактический период UTC: "
    f"**{summary['min_landing_time']} – {summary['max_landing_time']}**. "
    f"Отдельного разбора требуют **{summary['unique_flagged_rows']:,} записей с диагностическими флагами**, "
    f"повторы идентификаторов и особенности временного покрытия. "
    f"Максимальное число попыток: **{int(summary['max_n_attempts'])}**. "
    f"Это результаты диагностики, а не решения об исключении."
))''')
md('''Следующий этап – разобрать пропуски, покрытие, группы повторов и флагов; проверить смысл спорных полей по источнику и согласовать обработку. После этого формируется производная аналитическая база.

Полные траектории и независимая проверка меток пока отсутствуют. Нельзя объявлять таблицу готовой для раннего прогнозирования или делать выводы о причинах ухода и безопасности аэропортов. Обучение моделей в этом ноутбуке не выполняется.''')

for cell in cells:
    if cell.cell_type == 'code':
        compile(cell.source, '<audit-notebook-cell>', 'exec')
nb['cells']=cells
nb['metadata']={'kernelspec':{'display_name':'Python (окружение проекта)','language':'python','name':'python3'},
                'language_info':{'name':'python','version':'3.14.6'}}
nbformat.validate(nb)
path=REGION/'notebooks/01_database_audit.ipynb'
nbformat.write(nb,path)
print(f'Создан ноутбук: {len(cells)} ячеек.')
