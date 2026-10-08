from pathlib import Path
import ast
import nbformat

region = Path(__file__).resolve().parents[1]
nb = nbformat.v4.new_notebook()
nb.cells = [
    nbformat.v4.new_markdown_cell('''# 03 – Очистка и подготовка данных Европы

## Шаг 1 – Диагностика представлений пропусков

На этом шаге только читаем исходную базу и показываем результаты. Преобразование значений, заполнение пропусков, исключение строк и создание очищенной базы требуют отдельного согласования. Проверяем всю таблицу блоками, без загрузки полной базы в память.'''),
    nbformat.v4.new_code_cell('''from pathlib import Path
import hashlib
import json
from collections import Counter
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from IPython.display import display

ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents]
            if (p / 'europe/data/source/osn23_landings_merged.parquet.gzip').exists())
SOURCE = ROOT / 'europe/data/source/osn23_landings_merged.parquet.gzip'

def source_hash():
    h = hashlib.sha256()
    with SOURCE.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()

before = source_hash()
audit = json.loads((ROOT / 'europe/reports/profile_summary.json').read_text(encoding='utf8'))
assert before == audit['source_sha256'], 'Источник отличается от файла аудита 01.'
parquet = pq.ParquetFile(SOURCE)
assert parquet.metadata.num_rows == audit['rows']
print('Записей:', f'{parquet.metadata.num_rows:,}'.replace(',', ' '))
print('SHA256 совпадает с аудитом 01.')'''),
    nbformat.v4.new_markdown_cell('''## Подсчет и примеры

Разделяем физический `null`, точную пустую строку `""` и непустую строку, состоящую только из пробельных символов. Проверяем строковые столбцы верхнего уровня; содержимое списков `ILS` и `attempt_times` здесь не анализируется. Примеры ограничены тремя записями на поле и вид пропуска. Позиция строки – нумерация с нуля в исходном файле, а не идентификатор полета.

Пробелы проверяются диагностически через `str.strip()`. Исходные значения не заменяются и не обрезаются. Текстовые обозначения вроде `unknown` не признаются пропусками автоматически.'''),
    nbformat.v4.new_code_cell('''fields = [f.name for f in parquet.schema_arrow
          if pa.types.is_string(f.type) or pa.types.is_large_string(f.type)]
counts = {field: Counter() for field in fields}
examples = []
shown = Counter()
offset = 0
for batch in parquet.iter_batches(batch_size=100_000, columns=fields):
    frame = batch.to_pandas(ignore_metadata=True)
    for field in fields:
        values = frame[field]
        masks = {
            'null': values.isna(),
            'empty_string': values.eq('').fillna(False),
            'whitespace_only': (values.notna() & values.ne('')
                                & values.str.strip().eq('')).fillna(False),
        }
        for kind, mask in masks.items():
            counts[field][kind] += int(mask.sum())
            remaining = 3 - shown[(field, kind)]
            if remaining > 0:
                for position in frame.index[mask][:remaining]:
                    examples.append({'field': field, 'kind': kind,
                                     'source_row_position': offset + int(position),
                                     'raw_repr': repr(values.loc[position])})
                    shown[(field, kind)] += 1
    offset += len(frame)

assert offset == audit['rows']
missingness = pd.DataFrame([{'field': field, **dict(counts[field])} for field in fields])
for kind in ['null', 'empty_string', 'whitespace_only']:
    missingness[kind + '_percent'] = 100 * missingness[kind] / offset
assert source_hash() == before, 'Контрольная сумма источника изменилась.'
display(missingness)
print('Исходный файл не изменился. Преобразования не выполнялись.')'''),
    nbformat.v4.new_code_cell('''display(pd.DataFrame(examples))
print('Примеры показывают исходные представления; это не правила обработки.')'''),
    nbformat.v4.new_markdown_cell('''## Шаг 2 – Согласованная нормализация пустых строк

Согласовано: только в `registration`, `typecode` и `icaoaircrafttype` заменяем точную пустую строку на физический null. Сохраняем три булевых поля `<имя>_was_empty_string`: True означает исходную пустую строку, False – любое другое исходное значение, включая исходный null. Вместе с неизмененным источником это позволяет восстановить исходное представление.

Записи и остальные значения сохраняются. Это промежуточная рабочая копия после одного правила, а не окончательная аналитическая база. Заполнение пропусков и другие правила еще не согласованы.'''),
    nbformat.v4.new_code_cell('''import pyarrow.compute as pc

approved_fields = ['registration', 'typecode', 'icaoaircrafttype']
WORK = ROOT / 'europe/data/working'
WORK.mkdir(parents=True, exist_ok=True)
DESTINATION = WORK / '01_empty_strings_normalized.parquet'
TEMPORARY = WORK / '01_empty_strings_normalized.parquet.part'
output_schema = parquet.schema_arrow
for field in approved_fields:
    assert field + '_was_empty_string' not in output_schema.names
    output_schema = output_schema.append(pa.field(field + '_was_empty_string', pa.bool_()))

replaced = Counter()
written = 0
with pq.ParquetWriter(TEMPORARY, output_schema, compression='gzip', compression_level=1) as writer:
    for batch in parquet.iter_batches(batch_size=100_000):
        table = pa.Table.from_batches([batch])
        for field in approved_fields:
            original = table[field]
            mask = pc.fill_null(pc.equal(original, ''), False)
            replaced[field] += pc.sum(pc.cast(mask, pa.int64())).as_py()
            normalized = pc.if_else(mask, pa.scalar(None, type=original.type), original)
            table = table.set_column(table.schema.get_field_index(field),
                                     table.schema.field(field), normalized)
            table = table.append_column(field + '_was_empty_string', mask)
        writer.write_table(table)
        written += len(table)

# Проверяем обратимость каждого изменения и неизменность остальных полей
candidate = pq.ParquetFile(TEMPORARY)
assert candidate.metadata.num_rows == written == audit['rows']
checked = 0
after_nulls = Counter()
for source_batch, result_batch in zip(parquet.iter_batches(batch_size=100_000),
                                      candidate.iter_batches(batch_size=100_000), strict=True):
    original = pa.Table.from_batches([source_batch])
    result = pa.Table.from_batches([result_batch])
    for field in approved_fields:
        flag = result[field + '_was_empty_string']
        expected_flag = pc.fill_null(pc.equal(original[field], ''), False)
        assert flag.equals(expected_flag)
        assert pc.sum(pc.cast(pc.fill_null(pc.equal(result[field], ''), False), pa.int64())).as_py() == 0
        after_nulls[field] += result[field].null_count
        restored = pc.if_else(flag, pa.scalar('', type=original[field].type), result[field])
        result = result.set_column(result.schema.get_field_index(field), original.schema.field(field), restored)
    assert result.select(original.column_names).equals(original)
    checked += len(original)
assert checked == written
assert source_hash() == before
for field in approved_fields:
    assert replaced[field] == counts[field]['empty_string']
    assert after_nulls[field] == counts[field]['null'] + replaced[field]

candidate.close()
TEMPORARY.replace(DESTINATION)
change_log = pd.DataFrame([{'field': field, 'empty_strings_replaced': replaced[field],
                            'null_before': counts[field]['null'], 'null_after': after_nulls[field]}
                           for field in approved_fields])
display(change_log)
REPORT = ROOT / 'europe/reports/preparation'
REPORT.mkdir(parents=True, exist_ok=True)
change_log.to_csv(REPORT / '01_empty_strings_normalized.csv', index=False)
(REPORT / '01_empty_strings_normalized.json').write_text(json.dumps({
    'rule': 'exact empty string to null in three approved fields',
    'approved_fields': approved_fields,
    'source_sha256': before, 'source_unchanged': True,
    'output': str(DESTINATION.relative_to(ROOT)),
    'output_sha256': hashlib.file_digest(DESTINATION.open('rb'), 'sha256').hexdigest(),
    'rows_before': written, 'rows_after': written, 'rows_deleted': 0,
    'reversible_comparison_passed': True,
    'changes': change_log.to_dict(orient='records'),
}, ensure_ascii=False, indent=2) + '\\n', encoding='utf8')
print('Сохранена рабочая копия:', DESTINATION)
print('Все записи сохранены. Проверена обратимость и неизменность остальных полей.')'''),
    nbformat.v4.new_markdown_cell('''## Шаг 3 – Диагностика неоднозначных кодов самолетов

Читаем рабочую копию после согласованной нормализации пустых строк. Сравниваем `icaoaircrafttype` с преобладающим шаблоном `[A-Z][0-9][A-Z]` и с `typecode`. Это структурная проверка, а не справочник допустимых авиационных кодов. Верхний регистр используется только при сравнении, исходные значения не изменяются.

Показываем все непустые значения вне шаблона, пары с `typecode`, распределение по аэропортам и до трех исходных примеров на значение. Возможные соответствия внутри базы служат подсказкой для последующей проверки источника.'''),
    nbformat.v4.new_code_cell('''from pathlib import Path
from collections import Counter
import hashlib
import json
import pandas as pd
import pyarrow.parquet as pq
from IPython.display import display

ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents]
            if (p / 'europe/data/source/osn23_landings_merged.parquet.gzip').exists())
WORKING = ROOT / 'europe/data/working/01_empty_strings_normalized.parquet'
REPORT = ROOT / 'europe/reports/preparation'
log = json.loads((REPORT / '01_empty_strings_normalized.json').read_text(encoding='utf8'))

def hash_file(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

working_before = hash_file(WORKING)
assert working_before == log['output_sha256']
reader = pq.ParquetFile(WORKING)
pair_counts = Counter()
pair_events = Counter()
airport_counts = Counter()
consistent_pairs = Counter()
sample_counts = Counter()
samples = []
rows = flagged = matched_typecode = missing_class = 0
columns = ['flight_id', 'icao24', 'landing_time', 'airport', 'typecode',
           'icaoaircrafttype', 'GoA', 'n_attempts']
for batch in reader.iter_batches(batch_size=100_000, columns=columns):
    frame = batch.to_pandas(ignore_metadata=True)
    classification = frame['icaoaircrafttype']
    valid_shape = classification.str.fullmatch(r'[A-Z][0-9][A-Z]', na=False)
    suspicious = classification.notna() & ~valid_shape
    missing_class += int(classification.isna().sum())
    matches = (classification.notna() & frame['typecode'].notna()
               & classification.str.upper().eq(frame['typecode'].str.upper())).fillna(False)
    matched_typecode += int(matches.sum())
    for record in frame.loc[valid_shape & frame['typecode'].notna(), ['typecode', 'icaoaircrafttype']].itertuples(index=False):
        consistent_pairs[(record.typecode, record.icaoaircrafttype)] += 1
    for position, record in frame.loc[suspicious].iterrows():
        code = record['icaoaircrafttype']
        typecode = None if pd.isna(record['typecode']) else record['typecode']
        pair = (code, typecode)
        pair_counts[pair] += 1
        pair_events[pair] += int(record['GoA'])
        airport_counts[(code, record['airport'])] += 1
        if sample_counts[code] < 3:
            sample = record.to_dict()
            sample['source_row_position'] = rows + int(position)
            sample['equals_typecode_ignoring_case'] = bool(matches.loc[position])
            samples.append(sample)
            sample_counts[code] += 1
    flagged += int(suspicious.sum())
    rows += len(frame)

assert rows == log['rows_after']
assert hash_file(WORKING) == working_before
pairs = pd.DataFrame([{'icaoaircrafttype': code, 'typecode': typecode, 'rows': count,
                       'go_arounds': pair_events[(code, typecode)]}
                      for (code, typecode), count in pair_counts.items()]).sort_values('rows', ascending=False)
codes = pairs.groupby('icaoaircrafttype')[['rows', 'go_arounds']].sum().sort_values('rows', ascending=False)
codes['ga_percent'] = 100 * codes['go_arounds'] / codes['rows']
airport_distribution = pd.DataFrame([{'icaoaircrafttype': code, 'airport': airport, 'rows': count}
                                    for (code, airport), count in airport_counts.items()]).sort_values('rows', ascending=False)
comparison = pd.DataFrame([{'typecode': typecode, 'icaoaircrafttype': code, 'rows': count}
                           for (typecode, code), count in consistent_pairs.items()])
affected_types = pairs['typecode'].dropna().unique()
comparison = comparison.loc[comparison['typecode'].isin(affected_types)].sort_values(['typecode', 'rows'], ascending=[True, False])
codes.to_csv(REPORT / '02_aircraft_code_values.csv')
pairs.to_csv(REPORT / '02_aircraft_code_pairs.csv', index=False)
airport_distribution.to_csv(REPORT / '02_aircraft_code_airports.csv', index=False)
comparison.to_csv(REPORT / '02_aircraft_code_internal_reference.csv', index=False)
print('Всего записей:', rows)
print('Классификация отсутствует:', missing_class)
print('Непустые значения вне преобладающего шаблона:', flagged)
print('Совпадают с typecode без учета регистра:', matched_typecode)
display(codes)
print('Рабочая копия не изменена. Исправления не применялись.')'''),
    nbformat.v4.new_code_cell('''print('Пары значений вне шаблона и typecode:')
display(pairs)
print('Распределение по аэропортам:')
display(airport_distribution)
print('Другие классификации тех же typecode внутри базы – только для сравнения:')
display(comparison)
print('Примеры записей, до трех на значение классификации:')
with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    display(pd.DataFrame(samples))'''),
    nbformat.v4.new_markdown_cell('''## Согласование следующего правила

Исправления кодов не применены. Выход за преобладающий шаблон и совпадение двух полей не доказывают конкретное правильное значение. Следующее правило согласуем после просмотра групп и проверки происхождения полей. Нормализация регистра, перенос кодов между полями, замена на null и удаление строк здесь не выполняются.'''),
]
nb.cells.extend([
    nbformat.v4.new_markdown_cell('''## Шаг 4 – Проверка по справочнику и предложения

Проверка источников оформлена в [отдельном разборе](../reports/preparation/03_aircraft_code_review.md). Использована публичная таблица [ICAO Doc 8643](https://www.icao.int/operational-safety/doc-8643-aircraft-type-designators/search), полученная 08.10.2026. Текущий справочник не доказывает историческое состояние в 2019–2023 годах или тип конкретного борта.

18 значений имеют прямое соответствие с одним Description: 5 888 записей и 27 отмеченных уходов. Еще пять значений остаются неразрешенными: 519 записей и 2 отмеченных ухода. Таблица ниже содержит предложения; они не применены.'''),
    nbformat.v4.new_code_cell('''from pathlib import Path
import hashlib
import json
import pandas as pd
from IPython.display import display

ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents]
            if (p / 'europe/data/source/osn23_landings_merged.parquet.gzip').exists())
REPORT = ROOT / 'europe/reports/preparation'
proposals = pd.read_csv(REPORT / '03_aircraft_code_proposals.csv', keep_default_na=False)
assert len(proposals) == 23 and proposals['rows'].sum() == 6407
assert not proposals['applied'].any()
with (ROOT / 'europe/data/working/01_empty_strings_normalized.parquet').open('rb') as stream:
    digest = hashlib.file_digest(stream, 'sha256').hexdigest()
log = json.loads((REPORT / '01_empty_strings_normalized.json').read_text(encoding='utf8'))
assert digest == log['output_sha256']
with pd.option_context('display.max_rows', None):
    display(proposals[['source_icaoaircrafttype', 'rows', 'go_arounds',
                       'lookup_designator', 'reference_description', 'status']])
print('Рабочая копия не изменена. Справочные предложения не применены.')'''),
    nbformat.v4.new_markdown_cell('''## Требуется согласование справочного правила

Предлагаем добавить производное описание по прямым однозначным соответствиям справочника для 18 значений и признак источника сопоставления. Исходные `icaoaircrafttype` и `typecode` сохранить. Пять неразрешенных значений оставить для отдельного разбора. Перезапись исходных полей, удаление записей и заполнение typecode этим правилом не выполняются.

Только после согласования добавляется ячейка преобразования.'''),
])
nb.cells.extend([
    nbformat.v4.new_markdown_cell('''## Шаг 5 – Согласованное справочное обогащение

Согласовано добавление производного описания для 18 прямо сопоставленных значений. Исходные `icaoaircrafttype`, `typecode` и все остальные существующие поля сохраняются. Новая копия – `europe/data/working/02_aircraft_reference_enriched.parquet`.

Добавляем `aircraft_description_reference`, `aircraft_reference_status`, `aircraft_reference_source` и `aircraft_reference_checked_date`. Для 5 888 сопоставленных записей источник – публичная таблица [ICAO Doc 8643](https://www.icao.int/operational-safety/doc-8643-aircraft-type-designators/search), снимок 08.10.2026. Производное описание для остальных записей остается null: статус различает неразрешенные значения, отсутствующее исходное описание и значения вне текущего разбора.

Историческое соответствие справочника за 2019–2023 годы не подтверждено. Производное поле не означает независимое подтверждение типа конкретного борта. Сначала проверяем SHA256 входной копии и справочного снимка; после записи сравниваем все существующие поля с входом и проверяем производные значения.'''),
    nbformat.v4.new_code_cell('''from pathlib import Path
import runpy
import json
from IPython.display import display

ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents]
            if (p / 'europe/data/source/osn23_landings_merged.parquet.gzip').exists())
module = runpy.run_path(str(ROOT / 'europe/scripts/apply_aircraft_reference.py'))
changes = module['main']()
display(changes)
journal = json.loads((ROOT / 'europe/reports/preparation/04_aircraft_reference_applied.json').read_text(encoding='utf8'))
print('Записей до и после:', journal['rows_before'], journal['rows_after'])
print('Отмеченных уходов до и после:', journal['go_arounds_before'], journal['go_arounds_after'])
print('Справочное описание добавлено:', journal['rows_enriched'])
print('Статусы:', journal['status_counts'])
print('Все существующие поля неизменны:', journal['existing_columns_unchanged'])
print('Официальный источник:', journal['reference_url'])
print('Дата проверки:', journal['reference_checked_date'])'''),
    nbformat.v4.new_markdown_cell('''## Следующие действия требуют отдельного согласования

Выполнены только нормализация точных пустых строк в трех полях и согласованное справочное обогащение 18 значений. Исходные коды не перезаписаны. Пять неразрешенных значений не исправлены, записи не исключены. Следующий разбор и любое новое правило согласуются отдельно.'''),
])
nb.cells.extend([
    nbformat.v4.new_markdown_cell('''## Шаг 6 – Неразрешенные значения и история бортов

Только диагностика. Проверяем 519 записей с `82R`, `G550`, `G-V`, `GL6T`, `X` и все наблюдения тех же ICAO-адресов в рабочей копии после справочного обогащения. Регистрации и исходные коды сохраняются.

Официальные сведения, кандидаты и исторические ограничения оформлены в [отдельном разборе](../reports/preparation/05_unresolved_aircraft_review.md). Идентичность борта нельзя считать постоянной только по регистрации или адресу. Текущий реестр не заменяет историческое подтверждение.'''),
    nbformat.v4.new_code_cell('''from pathlib import Path
import runpy
import pandas as pd
from IPython.display import display

ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents]
            if (p / 'europe/data/source/osn23_landings_merged.parquet.gzip').exists())
module = runpy.run_path(str(ROOT / 'europe/scripts/diagnose_unresolved_aircraft.py'))
summary, history = module['main']()
display(summary)
print('Другие сочетания полей тех же адресов во всей базе:')
with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    display(history)
print('ICAO-адресов:', history['icao24'].nunique())
print('Рабочая копия проверена по SHA256 и не изменена.')'''),
    nbformat.v4.new_code_cell('''assessment = pd.read_csv(ROOT / 'europe/reports/preparation/05_unresolved_aircraft_assessment.csv', keep_default_na=False)
assert len(assessment) == 7 and assessment['rows'].sum() == 519
assert not assessment['applied'].any()
with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    display(assessment)
print('Кандидаты существуют только в диагностическом отчете. Исправления не применены.')'''),
    nbformat.v4.new_markdown_cell('''## Итог шага 6

Все 519 записей относятся к семи ICAO-адресам; других сочетаний кодов этих адресов в базе не найдено. Четыре борта имеют кандидатов с разными ограничениями подтверждения, три остаются без подтвержденного кандидата. Ни один найденный документ не подтверждает идентичность во все даты наблюдений группы.

Доступные официальные источники: [FAA NAARMO, июнь 2026, PDF-страница 170](https://www.faa.gov/sites/faa.gov/files/2026-06/NAARMO_US_IGA_RVSM_Approvals_Part_135_91_Section_3_0.pdf), [ANAC, отчет за 2023 год, PDF-страница 217](https://www.anac.pt/SiteCollectionDocuments/Informacao_Gestao/Relatorios_Gestao/Relatorio_de_Atividades_Gest%C3%A3o_e_Contas_2023.pdf). Сведения поискового индекса AACR и DAC Luxembourg отдельно помечены: полный оригинал не удалось получить.

Новые преобразования не выполнены. Прежде чем использовать кандидатов, отдельно согласуем допустимость неполного исторического подтверждения. Альтернатива – сохранить записи неразрешенными и перейти к диагностике C40.'''),
])
nb.cells.extend([
    nbformat.v4.new_markdown_cell('''## Шаг 7 – Временная диагностика C40

Проверяем все 7 984 254 записи рабочей копии после справочного обогащения. C40 – метаданные входа в ASMA, 40-мильную область вокруг опорной точки аэропорта; это не отметка входа на глиссаду ILS.

Первичный источник полей – [описание автора на Zenodo](https://zenodo.org/records/10210178). [Статья авторов, разделы 2.2–2.3](https://doi.org/10.59490/joas.2023.7219) объясняет разные источники: посадочные метаданные из OpenSky и ASMA из EUROCONTROL; сопоставление по дню, аэропорту и ICAO24. Эти ключи не гарантируют однозначности конкретной записи.

[EUROCONTROL/EASA AVENIR, раздел 5.4, PDF-страница 43](https://www.eurocontrol.int/sites/default/files/2025-09/avenir-wg-pillar-1-final-report.pdf) описывает переход от последнего пересечения 40 NM к первому входу в обновленной методике. Это допускает гипотезу повторного входа после ухода, но не устанавливает алгоритм формирования нашего файла. Публичный ноутбук автора использует готовую таблицу; код ее исходного сопоставления в нем не найден.

Отдельно проверяем C40 после первой попытки, после последней попытки и после посадки. Группы пересекаются: 42 записи входят в группу из 87, их нельзя складывать. Проверяем пропуски и первый элемент списка попыток относительно его минимума. Соседние записи того же адреса в окне ±24 часа служат только диагностическими кандидатами, а не правилом исправления. Все времена в UTC; допуск и временные сдвиги не применяются.'''),
    nbformat.v4.new_code_cell('''from pathlib import Path
import runpy
import json
import pandas as pd
from IPython.display import display

ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents]
            if (p / 'europe/data/source/osn23_landings_merged.parquet.gzip').exists())
module = runpy.run_path(str(ROOT / 'europe/scripts/diagnose_c40.py'))
c40_flags, c40_coverage, c40_records, c40_pairs = module['main']()
c40_journal = json.loads((ROOT / 'europe/reports/preparation/06_c40_diagnostics.json').read_text(encoding='utf8'))
display(c40_flags)
display(pd.DataFrame(c40_journal['categories']))
print('Уникальных записей для разбора:', c40_journal['unique_review_rows'])
print('Исходник и рабочая копия неизменны:', c40_journal['source_unchanged'], c40_journal['input_unchanged'])'''),
    nbformat.v4.new_code_cell('''print('Распределение 87 записей по аэропортам и группам:')
display(c40_records.groupby(['airport', 'c40_time_category']).agg(
    rows=('GoA', 'size'), go_arounds=('GoA', 'sum'),
    min_landing_minus_c40_minutes=('landing_minus_c40_minutes', 'min'),
    max_landing_minus_c40_minutes=('landing_minus_c40_minutes', 'max')).reset_index())

print('Все 42 записи с C40 после посадки:')
c40_review_columns = ['source_row_position', 'flight_id', 'icao24', 'airport', 'landing_time',
                     'first_attempt_time', 'last_attempt_time', 'C40_CROSS_TIME',
                     'GoA', 'n_attempts', 'landing_minus_c40_minutes']
with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    display(c40_records.loc[c40_records['after_landing'], c40_review_columns])'''),
    nbformat.v4.new_code_cell('''print('Все 45 записей с C40 после первой попытки, но до посадки:')
with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    display(c40_records.loc[c40_records['after_first_attempt'] & ~c40_records['after_landing'], c40_review_columns])
print('Повторное присоединение той же отметки C40 к другому наблюдению того же адреса и аэропорта:')
with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    display(c40_pairs.loc[c40_pairs['same_airport'] & c40_pairs['same_c40_time']])
print('Это свидетельство неоднозначности сопоставления; исправления и объединение записей не выполнены.')'''),
    nbformat.v4.new_markdown_cell('''## Итог шага 7 – Правило очистки пока не применено

Полный проход подтвердил аудит 01: 87 записей имеют C40 после первой попытки. В их составе 45 отмеченных уходов с 2–3 попытками: C40 предшествует последней попытке и посадке. Такие случаи совместимы с повторным входом, но требуют траекторий для подтверждения. Эта отметка C40 не подходит как момент прогноза до первой попытки; переносить ее назад искусственно нельзя.

Другие 42 записи имеют одну попытку, GoA=False и C40 через 26 секунд – 9 минут 12 секунд после указанной посадки. Это внутреннее противоречие метаданных. В 23 случаях аэропорт – LSGG; концентрация не является основанием исключить весь аэропорт. Три записи имеют тот же C40 и координаты, что более позднее наблюдение того же адреса, аэропорта и позывного. Это может отражать сопоставление или сегментацию; определить ошибочное поле и исправить метку ухода по одной таблице нельзя.

C40 отсутствует в 1 652 382 записях, включая 7 847 уходов. Пропуски не приравниваем к противоречиям. В LTFJ C40 отсутствует во всех записях; статья объясняет отсутствие покрытия в EU-NM. Для остальных аэропортов не переносим проценты покрытия статьи на наш полный файл.

Предлагаем следующим согласованным действием добавить только диагностические флаги в отдельную рабочую копию: C40 отсутствует; C40 после первой попытки; C40 после посадки. Все записи и исходные поля сохранить. Для будущего анализа отдельно определить допустимость C40 относительно момента прогноза. Обнуление C40, исключение строк, изменение GoA, объединение сегментов и исключение аэропортов сейчас не выполняются.

Временные диагностические флаги используют сведения завершенного наблюдения и не предназначены для подачи в модель раннего прогноза. Отбор по ним может изменить состав целевых классов и требует отдельной оценки.

Подробный [разбор C40](../reports/preparation/06_c40_review.md) содержит ограничения и источники. Проверены SHA256 исходника и рабочей копии до и после проходов; изменений нет.'''),
])
nb.cells.extend([
    nbformat.v4.new_markdown_cell('''## Шаг 8 – Согласованное добавление диагностических флагов C40

Добавляем в новую копию `europe/data/working/03_c40_diagnostic_flags.parquet` только три булевых поля. Все записи и существующие значения сохраняются. Правила основаны на полном разборе шага 7 и его [первичных источниках](../reports/preparation/06_c40_review.md), а не на новых авиационных порогах.

| Поле | True означает |
| --- | --- |
| `c40_time_missing` | В C40_CROSS_TIME физический null. |
| `c40_after_first_attempt` | C40 строго позже первого элемента attempt_times. |
| `c40_after_landing` | C40 строго позже landing_time. |

Флаги не содержат null. Если C40 отсутствует, оба временных флага False: это не означает подтвержденную корректность C40. Сравнения выполняются в UTC, без допуска и искусственного сдвига времени. Первый элемент списка проверяется на равенство минимальному времени; пустые списки и null-элементы не допускаются для этого проверенного входа.

Это диагностические поля. Временные флаги используют сведения завершенного наблюдения и не являются входными признаками раннего прогноза. Источник и входная копия проверяются по SHA256 до и после. После записи сравниваем каждое прежнее значение, независимо пересчитываем флаги через NumPy и сверяем счетчики с шагом 7.'''),
    nbformat.v4.new_code_cell('''from pathlib import Path
import runpy
import json
from IPython.display import display

ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents]
            if (p / 'europe/data/source/osn23_landings_merged.parquet.gzip').exists())
module = runpy.run_path(str(ROOT / 'europe/scripts/apply_c40_flags.py'))
c40_changes, c40_combinations = module['main']()
display(c40_changes)
print('Взаимоисключающие сочетания флагов:')
display(c40_combinations)
c40_applied = json.loads((ROOT / 'europe/reports/preparation/07_c40_flags_applied.json').read_text(encoding='utf8'))
print('Новая рабочая копия:', c40_applied['output'])
print('Записей до и после:', c40_applied['rows_before'], c40_applied['rows_after'])
print('Отмеченных уходов до и после:', c40_applied['go_arounds_before'], c40_applied['go_arounds_after'])
print('Все существующие поля неизменны:', c40_applied['existing_columns_unchanged'])
print('Флаги независимо проверены для каждой строки:', c40_applied['independent_per_row_flag_validation'])'''),
    nbformat.v4.new_markdown_cell('''## Итог шага 8

Сохранены все 7 984 254 записи и 26 253 отмеченных ухода. Добавлены флаги: C40 отсутствует – 1 652 382 записи; после первой попытки – 87; после посадки – 42. Последние 42 входят в 87, а не образуют дополнительную группу. У 45 оставшихся записей с флагом после первой попытки отмечен уход; исправления этих случаев не выполнялись.

Неизменность всех прежних полей проверена полным сравнением. Исходный Parquet и предыдущая рабочая копия сохранены. Новые флаги не меняют тип самолета, время или GoA и не исключают аэропорты. [Журнал преобразования](../reports/preparation/07_c40_flags_applied.json) хранит определения, счетчики и контрольные суммы.

Следующий разбор – повторяющиеся flight_id: отличаем повтор идентификатора от дубля наблюдения и возможного разделения одного полета на сегменты. Новое правило удаления или объединения заранее не задано.'''),
])
nb.cells.extend([
    nbformat.v4.new_markdown_cell('''## Шаг 9 – Повторяющиеся flight_id

Читаем последнюю копию `03_c40_diagnostic_flags.parquet`. Сначала считаем идентификаторы по всему файлу, затем извлекаем все записи повторяющихся значений. Сравниваем UTC-даты, ICAO24, аэропорты, позывные, времена и содержательные поля. Проверяем все пары внутри каждой группы, включая возможные совпадения в один день; примерная голова файла не используется для общих выводов.

В [документации разработчиков traffic](https://traffic-viz.github.io/api_reference/traffic.core.traffic.html#traffic.core.Traffic.assign_id) идентификатор может строиться из позывного и последовательного индекса. [Flight.assign_id](https://traffic-viz.github.io/api_reference/traffic.core.flight.html#traffic.core.Flight.assign_id) описывает шаблон `{self.callsign}_{idx:>03}`. Сходство нашего формата – подсказка, а не доказательство алгоритма автора за 2019–2023 годы. Сохраненный публичный ноутбук автора читает готовый Parquet и не объясняет создание этих идентификаторов.

Повтор flight_id не равен дублю посадки. Отдельно сравниваем все поля рабочей копии без служебного индекса и набор исходных содержательных полей после нормализации. Ключ `landing_time + icao24 + airport` проверен в аудите 01; его уникальность в файле не доказывает идентичность физического полета. Все источники и текущая копия проверяются по SHA256.'''),
    nbformat.v4.new_code_cell('''from pathlib import Path
import runpy
import json
import pandas as pd
from IPython.display import display

ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents]
            if (p / 'europe/data/source/osn23_landings_merged.parquet.gzip').exists())
module = runpy.run_path(str(ROOT / 'europe/scripts/diagnose_flight_ids.py'))
id_groups, id_categories, id_examples, id_near_pairs = module['main']()
id_journal = json.loads((ROOT / 'europe/reports/preparation/08_flight_id_diagnostics.json').read_text(encoding='utf8'))
display(id_categories)
display(pd.read_csv(ROOT / 'europe/reports/preparation/08_flight_id_group_sizes.csv'))
print('Повторяющихся идентификаторов:', id_journal['repeated_identifiers'])
print('Записей в этих группах:', id_journal['rows_with_repeated_identifier'])
print('Превышение числа записей над числом идентификаторов:', id_journal['excess_identifier_rows'])
print('Точных дублей рабочей копии в группах:', id_journal['exact_working_duplicate_excess_in_repeated_groups'])
print('Групп с разными адресами:', id_journal['groups_with_multiple_aircraft_addresses'])
print('Групп с разными аэропортами:', id_journal['groups_with_multiple_airports'])
print('Пар одного адреса и аэропорта в одну UTC-дату:', id_journal['same_day_same_address_airport_pairs'])'''),
    nbformat.v4.new_code_cell('''print('Полные примеры групп повторяющихся идентификаторов:')
with pd.option_context('display.max_rows', None, 'display.max_columns', None):
    display(id_examples)
print('Записей с отмеченным уходом в повторяющихся группах:', id_journal['go_arounds_in_repeated_identifier_rows'])
print('Групп со смешанными GoA:', id_journal['groups_with_mixed_go_around_labels'])
print('Если оставить первую строку каждого flight_id, потеряем отмеченных уходов:',
      id_journal['go_arounds_lost_if_deduplicating_flight_id_keep_first_in_file_order'])
print('Это расчет последствий возможного правила; строки не удалялись.')'''),
    nbformat.v4.new_code_cell('''print('Группы с повтором даты или контекста в один день:')
display(id_groups.loc[~id_groups['date_alone_disambiguates_identifier']])
display(id_near_pairs)
print('Исходник и рабочая копия неизменны:', id_journal['source_unchanged'], id_journal['input_unchanged'])
print('Изменено / удалено строк:', id_journal['rows_changed'], id_journal['rows_deleted'])'''),
    nbformat.v4.new_markdown_cell('''## Итог шага 9

23 889 значений flight_id повторяются в 53 127 записях. Все записи внутри каждого такого идентификатора относятся к разным UTC-датам. В 22 547 группах различаются ICAO24; в 1 667 – аэропорты. Совпадение имени не связывает их в один физический полет. Среди этих записей нет точных содержательных дублей или повторов ключа посадки.

121 группа содержит ровно одну запись GoA=True и другие записи GoA=False. Удаление по flight_id может уничтожить реальные наблюдения и изменить состав целевого класса. Идентификатор нельзя использовать как единственный ключ присоединения меток, попыток или траекторий; иначе возможны связи между разными датами и бортами. Совпадение `flight_id + UTC-дата посадки` различает записи текущего файла, но не гарантирует уникальность будущих данных и не заменяет проверку траекторного сопоставления, особенно около полуночи.

Решение: все строки сохраняем, исходный flight_id не переписываем. При подготовке связей используем проверенный составной ключ наблюдения и контекст времени, адреса и аэропорта. Отдельный идентификатор записи можно добавить следующим согласованным правилом; он будет служебным, а не признаком модели. Идентификатор физического полета потребует иной проверки.

Подробный [разбор повторов](../reports/preparation/08_flight_id_review.md) содержит ограничения и первичные источники. Следующий диагностический этап – записи с большим числом попыток; они не считаются ошибочными или учебными автоматически.'''),
])
nb.cells.extend([
    nbformat.v4.new_markdown_cell('''## Шаг 10 – записи с большим числом попыток

Проверяем весь файл, затем отдельно разбираем n_attempts > 2. Этот порог служит диагностике, а не авиационной норме. [Описание набора](https://zenodo.org/records/10210178) определяет число обнаруженных подходов; GoA отмечает наличие хотя бы одного ухода. Число n_attempts - 1 не принимаем за подтвержденное число эксплуатационных уходов. ILS содержит обозначения ВПП, а не доказательство фактического применения системы ILS.

Сохраняем времена, интервалы и исходные строки. Длительность между первой и последней попытками не является полной длительностью полета. Проверяем контрольные суммы исходника и рабочей копии.'''),
    nbformat.v4.new_code_cell('''from pathlib import Path
import json
import runpy
import pandas as pd
region = Path.cwd().parent if Path.cwd().name == 'notebooks' else Path.cwd() / 'europe'
attempt_module = runpy.run_path(str(region / 'scripts/diagnose_attempts.py'))
attempt_counts, attempt_airports, high_attempts, attempt_intervals = attempt_module['main']()
attempt_journal = json.loads((region / 'reports/preparation/09_attempts_diagnostics.json').read_text(encoding='utf8'))
display(attempt_counts)
display(pd.Series(attempt_journal['structural_flags'], name='Число противоречий'))'''),
    nbformat.v4.new_code_cell('''display(attempt_airports)
with pd.option_context('display.max_columns', None):
    display(high_attempts.nlargest(5, 'n_attempts'))
print('Медиана интервала, минуты:', attempt_journal['high_attempt_gap_minutes']['median'])
print('Медиана окна первой–последней попытки, минуты:', attempt_journal['high_attempt_span_minutes']['median'])'''),
    nbformat.v4.new_code_cell('''with pd.option_context('display.max_columns', None):
    display(high_attempts.nsmallest(5, 'min_attempt_gap_minutes'))
    display(pd.read_csv(region / 'reports/preparation/09_high_attempt_examples.csv'))
print('C40 после первой попытки:', attempt_journal['high_attempt_c40_after_first_rows'])
print('Исходник и рабочая копия неизменны:', attempt_journal['source_unchanged'], attempt_journal['input_unchanged'])
print('Изменено / удалено строк:', attempt_journal['rows_changed'], attempt_journal['rows_deleted'])'''),
    nbformat.v4.new_markdown_cell('''## Итог шага 10

Выделено 1 052 записи (4,01% строк GoA=True), максимум 33 попытки. Противоречий длины списков, порядка времен и связи GoA с числом попыток не обнаружено. Это структурная согласованность, а не подтверждение каждой метки по траектории.

У записи с 33 попытками указан DCFME. [Оператор FCS](https://fcs.aero/en/about-fcs/) описывает D-CFME как самолет летных проверок. Эта информация поддерживает гипотезу о специальном назначении, но не устанавливает задачу конкретного полета 2023 года. Другие записи автоматически учебными или проверочными не считаем. Текущая документация traffic не доказывает точную историческую конфигурацию алгоритма.

Сохраняем все записи. Число попыток, итоговые времена и последовательность ВПП известны после события и требуют исключения из предсобытийных признаков. Подробности и отложенные решения: [разбор](../reports/preparation/09_attempts_review.md). Следующий шаг – свести результаты диагностики в журнал правил и определить состав наблюдения для моделирования до очистки аэропортов и строк.'''),
])
# Комментарии хранятся в генераторе, чтобы сохраняться при пересборке ноутбука.
cell_comments = {
    1: [('from pathlib', 'Подключаем библиотеки для чтения Parquet, таблиц и проверки целостности.'),
        ('ROOT =', 'Находим корень проекта независимо от папки запуска ноутбука.'),
        ('def source_hash', 'Считаем SHA256 блоками: весь исходный файл не нужно загружать в память.'),
        ('before =', 'Сверяем источник и число строк с уже выполненным аудитом.')],
    3: [('fields =', 'Выбираем строковые поля верхнего уровня; вложенные списки здесь не проверяем.'),
        ('offset =', 'Читаем по 100 000 строк и сохраняем глобальную позицию примеров.'),
        ("        masks =", 'Различаем null, точную пустую строку и строку из пробелов.', 8),
        ('            remaining =', 'Ограничиваем примеры тремя строками на поле и вид пропуска.', 12),
        ('assert offset', 'Проверяем полный охват файла и рассчитываем доли от всех строк.')],
    4: [('display(', 'Показываем исходные представления пропусков для обсуждения правил.')],
    6: [('import pyarrow', 'Подключаем операции Arrow для преобразования без потери типов.'),
        ('approved_fields =', 'Нормализуем только три согласованных поля; создаем отдельную рабочую копию.'),
        ('with pq.ParquetWriter', 'Сначала пишем временный файл; для каждой замены сохраняем обратимый флаг.'),
        ('candidate.close', 'После всех проверок публикуем рабочую копию заменой временного файла.'),
        ('REPORT =', 'Сохраняем счетчики и контрольные суммы в журнал воспроизводимости.')],
    8: [('from pathlib', 'Готовим диагностику классификаций самолетов в нормализованной копии.'),
        ('working_before =', 'Проверяем, что вход соответствует журналу предыдущего шага.'),
        ('for batch', 'Сканируем всю таблицу и собираем сочетания кодов, метки и примеры.'),
        ('    valid_shape =', 'Шаблон выделяет значения для разбора, но сам по себе не доказывает ошибку.', 4),
        ('assert rows', 'Проверяем охват и неизменность входа, затем строим диагностические таблицы.'),
        ('codes.to_csv', 'Сохраняем отчеты; значения в рабочей базе не исправляем.')],
    9: [("print('Пары", 'Сопоставляем подозрительные коды с typecode и распределением по аэропортам.'),
        ("print('Другие", 'Внутренние сочетания служат сравнением, а не официальным справочником.'),
        ("print('Примеры", 'Показываем ограниченное число исходных строк для ручной проверки.')],
    12: [('from pathlib', 'Читаем справочные предложения без автоматического применения.'),
         ('proposals =', 'Сохраняем текстовые значения CSV и сверяем размер набора предложений.'),
         ('with (ROOT', 'Контрольная сумма подтверждает, что рабочая копия не менялась.'),
         ('with pd.option_context', 'Показываем источник, описание и статус каждого предложения.')],
    15: [('from pathlib', 'Запускаем согласованное справочное обогащение отдельным скриптом.'),
         ('module =', 'run_path загружает функции скрипта; main выполняет преобразование и проверки.'),
         ('journal =', 'Показываем журнал: сохранность строк, меток и всех существующих полей.')],
    18: [('from pathlib', 'Исследуем оставшиеся неразрешенные классификации самолетов.'),
         ('module =', 'Скрипт ищет другие сочетания тех же адресов во всей базе, не исправляя ее.'),
         ('display(summary)', 'Сводка и история помогают оценить кандидатов на дальнейшую проверку.')],
    19: [('assessment =', 'Читаем семь отложенных оценок; все 519 строк пока остаются без исправлений.'),
         ('assert len', 'Проверяем ожидаемый охват и отсутствие примененных предложений.')],
    22: [('from pathlib', 'Проверяем положение времени C40 относительно попыток и посадки.'),
         ('module =', 'Диагностика сканирует всю базу и сохраняет категории, строки и пары для разбора.'),
         ('display(c40_flags)', 'Показываем счетчики и подтверждение неизменности входных файлов.')],
    23: [("print('Распределение", 'Сводим записи по аэропортам и временным категориям C40.'),
         ("print('Все 42", 'Выводим все случаи времени C40 после посадки с ключевыми временами.')],
    24: [("print('Все 45", 'Выделяем C40 после первой попытки, исключая случаи после посадки.'),
         ("print('Повторное", 'Проверяем повторное использование одной отметки у того же борта и аэропорта.')],
    27: [('from pathlib', 'Создаем согласованную копию с тремя диагностическими флагами C40.'),
         ('module =', 'Скрипт сохраняет исходные поля и независимо проверяет флаги каждой строки.'),
         ('c40_applied =', 'Читаем журнал преобразования и подтверждаем сохранение строк и меток.')],
    30: [('from pathlib', 'Исследуем повторяющиеся flight_id, не удаляя наблюдения.'),
         ('module =', 'Скрипт сравнивает даты, адреса, аэропорты и точные значения внутри групп.'),
         ('display(id_categories)', 'Показываем размеры групп и отличаем коллизии имен от точных дублей.')],
    31: [("print('Полные", 'Разбираем примеры коллизий flight_id с полным контекстом.'),
         ("print('Если оставить", 'Оцениваем потери целевого класса при гипотетическом удалении по имени.')],
    32: [("print('Группы", 'Проверяем, различает ли UTC-дата строки внутри каждого повторяющегося имени.'),
         ('display(id_near_pairs)', 'Показываем близкие пары и подтверждаем отсутствие изменений базы.')],
    35: [('from pathlib', 'Готовим разбор нескольких попыток и интервальных характеристик.'),
         ('attempt_module =', 'Скрипт проверяет все строки, сохраняет группу n_attempts > 2 и контрольные суммы.'),
         ('display(attempt_counts)', 'Сравниваем число попыток и показываем структурные противоречия.')],
    36: [('display(attempt_airports)', 'Показываем распределение по аэропортам и пять максимальных чисел попыток.'),
         ("print('Медиана интервала", 'Различаем медиану отдельных промежутков и медиану окна по наблюдениям.')],
    37: [('with pd.option_context', 'Короткие промежутки и полные последовательности требуют траекторного разбора.'),
         ("print('C40", 'Проверяем пересечение с проблемами C40 и сохранность исходных данных.')],
}
assert set(cell_comments) == {i for i, cell in enumerate(nb.cells) if cell.cell_type == 'code'}
for index, explanations in cell_comments.items():
    cell = nb.cells[index]
    original_tree = ast.dump(ast.parse(cell.source), include_attributes=False)
    lines = cell.source.splitlines()
    for entry in explanations:
        anchor, explanation = entry[:2]
        indent = entry[2] if len(entry) == 3 else 0
        positions = [i for i, line in enumerate(lines) if line.startswith(anchor)]
        assert len(positions) == 1, (index, anchor)
        lines.insert(positions[0], ' ' * indent + '# ' + explanation.rstrip('.'))
    cell.source = '\n'.join(lines)
    assert ast.dump(ast.parse(cell.source), include_attributes=False) == original_tree
nb.metadata = {'kernelspec': {'name': 'python3', 'display_name': 'Python (окружение проекта)',
                             'language': 'python'}, 'language_info': {'name': 'python'}}
for cell in nb.cells:
    if cell.cell_type == 'code':
        compile(cell.source, '<preparation-cell>', 'exec')
nbformat.validate(nb)
path = region / 'notebooks/03_data_preparation/archive/03_data_preparation.ipynb'
path.parent.mkdir(parents=True, exist_ok=True)
if path.exists():
    previous = nbformat.read(path, as_version=4)
    for cell in nb.cells:
        for old in previous.cells:
            unchanged_code = (cell.cell_type == old.cell_type == 'code'
                              and ast.dump(ast.parse(cell.source), include_attributes=False)
                              == ast.dump(ast.parse(old.source), include_attributes=False))
            if cell.cell_type == old.cell_type and (cell.source == old.source or unchanged_code):
                cell.id = old.id
                if cell.cell_type == 'code':
                    cell.outputs = old.outputs
                    cell.execution_count = old.execution_count
                break
nbformat.write(nb, path)
print('Created preparation notebook:', len(nb.cells), 'cells')
