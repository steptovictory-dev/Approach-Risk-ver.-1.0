"""Создание самостоятельного шага 04_02."""
from pathlib import Path
import nbformat

region = Path(__file__).resolve().parents[1]
path = region / 'notebooks/c40/04_02_feature_availability.ipynb'
if path.exists():
    raise FileExistsError('Редактируйте существующий ноутбук непосредственно')
cells = [
    nbformat.v4.new_markdown_cell('''# 04_02 – Доступность признаков и утечка исхода

Рассматриваем прогноз до первой наблюдаемой попытки. Согласованный момент прогноза – вход C40; его сопоставление с первой попыткой требует проверки. [Описание авторов набора](https://zenodo.org/records/10210178) проверено 08.10.2026. Названия и физические типы берем из последней рабочей копии; интерпретации доступности – наши исследовательские решения с явно указанными ограничениями.

Полнота поля не доказывает доступность до события. Целевой аэропорт нельзя автоматически заменить аэропортом конечной посадки. Справочник, проверенный в 2026 году, не становится историческим справочником 2019–2023.'''),
    nbformat.v4.new_code_cell('''# Находим проект и подключаем диагностику без зависимости от предыдущего ядра
from pathlib import Path
import runpy
import pandas as pd
from IPython.display import display
ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents]
            if (p / 'europe/data/source/osn23_landings_merged.parquet.gzip').exists())
module = runpy.run_path(str(ROOT / 'europe/scripts/audit_feature_availability.py'))

# Проверяем все строки и сохраняем карту полей и счетчики без фильтрации базы
feature_map, c40_coverage = module['main']()
with pd.option_context('display.max_rows', None, 'display.max_colwidth', 100):
    display(feature_map)'''),
    nbformat.v4.new_markdown_cell('''## Роли полей

Разделяем целевую метку, сведения об исходе, идентификаторы, следы подготовки и условные кандидаты. Для каждого кандидата указана недостающая проверка. Категория «условно доступно» не означает разрешение включить поле в модель. Флаги C40 после первой попытки/посадки используют будущие времена и исключаются из раннего прогноза.'''),
    nbformat.v4.new_code_cell('''# Сводим роли и отдельно показываем поля с информацией об исходе
display(feature_map.groupby(['role', 'availability']).size().rename('fields').reset_index())
with pd.option_context('display.max_colwidth', 100):
    display(feature_map.loc[feature_map['role'].isin(['outcome', 'outcome_diagnostic']),
                           ['field', 'reason']])'''),
    nbformat.v4.new_markdown_cell('''## Что изменит требование C40

Считаем последствия нескольких диагностических условий на всей базе. Группы пересекаются и не должны суммироваться. «Не позже первой попытки» включает равенство времен и не гарантирует запас времени для прогноза. Полнота четырех полей и конечность чисел не доказывают правильность единиц, координат или присоединения.

Эти условия используют ретроспективную информацию для аудита. Их нельзя подавать в модель как признаки; исключение строк по ним может изменить состав целевого класса.'''),
    nbformat.v4.new_code_cell('''# Сравниваем покрытие всей базы и отдельно покрытие положительного класса
with pd.option_context('display.max_columns', None):
    display(c40_coverage)

# Показываем условных кандидатов вместе с пропусками и необходимыми проверками
candidate_roles = ['context_candidate', 'metadata_candidate', 'event_candidate', 'reference_candidate']
with pd.option_context('display.max_rows', None, 'display.max_colwidth', 100):
    display(feature_map.loc[feature_map['role'].isin(candidate_roles),
                           ['field', 'null_percent', 'availability', 'required_verification']])'''),
    nbformat.v4.new_markdown_cell('''## Решение шага 04_02

Карта охватывает все физические поля. Данные не изменены, фильтрация C40 не применялась и окончательный набор признаков не утвержден. Погодных полей и полных траекторий в текущем файле нет; их наличие нельзя предполагать.

Далее 04_03 – оценка исследовательской популяции и покрытия по аэропортам/времени. Согласовано: все наблюдаемые полеты, включая специальные; прогноз в точке входа C40. Проверяем сопоставление и покрытие по группам. Только после этого оцениваем конкретные правила отбора и создаем аналитическую копию.

[Карта полей и счетчики](../../reports/analytical_sample/04_02_review.md), [постановка задачи](04_01_task_definition.ipynb).'''),
]
notebook = nbformat.v4.new_notebook(cells=cells, metadata={
    'kernelspec': {'name': 'python3', 'display_name': 'Python (окружение проекта)', 'language': 'python'},
    'language_info': {'name': 'python'}})
nbformat.validate(notebook)
nbformat.write(notebook, path)
print('Created:', path)
