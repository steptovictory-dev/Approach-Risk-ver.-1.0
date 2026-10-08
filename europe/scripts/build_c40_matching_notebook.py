"""Самостоятельный шаг проверки сопоставления C40."""
from pathlib import Path
import nbformat

region = Path(__file__).resolve().parents[1]
path = region / 'notebooks/c40/04_04_c40_matching.ipynb'
if path.exists():
    raise FileExistsError('Редактируйте существующий ноутбук непосредственно')
cells = [nbformat.v4.new_markdown_cell('''# 04_04 – Проверка сопоставления C40 перед отбором

Согласовано сначала дополнительно проверить сопоставление. Правило заполненного C40 до первой попытки пока не применяем. Предварительный расчет сохранен отдельно; аналитические копии не созданы.

Проверяем весь файл на повторное использование одной отметки C40 у одного адреса и аэропорта. Разные посадки с одинаковым C40 – кандидаты на разбор, не доказанные дубли и не основание слияния. Дополнительно оцениваем неоднозначность контекста ICAO24/аэропорт/UTC-дата посадки среди строк с известным C40. Дата около полуночи не гарантирует корректность связи.'''),
nbformat.v4.new_code_cell('''# Подключаем полный анализ в отдельном ядре
from pathlib import Path
import sys
import json
import pandas as pd
from IPython.display import display
ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents]
            if (p / 'europe/data/source/osn23_landings_merged.parquet.gzip').exists())
sys.path.insert(0,str(ROOT / 'europe/scripts'))
import audit_c40_matching as analysis

# Сохраняем группы и проекцию для ручной проверки без изменения меток и времен
c40_groups,c40_records,c40_contexts = analysis.main()
display(c40_groups)'''),
nbformat.v4.new_code_cell('''# Показываем временные связи повторно использованных отметок
with pd.option_context('display.max_rows',100,'display.max_columns',None):
    display(c40_records.sort_values(['icao24','airport','C40_CROSS_TIME','landing_time']).head(100))
display(c40_contexts.head(30))'''),
nbformat.v4.new_markdown_cell('''## Что установлено и что остается неизвестным

Табличная проверка выявляет неоднозначные связи, но не удостоверяет истинный полет, первый/последний вход в ASMA или историческую геометрию. Для подтверждения нужна независимая траектория или исходная запись EUROCONTROL с правилами присоединения.

Нельзя исправлять C40 выбором ближайшей посадки только по этой таблице: это будет новая гипотеза, а не восстановленный факт. Временные флаги остаются диагностическими. До разбора не применяем правило отбора и сохраняем все полеты.

[Подробный разбор](../../reports/analytical_sample/04_04_review.md), [предыдущая проверка временных аномалий](../../reports/preparation/06_c40_review.md).''')]
nb = nbformat.v4.new_notebook(cells=cells,metadata={'kernelspec':{'name':'python3','display_name':'Python (окружение проекта)','language':'python'},'language_info':{'name':'python'}})
nbformat.validate(nb)
nbformat.write(nb,path)
print('Created:',path)
