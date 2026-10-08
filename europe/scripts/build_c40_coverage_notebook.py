"""Создание шага 04_03 с отдельной ячейкой для каждого графика."""
from pathlib import Path
import nbformat

region = Path(__file__).resolve().parents[1]
path = region / 'notebooks/c40/04_03_c40_coverage.ipynb'
if path.exists():
    raise FileExistsError('Редактируйте существующий ноутбук непосредственно')
cells = [nbformat.v4.new_markdown_cell('''# 04_03 – Покрытие и временная согласованность C40

Исследуем все наблюдаемые полеты, включая специальные; прогноз предполагается при входе C40. Агрегаты относятся к записанным меткам, а не к независимо подтвержденным эксплуатационным событиям. Отбор не применяем.

Текущая таблица позволяет проверить порядок времен и возможное повторное присоединение, но не подтверждает однозначно правильный вход C40 по полной траектории. Геометрия и исторические опорные точки аэропортов отдельно не проверены. Сегмент перевозок не равен подтвержденному назначению полета. Полные группы с числом строк и меток сохраняются в CSV.'''),
nbformat.v4.new_code_cell('''# Находим проект и подключаем анализ с проверкой входной копии
from pathlib import Path
import sys
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
ROOT = next(p for p in [Path.cwd(), *Path.cwd().parents]
            if (p / 'europe/data/source/osn23_landings_merged.parquet.gzip').exists())
sys.path.insert(0,str(ROOT / 'europe/scripts'))
import audit_c40_coverage as analysis

# Полный проход сохраняет агрегаты, временные проверки и контрольную сумму
coverage = analysis.main()
display(coverage['airport'])'''),
nbformat.v4.new_markdown_cell('''## Пропуски по аэропортам

Знаменатель каждой доли – все наблюдения данного аэропорта. Нулевое покрытие C40 может ограничивать эксперимент в точке C40, но само по себе не является основанием удалять аэропорт из исходной базы.'''),
nbformat.v4.new_code_cell('''# Рисуем долю отсутствующего времени C40 по каждому аэропорту
analysis.missing_chart(coverage['airport'],'airport','04_03_airport_missing',
                       'Отсутствующее время C40 по аэропортам')
plt.show()'''),
nbformat.v4.new_code_cell('''# Сравниваем покрытие строк и положительных меток по UTC-месяцам
analysis.monthly_chart(coverage['month'])
plt.show()'''),
nbformat.v4.new_code_cell('''# Смотрим пропуски по сегментам, сохраняя отдельную группу null
analysis.missing_chart(coverage['market_segment'],'market_segment','04_03_segment_missing',
                       'Отсутствующее время C40 по сегментам источника')
plt.show()
display(coverage['market_segment'])'''),
nbformat.v4.new_markdown_cell('''## Ограничения и следующий шаг

Данные сохранены, автоматически аэропорты и специальные полеты не исключаем. Отсутствующее C40 не восстанавливаем по итоговым временам. Различие частот между группами не доказывает влияние C40 на риск.

Дальше обсуждаем эксперимент в точке C40: определяем допустимость наблюдений и что делать с отсутствующими/неоднозначными отметками. Траекторное сопоставление, дополнительные источники и историческая геометрия требуют отдельной работы.

[Разбор результатов](../../reports/analytical_sample/04_03_review.md), [предыдущая карта признаков](04_02_feature_availability.ipynb), [разбор сопоставления C40](../../reports/preparation/06_c40_review.md).''')]
notebook = nbformat.v4.new_notebook(cells=cells,metadata={
    'kernelspec': {'name':'python3','display_name':'Python (окружение проекта)','language':'python'},
    'language_info':{'name':'python'}})
nbformat.validate(notebook)
nbformat.write(notebook,path)
print('Created:',path)
