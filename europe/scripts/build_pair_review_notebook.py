"""Создание шага 04_05: разбор всех пар C40."""
from pathlib import Path
import nbformat

region=Path(__file__).resolve().parents[1]
path=region/'notebooks/c40/04_05_c40_pair_review.ipynb'
if path.exists():
    raise FileExistsError('Редактируйте существующий ноутбук непосредственно')
cells=[nbformat.v4.new_markdown_cell('''# 04_05 – Разбор 147 пар повторного C40

Сравниваем все 294 исходные строки, включая списки времен и ВПП, метки, идентификаторы и справочные поля. Исходные строки извлекаются по позициям из рабочей копии Parquet; CSV используется для просмотра результатов. Решения об исправлении или удалении пока не принимаем.'''),
nbformat.v4.new_code_cell('''# Подключаем разбор полных исходных записей в отдельном ядре
from pathlib import Path
import sys
import pandas as pd
from IPython.display import display
ROOT=next(p for p in [Path.cwd(),*Path.cwd().parents]
          if (p/'europe/data/source/osn23_landings_merged.parquet.gzip').exists())
sys.path.insert(0,str(ROOT/'europe/scripts'))
import review_c40_pairs as analysis

# Проверяем контрольную сумму и сравниваем каждый исходный столбец внутри пары
pairs,differences=analysis.main()
display(pairs.groupby(['airport','label_pattern']).size().rename('pairs').reset_index())'''),
nbformat.v4.new_code_cell('''# Показываем все 147 пар с временами и исходными метками
columns=['pair_id','airport','early_flight_id','late_flight_id','early_landing','late_landing',
         'landing_gap_minutes','early_n_attempts','late_n_attempts','early_GoA','late_GoA',
         'early_c40_to_first_minutes','late_c40_to_first_minutes','same_typecode','same_ILS']
with pd.option_context('display.max_rows',None,'display.max_columns',None):
    display(pairs[columns])'''),
nbformat.v4.new_code_cell('''# Отдельно разбираем пары, затрагивающие положительную целевую метку
with pd.option_context('display.max_rows',None,'display.max_columns',None):
    display(pairs.loc[pairs.label_pattern!='both_false'])

# Различия справочных полей не приравниваем к смене физического самолета
display(differences.groupby('field').size().rename('pairs_with_difference').reset_index())
display(differences.loc[differences.field=='typecode'])'''),
nbformat.v4.new_markdown_cell('''## Решение

Общие позывной, регистрация, геометрия C40 и соседние суффиксы идентификаторов поддерживают гипотезу связанных сегментов обработки. Они не доказывают истинный момент посадки или физическую идентичность полета. Удаление первой записи может потерять положительные метки; выбор второй записи автоматически не обоснован.

Для подтверждения нужны независимые траектории и правила выделения сегментов. [Репозиторий авторов](https://github.com/figuetbe/OSN23-GoA) прямо указывает, что использованные траектории не включены в него. В текущей базе их также нет.

Все записи сохранены; C40 и GoA не переписаны. [Подробный разбор](../../reports/analytical_sample/04_05_review.md), [поиск пар](04_04_c40_matching.ipynb).''')]
nb=nbformat.v4.new_notebook(cells=cells,metadata={'kernelspec':{'name':'python3','display_name':'Python (окружение проекта)','language':'python'},'language_info':{'name':'python'}})
nbformat.validate(nb)
nbformat.write(nb,path)
print('Created:',path)
