"""Создание шага 04_06 без заявления о завершенной независимой проверке."""
from pathlib import Path
import nbformat

region=Path(__file__).resolve().parents[1]
path=region/'notebooks/c40/04_06_independent_check.ipynb'
if path.exists():
    raise FileExistsError('Редактируйте существующий ноутбук непосредственно')
cells=[nbformat.v4.new_markdown_cell('''# 04_06 – Подготовка независимой проверки трех пар

Проверка исходных измерений пока не завершена. Публичный архив OpenSky не содержит нужные интервалы: для 02.01.2019 и 30.04.2019 целевых снимков нет; для 12.08.2019 доступны только часы 00–12 UTC, а событие EDDF относится к 16–17 UTC. Результат проверки каталога сохранен отдельно.

Из локального validation_sample.zip извлекаем точки трех ранних сегментов. Это дополнительные материалы исходного исследования, не независимые измерения другой системы или заново полученная траектория OpenSky. Идентификация в HTML основана на flight_id; ICAO24 в извлеченных точках отсутствует. Высоту сохраняем в исходных значениях без предположения об единице.

Подготавливаем три ограниченных SQL-запроса к историческому Trino. [Официальная документация](https://openskynetwork.github.io/opensky-api/trino.html) требует одобренного доступа и ограничения по часовым разделам.'''),
nbformat.v4.new_code_cell('''# Находим проект и извлекаем уже имеющиеся материалы без повторной загрузки архива
from pathlib import Path
import sys
import matplotlib.pyplot as plt
from IPython.display import display
ROOT=next(p for p in [Path.cwd(),*Path.cwd().parents]
          if (p/'europe/data/source/osn23_landings_merged.parquet.gzip').exists())
sys.path.insert(0,str(ROOT/'europe/scripts'))
import prepare_independent_checks as analysis

# Сохраняем происхождение точек и отдельный статус независимой проверки
cases=analysis.main()
display(cases)''')]
for airport in ['EDDM','LFPG','EDDF']:
    cells.append(nbformat.v4.new_code_cell(f'''# Сравниваем опубликованные точки {airport} с временами обеих записей
analysis.plot_case('{airport}')
plt.show()'''))
cells.append(nbformat.v4.new_markdown_cell('''## Вывод и дальнейшие действия

Во всех трех опубликованных ранних сегментах последняя точка имеет ненулевое значение altitude и совпадает с landing_time ранней записи. Это ставит под сомнение интерпретацию landing_time как времени касания ВПП, но не удостоверяет истинный момент посадки. Высота не приравнивается к высоте над аэродромом, наземная фаза не восстановлена.

Для независимой проверки необходимо получить state vectors по подготовленным запросам, проверить свежесть позиции и контакта, пробелы наблюдения, снижение/набор, onground и движение после предполагаемой посадки. Отдельно проверить вход C40. Отсутствие наблюдения не доказывает отсутствие события.

До получения независимых данных обе записи каждой пары сохраняются. [Статус и источники](../../reports/analytical_sample/04_06_review.md).'''))
nb=nbformat.v4.new_notebook(cells=cells,metadata={'kernelspec':{'name':'python3','display_name':'Python (окружение проекта)','language':'python'},'language_info':{'name':'python'}})
nbformat.validate(nb)
nbformat.write(nb,path)
print('Created:',path)
