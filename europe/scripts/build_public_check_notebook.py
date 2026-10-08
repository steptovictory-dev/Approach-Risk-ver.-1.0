"""Создание шага 04_07 для открытой проверки одной пары."""
from pathlib import Path
import nbformat

region=Path(__file__).resolve().parents[1]
path=region/'notebooks/04_с40/04_07_public_pair_check.ipynb'
if path.exists():
    raise FileExistsError('Редактируйте существующий ноутбук непосредственно')
cells=[nbformat.v4.new_markdown_cell('''# 04_07 – Проверка пары по открытым измерениям

Работаем без Trino и аффилиации. Пара 134: CES773, ICAO24 780b33, LFPG, 04.03.2019. Выбрана по доступности публичного снимка, а не случайно; вывод не распространяется автоматически на остальные 146 пар. Обе исходные метки GoA=False.

Два почасовых снимка OpenSky 04–05 UTC уже сохранены. Источник независим от производной таблицы посадок, но использует ту же сеть измерений OpenSky. Сохраняем все исходные значения, включая пропуски и выбросы. [Описание публичного набора](https://opensky-network.org/data/scientific), [определения полей](https://s3.opensky-network.org/data-samples/states/README.txt).'''),
nbformat.v4.new_code_cell('''# Загружаем уже сохраненные публичные данные, не скачивая их повторно
from pathlib import Path
import sys
import matplotlib.pyplot as plt
from IPython.display import display
ROOT=next(p for p in [Path.cwd(),*Path.cwd().parents]
          if (p/'europe/data/source/osn23_landings_merged.parquet.gzip').exists())
sys.path.insert(0,str(ROOT/'europe/scripts'))
import check_public_pair as analysis

# Проверяем происхождение и свежесть точек, затем сопоставляем времена записи
raw,event_measurements=analysis.main()
display(event_measurements)'''),
nbformat.v4.new_code_cell('''# Показываем снижение, набор высоты и наземные сообщения относительно двух времен
analysis.plot(raw)
plt.show()'''),
nbformat.v4.new_markdown_cell('''## Вывод

В 04:53:00, точно в момент раннего landing_time, свежие измерения показывают onground=False, геометрическую высоту 1 524 м и положительную вертикальную скорость 4,552 м/с. Такое наблюдение не соответствует касанию ВПП в этот момент. После него видны продолжение движения и последующее снижение; первый свежий onground=True – 05:13:50, около позднего landing_time 05:13:31.

Точное время касания и эксплуатационное число уходов не установлены. Ненулевая высота не интерпретируется как высота над аэродромом. Найдено отдельное аномальное значение барометрической высоты около перехода на наземную фазу; оно сохранено и не используется для вывода о наборе.

Раннюю запись нельзя считать независимой подтвержденной посадкой только по landing_time. Поздняя запись согласуется с прибытием, но это не разрешает автоматически удалить раннюю или исправить GoA. Геометрическое совпадение C40 с границей 40 NM в этом шаге не проверено. [Отчет](../../reports/analytical_sample/04_07_review.md).''')]
nb=nbformat.v4.new_notebook(cells=cells,metadata={'kernelspec':{'name':'python3','display_name':'Python (окружение проекта)','language':'python'},'language_info':{'name':'python'}})
nbformat.validate(nb)
nbformat.write(nb,path)
print('Created:',path)
