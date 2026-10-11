"""Execute architecture checks without fitting classifiers."""
from pathlib import Path
import nbformat as nb
from nbclient import NotebookClient
ROOT=Path(__file__).resolve().parents[3]
BASE=ROOT/'research/boosting-weather-comparison'
n=nb.v4.new_notebook(cells=[nb.v4.new_markdown_cell('''# 00. Общая архитектура и входы нового этапа

Методика будущего обучения: бинарная классификация GoA бустингом деревьев.
Здесь обучение не выполняется. Выделены общий пакет src/approach_risk и
реестр configs/datasets.json; настройки CatBoost находятся в отдельном конфиге.
Старые исследования сохранены, новые используют общий API. Данные пока
физически остаются по прежним путям, указанным в реестре. Модели новых
запусков будут сохранены в artifacts вне Git.

Проверяем все 93 хеша таблиц, размеры частей, метки, эквивалентность признаков,
граничные случаи метрик и воспроизведение сохраненных test-метрик через
выделенный код. Старый test не используется для выбора новых настроек.
'''),nb.v4.new_code_cell('''from pathlib import Path
import runpy,json
import pandas as pd
from IPython.display import display
ROOT=Path.cwd()
runpy.run_path(str(ROOT/'scripts/verify_shared_architecture.py'),run_name='__main__')
runpy.run_path(str(ROOT/'research/boosting-weather-comparison/scripts/check_inputs.py'),run_name='__main__')
report=json.loads((ROOT/'research/boosting-weather-comparison/reports/00_architecture_verification.json').read_text())
assert report['verified'] and not report['fit_executed']
display(pd.DataFrame({'rows':report['split_rows'],'GoA':report['labels']}))
'''),nb.v4.new_markdown_cell('''## Точка продолжения

Общий загрузчик → признаки → классификатор → общая оценка → отчет исследования.
Проверки перехода пройдены. CatBoost пока не установлен и модели нового этапа
не обучены. Следующий шаг – проверка среды и обучение кандидатов только по
train/validation. Завершенные модели, исходные данные и архивы не менялись.
''')],metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}})
p=BASE/'notebooks/00_architecture.ipynb';nb.write(n,p)
try:NotebookClient(n,timeout=600,resources={'metadata':{'path':str(ROOT)}}).execute()
finally:nb.write(n,p)
print('Executed',p)
