"""Диагностика зафиксированной модели без настройки по test."""
from pathlib import Path
import sys
import nbformat as nbf

STUDY = Path(__file__).resolve().parents[1]


def main():
    cells = []
    def md(s): cells.append(nbf.v4.new_markdown_cell(s))
    def code(s): cells.append(nbf.v4.new_code_cell(s))
    md('''# 09 · Где обучаемая поправка помогает и где ошибается

Анализируем сохраненные результаты 08. **Модель, признаки, регуляризацию и разбиение не меняем**. Главный разрез – два предварительных test-рейса. Положительное улучшение означает уменьшение ошибки; отрицательное – ухудшение.

Сравниваем парные ошибки на одних и тех же целях. Окна перекрываются: отдельные строки не являются независимыми испытаниями. Группировки ниже описательные, без утверждений о статистической значимости.''')
    code('''# Читаем ошибки и объединяем методы по идентичным задачам восстановления
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython import get_ipython
from IPython.display import display, Markdown
if get_ipython() is not None:
    get_ipython().run_line_magic('matplotlib','inline')
root = next(p for p in [Path.cwd(),*Path.cwd().parents] if (p/'research/c40_geometry').is_dir())
study = root/'research/c40_geometry'
plt.rcParams.update({'figure.dpi':120,'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
model_path = study/'data/public_pilot/08_ridge_model.npz'
model_sha = hashlib.sha256(model_path.read_bytes()).hexdigest()
errors = pd.read_csv(study/'data/public_pilot/08_ml_errors.csv')
key = ['part','flight_id','gap_s','left_time','target_time']
paired = errors.pivot(index=key,columns='method',values='error_m').reset_index()
assert paired[['linear_lat_lon','ridge_correction']].notna().all().all()
paired['improvement_m'] = paired.linear_lat_lon-paired.ridge_correction
paired['worse'] = paired.improvement_m < 0
test = paired[paired.part.eq('test')].copy()
display(test.groupby('flight_id').agg(tasks=('worse','size'),
    fraction_worse=('worse','mean'),linear_mean_m=('linear_lat_lon','mean'),
    model_mean_m=('ridge_correction','mean'),mean_improvement_m=('improvement_m','mean')).round(3))''')
    md('''## 1. Ошибка при разной длине пропуска

Показываем среднюю ошибку и долю целей, где модель хуже линейного метода. Сначала считаем показатели по каждому рейсу, затем даем рейсам одинаковый вес. Хороший общий результат не означает улучшения каждой точки.''')
    code('''# Агрегируем парные результаты по рейсу и длине пропуска
flight_gap = test.groupby(['flight_id','gap_s']).agg(
    linear_mean_m=('linear_lat_lon','mean'),model_mean_m=('ridge_correction','mean'),
    fraction_worse=('worse','mean'),mean_improvement_m=('improvement_m','mean')).reset_index()
gap = flight_gap.groupby('gap_s').mean(numeric_only=True)
display(flight_gap.round(3))
fig, axes = plt.subplots(1,2,figsize=(12,4))
axes[0].plot(gap.index,gap.linear_mean_m,'o-',color='#00897B',label='Линейный метод')
axes[0].plot(gap.index,gap.model_mean_m,'o-',color='#D65A4A',label='Модель')
axes[0].set(xlabel='Интервал, с',ylabel='Средняя ошибка, м',title='Среднее по рейсам')
axes[0].legend()
axes[1].bar(gap.index,100*gap.fraction_worse,width=8,color='#D65A4A')
axes[1].set(xlabel='Интервал, с',ylabel='Целей с ухудшением, %',title='Доля ухудшений',ylim=(0,100),xticks=gap.index)
plt.tight_layout()
plt.show()''')
    md('''## 2. Изменение направления между концами

Используем заранее описанные в 07 категории до 5°, 5–15° и более 15°. Это признаки концов окна, не полная характеристика поворота. Группировка не устанавливает причину ошибок. Отсутствующая категория не заменяется нулевой ошибкой.''')
    code('''# Присоединяем диагностические поля из этапа 07 без изменения модели
diagnostics = pd.read_csv(study/'data/public_pilot/07_hidden_point_errors.csv')
diagnostics = diagnostics[diagnostics.method.eq('linear_lat_lon')]
test = test.merge(diagnostics[['flight_id','gap_s','left_time','target_time',
    'endpoint_heading_change_deg','target_position_age_s']],on=key[1:],validate='one_to_one')
test['turn_group'] = pd.cut(test.endpoint_heading_change_deg,[-.001,5,15,180],
                           labels=['До 5°','5–15°','Более 15°'])
# Фиксируем интервал 120 секунд для сопоставимости категорий
long = test[test.gap_s.eq(120)]
turn_flight = long.groupby(['flight_id','turn_group'],observed=True).agg(
    tasks=('worse','size'),linear_mean_m=('linear_lat_lon','mean'),
    model_mean_m=('ridge_correction','mean'),fraction_worse=('worse','mean')).reset_index()
display(turn_flight.round(3))
turn = turn_flight.groupby('turn_group',observed=True)[['linear_mean_m','model_mean_m']].mean()
ax = turn.rename(columns={'linear_mean_m':'Линейный метод','model_mean_m':'Модель'}).plot.bar(
    color=['#00897B','#D65A4A'],figsize=(9,4))
ax.set(xlabel='Изменение направления',ylabel='Средняя ошибка, м',title='Интервалы 120 секунд')
ax.tick_params(axis='x',rotation=0)
plt.tight_layout()
plt.show()''')
    md('''## 3. Проверяем свежесть и крупнейшие ухудшения

Возраст скрытой позиции используется только для диагностики: он не был признаком модели. Порог до 2 секунд уже ограничивает этот пилот. Категории 0–0,5, 0,5–1 и 1–2 секунды выбраны для описания, а не для исключения неудобных ошибок.

Сначала ранжируем окна по средней величине ухудшения. Затем показываем ошибки каждой внутренней цели худшего окна. Это позволяет увидеть, ухудшается ли весь участок или только отдельные точки.''')
    code('''# Показываем возраст скрытого наблюдения отдельно от длины пропуска
test['age_group'] = pd.cut(test.target_position_age_s,[-.001,.5,1,2],labels=['0–0,5 с','0,5–1 с','1–2 с'])
age = test.groupby(['flight_id','gap_s','age_group'],observed=True).agg(
    tasks=('worse','size'),linear_mean_m=('linear_lat_lon','mean'),model_mean_m=('ridge_correction','mean')).reset_index()
display(age.round(2))
# Рассматриваем каждое окно один раз, усредняя ошибки его внутренних целей
windows = test.groupby(['flight_id','gap_s','left_time']).agg(
    targets=('worse','size'),linear_mean_m=('linear_lat_lon','mean'),
    model_mean_m=('ridge_correction','mean'),improvement_m=('improvement_m','mean')).reset_index()
worst = windows.sort_values('improvement_m').head(10)
display(worst.round(2))
windows.to_csv(study/'reports/09_test_windows.csv',index=False)
flight_gap.to_csv(study/'reports/09_test_by_gap.csv',index=False)
turn_flight.to_csv(study/'reports/09_test_by_turn.csv',index=False)
age.to_csv(study/'reports/09_test_by_age.csv',index=False)
# Проверяем, что анализ не изменил зафиксированную модель
assert hashlib.sha256(model_path.read_bytes()).hexdigest() == model_sha
result = {'model_sha256':model_sha,'test_tasks':len(test),
          'test_windows':len(windows),'model_modified':False,
          'fraction_worse_by_flight':test.groupby('flight_id').worse.mean().to_dict(),
          'worst_window':worst.iloc[0].to_dict()}
(study/'reports/09_error_analysis_audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\\n',encoding='utf8')''')
    code('''# Показываем парные ошибки внутренних целей худшего окна
worst_window = worst.iloc[0]
detail = test[(test.flight_id == worst_window.flight_id) &
              (test.gap_s == worst_window.gap_s) &
              (test.left_time == worst_window.left_time)].sort_values('target_time')
fig, ax = plt.subplots(figsize=(9,4))
offset = detail.target_time-detail.left_time
ax.plot(offset,detail.linear_lat_lon,'o-',color='#00897B',label='Линейный метод')
ax.plot(offset,detail.ridge_correction,'o-',color='#D65A4A',label='Модель')
ax.set(xlabel='Время от левого конца, с',ylabel='Ошибка, м',
       title=f'Худшее окно по среднему ухудшению · {worst_window.flight_id}')
ax.legend()
plt.tight_layout()
plt.show()
display(detail[['target_time','linear_lat_lon','ridge_correction','improvement_m']].round(2))''')
    md('''## Вывод и дальнейшая проверка

Доля ухудшений и крупнейшие ошибки показаны выше. Они сохраняются в отчетах, не исключаются из результата. Даже если среднее уменьшается, модель может ухудшать отдельные цели и короткие интервалы.

Для следующего этапа сохраняем коэффициенты 08 без переобучения и заранее выбираем другой день по доступности публичного архива и времени рейсов, без просмотра ошибок модели. Независимая дата усиливает проверку, но еще не доказывает переносимость на любые аэропорты, борта и источники.''')
    nb = nbf.v4.new_notebook(cells=cells)
    nb.metadata['kernelspec'] = dict(display_name='Python 3',language='python',name='python3')
    from nbclient import NotebookClient
    NotebookClient(nb,timeout=300,resources={'metadata':{'path':str(STUDY.parents[1])}}).execute()
    nbf.validate(nb)
    nbf.write(nb,STUDY/'notebooks/experiments/09_error_analysis.ipynb')
    print('Выполнен 09_error_analysis.ipynb',flush=True)


if __name__ == '__main__': main()
