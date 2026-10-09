"""Проверка сохраненной модели на заранее выбранной другой дате."""
from pathlib import Path
import nbformat as nbf

STUDY = Path(__file__).resolve().parents[1]


def main():
    # Показываем тот же код формирования признаков, который выполнен в 08
    previous = nbf.read(STUDY/'notebooks/experiments/08_ml_correction.ipynb',as_version=4)
    feature_code = next(c.source for c in previous.cells if c.cell_type == 'code' and
                        '# Берем точно те же цели' in c.source)
    cells = []
    def md(s): cells.append(nbf.v4.new_markdown_cell(s))
    def code(s): cells.append(nbf.v4.new_code_cell(s))
    md('''# 10 · Проверка зафиксированной модели на другой дате

**Вопрос:** сохраняется ли улучшение при переходе с 21 на 28 октября 2019 года?

До просмотра ошибок выбраны первые 10 хронологических рейсов EGCC/EPWA с C40 после 22:05 и посадкой до 22:58 UTC. Это позволяет использовать один публичный час. Пропавшие рейсы не заменяем другими. Модель 08, ее коэффициенты, стандартизация и список признаков зафиксированы. **Обучения и настройки на новых данных нет**.

Новая дата не была частью 07–09. Но аэропорты и сезон близки к обучению, а наблюдения поступают из того же источника. Это проверка на другом дне, не доказательство универсальной переносимости. Проверочный ответ – скрытое наблюдение OpenSky.''')
    code('''# Загружаем протокол, происхождение новых данных и замороженную модель
from pathlib import Path
import contextlib
import io
import hashlib
import json
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython import get_ipython
from IPython.display import display, Markdown
root = next(p for p in [Path.cwd(),*Path.cwd().parents] if (p/'research/c40_geometry').is_dir())
study = root/'research/c40_geometry'
sys.path.insert(0,str(study/'scripts'))
from evaluate_hidden_points import main as evaluate_baselines
from audit_c40_radial_cases import ellipsoid_distance
if get_ipython() is not None:
    get_ipython().run_line_magic('matplotlib','inline')
plt.rcParams.update({'figure.dpi':120,'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
protocol = json.loads((study/'reports/10_external_protocol.json').read_text(encoding='utf8'))
download = json.loads((study/'reports/10_external_download.json').read_text(encoding='utf8'))
model_path = study/'data/public_pilot/08_ridge_model.npz'
assert hashlib.sha256(model_path.read_bytes()).hexdigest() == protocol['model_sha256']
model = np.load(model_path,allow_pickle=False)
source_path = study/'data/external_day/external_geometry.csv'
assert hashlib.sha256(source_path.read_bytes()).hexdigest() == download['selected_sha256']
source = pd.read_csv(source_path)
assert pd.to_datetime(source.time,unit='s',utc=True).dt.strftime('%Y-%m-%d').eq(protocol['day']).all()
original_split = json.loads((study/'reports/08_ml_protocol.json').read_text(encoding='utf8'))['split']
assert not set(source.case_flight_id).intersection(original_split)
display(pd.DataFrame(download['flights']))
display(Markdown(f"**Дата:** {protocol['day']} · **Источник:** {download['url']} · **Модель:** α={float(model['alpha']):g}"))
# Повторяем неизменные правила скрытия точек и сохраняем отдельные результаты
with contextlib.redirect_stdout(io.StringIO()):
    evaluate_baselines(source_path,prefix='10_external')
baseline_errors = pd.read_csv(source_path.parent/'10_external_hidden_point_errors.csv')
split = {f:'external' for f in source.case_flight_id.unique()}''')
    md('''## 1. Формируем те же признаки по доступным концам

Код ниже совпадает с формированием признаков 08. Скрытые координаты входят только в проверочную поправку. При отсутствии необходимых признаков сохраняем учет пропусков и сравниваем методы на общих целях.''')
    code(feature_code)
    md('''## 2. Применяем сохраненные коэффициенты

Среднее и масштаб признаков загружаем из модели 08, а не пересчитываем по новой дате. Проверяем порядок признаков и точное воспроизведение линейного ориентира. Уменьшение ошибки не гарантировано.''')
    code('''# Проверяем схему признаков перед применением модели
assert list(features.columns) == model['feature_names'].tolist()
x = np.column_stack([np.ones(len(features)),(features.to_numpy()-model['mean'])/model['scale']])
correction = x@model['coef']
pred_lat = linear_lat+correction[:,1]/north_scale
pred_lon = linear_lon+correction[:,0]/east_scale
model_errors = 1000*ellipsoid_distance(tasks.target_lat.to_numpy(),tasks.target_lon.to_numpy(),pred_lat,pred_lon)
assert np.isfinite(model_errors).all()
evaluation = baseline_errors.merge(tasks[key],on=key,how='inner',validate='many_to_one')
ml = tasks[key].copy()
ml['method'] = 'ridge_correction'
ml['error_m'] = model_errors
evaluation = pd.concat([evaluation,ml],ignore_index=True)
assert evaluation.groupby(key).method.nunique().eq(4).all()
linear_error = 1000*ellipsoid_distance(tasks.target_lat.to_numpy(),tasks.target_lon.to_numpy(),linear_lat,linear_lon)
saved_linear = evaluation[evaluation.method.eq('linear_lat_lon')].set_index(key).error_m
assert np.allclose(saved_linear.reindex(pd.MultiIndex.from_frame(tasks[key])),linear_error,atol=1e-5)
# Ошибки агрегируем по рейсам, чтобы количество окон не определяло их вес
summary = evaluation.groupby(['flight_id','method']).error_m.agg(
    targets='size',mean_m='mean',median_m='median',p95_m=lambda s:s.quantile(.95),max_m='max').reset_index()
macro = summary.groupby('method')[['mean_m','median_m','p95_m']].mean()
display(macro.round(2))
display(summary.round(2))
# Проверяем, что коэффициенты не были перезаписаны
assert hashlib.sha256(model_path.read_bytes()).hexdigest() == protocol['model_sha256']''')
    md('''## 3. Сравниваем рейсы, пропуски и ухудшения

Зеленый – линейная интерполяция, красный – зафиксированная модель. Среднее рейсовых P95 не является P95 объединенной выборки. Доля ухудшений считается по парным целям, а не по разным подвыборкам.''')
    code('''colors = {'linear_lat_lon':'#00897B','ridge_correction':'#D65A4A'}
labels = {'linear_lat_lon':'Линейный метод','ridge_correction':'Модель'}
fig, axes = plt.subplots(1,2,figsize=(14,4.8))
table = summary[summary.method.isin(labels)].pivot(index='flight_id',columns='method',values='mean_m')
table.rename(columns=labels).plot.bar(ax=axes[0],color=[colors[m] for m in table.columns])
axes[0].set(xlabel='',ylabel='Средняя ошибка, м',title='Новые рейсы · другая дата')
axes[0].tick_params(axis='x',rotation=50)
axes[0].legend(fontsize=8)
by_gap = evaluation.groupby(['flight_id','gap_s','method']).error_m.mean().reset_index()
gap = by_gap.groupby(['gap_s','method']).error_m.mean().reset_index()
for method in labels:
    group = gap[gap.method.eq(method)]
    axes[1].plot(group.gap_s,group.error_m,'o-',color=colors[method],label=labels[method])
axes[1].set(xlabel='Интервал, с',ylabel='Средняя ошибка, м',title='Равный вес рейсов',xticks=[20,30,60,120])
axes[1].legend()
plt.tight_layout()
fig.savefig(study/'figures/10_external_validation.png',dpi=180)
plt.show()
# Считаем долю целей с ухудшением отдельно для каждого нового рейса
paired = evaluation.pivot(index=key,columns='method',values='error_m').reset_index()
paired['worse'] = paired.ridge_correction > paired.linear_lat_lon
display(paired.groupby('flight_id').worse.mean().rename('Доля ухудшений').round(3))
baseline = float(macro.loc['linear_lat_lon','mean_m'])
learned = float(macro.loc['ridge_correction','mean_m'])
improvement = 100*(baseline-learned)/baseline
wins = int((table.ridge_correction < table.linear_lat_lon).sum())
display(Markdown(f'**Новая дата:** средняя рейсовая ошибка {baseline:.1f} → {learned:.1f} м; '
                 f'изменение в пользу модели {improvement:+.1f}%. Улучшение среднего на {wins} из {len(table)} рейсов.'))
summary.to_csv(study/'reports/10_external_by_flight.csv',index=False)
macro.to_csv(study/'reports/10_external_macro.csv')
gap.to_csv(study/'reports/10_external_by_gap.csv',index=False)
evaluation.to_csv(source_path.parent/'10_external_ml_errors.csv',index=False)
result = {'day':protocol['day'],'selected_flights':len(protocol['cases']),'evaluated_flights':len(table),
    'tasks':len(tasks),'linear_macro_mean_m':baseline,'model_macro_mean_m':learned,
    'improvement_percent':improvement,'flights_improved':wins,
    'model_sha256':protocol['model_sha256'],'source_sha256':download['selected_sha256'],
    'model_refit':False,'fraction_worse_by_flight':paired.groupby('flight_id').worse.mean().to_dict()}
(study/'reports/10_external_audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\\n',encoding='utf8')''')
    md('''## Границы результата

Результат относится к выбранному часу другой даты и найденным рейсам. Их выбор зафиксирован до оценки модели. Коэффициенты и параметры преобразований сохранены из 08 без обучения на новых данных.

Перекрывающиеся окна создают зависимые ошибки. Тот же борт может встречаться в разных датах, поэтому это разделение по полетам и дате, не по самолетам. Близкие аэропорты, сезон и источник ограничивают переносимость. Дальнейшее изменение метода потребует нового независимого набора.

Сырые архивы, траектории и подробные ошибки остаются локально. Источник – публичный архив OpenSky; его URL, размер и SHA256 сохранены в реестре загрузки.''')
    nb = nbf.v4.new_notebook(cells=cells)
    nb.metadata['kernelspec'] = dict(display_name='Python 3',language='python',name='python3')
    from nbclient import NotebookClient
    NotebookClient(nb,timeout=300,resources={'metadata':{'path':str(STUDY.parents[1])}}).execute()
    nbf.validate(nb)
    nbf.write(nb,STUDY/'notebooks/experiments/10_external_validation.ipynb')
    print('Выполнен 10_external_validation.ipynb',flush=True)


if __name__ == '__main__': main()
