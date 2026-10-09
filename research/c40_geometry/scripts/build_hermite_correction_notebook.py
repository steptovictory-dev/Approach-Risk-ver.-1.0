"""Обучаемая поправка к Эрмиту с проверкой на двух новых датах."""
from pathlib import Path
import nbformat as nbf

STUDY = Path(__file__).resolve().parents[1]


def main():
    source11 = nbf.read(STUDY/'notebooks/experiments/11_expanded_training.ipynb',as_version=4)
    code11 = [c.source for c in source11.cells if c.cell_type == 'code']
    data_code = next(c for c in code11 if '# Формируем список источников' in c)
    data_code = data_code.replace("[plan['validation_day'],plan['test_day']]", "[plan['validation_day']]+plan['test_days']")
    data_code = data_code.replace("folder = study/f'data/expanded_training/{day}'",
        "folder = study/f'data/expanded_training/{day}' if day not in plan['test_days'] else study/f'data/hermite_validation/{day}'\n    prefix = ('12_' if day in plan['test_days'] else '11_')+day")
    data_code = data_code.replace("folder/f'11_{day}_hidden_point_errors.csv','11_'+day", "folder/f'{prefix}_hidden_point_errors.csv',prefix")
    feature_code = next(c for c in code11 if '# Берем точно те же цели' in c)
    trainer = next(c for c in code11 if 'def fit_ridge(' in c).split('# Проверяем, что тот же код')[0]
    trainer = trainer.replace('target[mask]', 'hermite_target[mask]')
    hermite_code = next(c for c in code11 if "predictions['hermite']" in c)
    hermite_code = '# Эрмит использует только доступные концы и их горизонтальные скорости\n'+hermite_code.split('# Эрмит использует только доступные концы и их горизонтальные скорости\n')[1].split("predictions['hermite']")[0]
    cells = []
    def md(s): cells.append(nbf.v4.new_markdown_cell(s))
    def code(s): cells.append(nbf.v4.new_code_cell(s))
    md('''# 12 · Может ли ML улучшить интерполяцию Эрмита?

**Вопрос:** дает ли обучаемая поправка пользу сверх сильного метода без обучения?

Сохраняем 41 обучающий рейс и 8 validation-рейсов из 11. Признаки – те же 19 характеристик доступных концов. Меняется проверочная поправка: теперь это отличие скрытого наблюдения от Эрмита, а не от прямой линии.

Регуляризация выбирается только по validation 25 ноября. Проверка выполняется на заранее назначенных новых датах **9 и 16 декабря 2019 года**. День 2 декабря, уже просмотренный в 11, в обучение и проверку не включаем. Сравниваем чистый Эрмит, сохраненную модель 11 и Эрмит с обучаемой поправкой. Изменения методов после просмотра новых ошибок не выполняем.''')
    code('''# Подключаем библиотеки и проверяем целостность модели 11
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
plan = json.loads((study/'reports/12_hermite_protocol.json').read_text(encoding='utf8'))
plan11 = json.loads((study/'reports/11_expanded_protocol.json').read_text(encoding='utf8'))
plan['old_model_sha256'] = plan11['old_model_sha256']
model11_path = study/'data/expanded_training/11_ridge_model.npz'
assert hashlib.sha256(model11_path.read_bytes()).hexdigest() == plan['model11_sha256']
model11 = np.load(model11_path,allow_pickle=False)
old_split = json.loads((study/'reports/08_ml_protocol.json').read_text(encoding='utf8'))['split']
small_flights = [f for f,part in old_split.items() if part == 'train']
display(plan)''')
    md('''## 1. Проверяем даты, происхождение данных и разделение

Подготовительные скрипты только получают наблюдения. Здесь виден состав каждого дня, число целей и проверки контрольных сумм. Общие правила пригодности и скрытия точек остаются прежними: воздушное состояние, возраст позиции 0–2 секунды, исходная сетка 10 секунд, интервалы 20–120 секунд.''')
    code(data_code)
    md('''## 2. Готовим признаки без скрытых координат

Признаки используют позиции, скорости, направления и свежесть двух доступных концов, а также известное время цели. Скрытые координаты используются только в ответе. Все методы проверяются на одинаковых пригодных целях.''')
    code(feature_code)
    md('''## 3. Строим Эрмита и определяем новую поправку

Кубический Эрмит согласует позиции и скорости концов. В локальных метрах вычисляем его отличие от линейной интерполяции. Затем вычитаем эту величину из прежнего проверочного ответа: остаток и есть цель ML.

Итоговая позиция = линейная позиция + поправка Эрмита + обучаемая поправка. Эрмит не обучается. Проверяем, что новое выражение для ответа восстанавливает наблюдаемые координаты.''')
    code(hermite_code+'''
# Ответ модели – только оставшаяся ошибка относительно Эрмита
hermite_target = target-hermite
assert np.isfinite(hermite_target).all()
assert np.allclose(hermite+hermite_target,target)
# Проверяем сохранение граничных позиций кубическим базисом
endpoint_w = np.array([0.,1.])
assert np.allclose(-2*endpoint_w**3+3*endpoint_w**2,endpoint_w)
assert np.allclose(endpoint_w**3-2*endpoint_w**2+endpoint_w,0)
assert np.allclose(endpoint_w**3-endpoint_w**2,0)
display(pd.DataFrame(hermite_target[tasks.part.eq('train').to_numpy()],
    columns=['Поправка к востоку, м','Поправка к северу, м']).describe().round(2))''')
    md('''## 4. Обучаем поправку и выбираем регуляризацию

Стандартизация – только по train. Каждому обучающему рейсу даем одинаковый суммарный вес. Регуляризацию выбираем по средней рейсовой ошибке WGS84 на validation. Чистый Эрмит показан для сравнения; даже на validation улучшение не предполагается заранее.''')
    code(trainer+'''
# Подтверждаем, что состав обучения и преобразование признаков совпадают с 11
assert tasks.loc[train_mask,'flight_id'].nunique() == 41
assert tasks.loc[val_mask,'flight_id'].nunique() == 8
assert list(features.columns) == model11['feature_names'].tolist()
selection = []
for alpha in plan['alphas']:
    model,x = fit_ridge(train_mask,alpha)
    error = correction_error(val_mask,hermite[val_mask]+x[val_mask]@model['coef'])
    selection.append({'alpha':alpha,'validation_mean_m':macro_mean(val_mask,error)})
selection = pd.DataFrame(selection).sort_values(['validation_mean_m','alpha'])
alpha = float(selection.iloc[0].alpha)
model12,x = fit_ridge(train_mask,alpha)
assert np.allclose(model12['mean'],model11['mean'])
assert np.allclose(model12['scale'],model11['scale'])
display(selection.round(3))
val_hermite = macro_mean(val_mask,correction_error(val_mask,hermite[val_mask]))
display(Markdown(f'**Validation:** чистый Эрмит {val_hermite:.1f} м; '
                 f'Эрмит + ML {selection.iloc[0].validation_mean_m:.1f} м; α={alpha:g}'))
# Сохраняем модель до открытия ошибок test
np.savez(study/'data/hermite_validation/12_ridge_hermite.npz',**model12,
         feature_names=features.columns.to_numpy(dtype=str),reference='cubic_hermite_local')
selection.to_csv(study/'reports/12_validation_selection.csv',index=False)''')
    md('''## 5. Открываем две новые даты проверки

Модель 11 применяется с сохраненными параметрами. Модель 12 обучена только по остаткам Эрмита на train. На test ничего не подбираем. Средние, медианы и P95 сначала считаются по рейсам; рейсы имеют одинаковый вес. Среднее рейсовых P95 не является P95 объединенной выборки.''')
    code('''# Применяем зафиксированные преобразования к новым рейсам
def predict(model,mask):
    a = np.column_stack([np.ones(int(mask.sum())),(features.loc[mask].to_numpy()-model['mean'])/model['scale']])
    return a@model['coef']
predictions = {'ridge_11':predict(model11,test_mask),
               'hermite':hermite[test_mask],
               'hermite_ml':hermite[test_mask]+predict(model12,test_mask)}
test_tasks = tasks.loc[test_mask,key].copy()
evaluation = baseline_errors.merge(test_tasks,on=key,how='inner',validate='many_to_one')
for method,correction in predictions.items():
    rows = test_tasks.copy()
    rows['method'],rows['error_m'] = method,correction_error(test_mask,correction)
    evaluation = pd.concat([evaluation,rows],ignore_index=True)
assert evaluation.groupby(key).method.nunique().eq(6).all()
assert np.isfinite(evaluation.error_m).all()
evaluation['day'] = evaluation.flight_id.map(flight_day)
summary = evaluation.groupby(['day','flight_id','method']).error_m.agg(
    targets='size',mean_m='mean',median_m='median',p95_m=lambda s:s.quantile(.95),max_m='max').reset_index()
macro = summary.groupby('method')[['mean_m','median_m','p95_m']].mean()
per_day = summary.groupby(['day','method'])[['mean_m','median_m','p95_m']].mean()
display(macro.round(2))
display(per_day.round(2))
display(summary.round(2))
# Для каждой парной цели отмечаем ухудшение относительно чистого Эрмита
paired = evaluation.pivot(index=key,columns='method',values='error_m').reset_index()
paired['day'] = paired.flight_id.map(flight_day)
paired['worse'] = paired.hermite_ml > paired.hermite
paired['improvement_m'] = paired.hermite-paired.hermite_ml
display(paired.groupby(['day','flight_id']).agg(tasks=('worse','size'),fraction_worse=('worse','mean'),
    mean_improvement_m=('improvement_m','mean')).round(3))''')
    md('''## 6. Длина пропуска, изменение направления и доля ухудшений

Зеленый – чистый Эрмит, красный – Эрмит с ML, серый – модель 11. Отдельно показываем оба новых дня. Категории изменения направления до 5°, 5–15° и более 15° взяты из 07. Это изменение между концами, а не полное описание маневра. Для сравнения категорий фиксируем интервал 120 секунд.''')
    code('''# Разделяем общий результат по длине пропуска и по новой дате
by_gap = evaluation.groupby(['day','flight_id','gap_s','method']).error_m.mean().reset_index()
gap = by_gap.groupby(['gap_s','method']).error_m.mean().unstack()
display(gap.round(2))
labels = {'hermite':'Эрмит','hermite_ml':'Эрмит + ML','ridge_11':'Модель 11'}
colors = {'hermite':'#00897B','hermite_ml':'#D65A4A','ridge_11':'#85929E'}
fig,axes = plt.subplots(1,2,figsize=(13,4.5))
day_table = summary[summary.method.isin(labels)].groupby(['day','method']).mean_m.mean().unstack()
day_table.rename(columns=labels).plot.bar(ax=axes[0],color=[colors[m] for m in day_table.columns])
axes[0].set(xlabel='Новая дата',ylabel='Средняя ошибка, м',title='Равный вес рейсов внутри дня')
axes[0].tick_params(axis='x',rotation=0)
axes[0].legend(fontsize=9)
for method in labels:
    axes[1].plot(gap.index,gap[method],'o-',color=colors[method],label=labels[method])
axes[1].set(xlabel='Интервал, с',ylabel='Средняя ошибка, м',title='Влияние длины пропуска',xticks=gap.index)
axes[1].legend(fontsize=9)
plt.tight_layout()
fig.savefig(study/'figures/12_hermite_correction.png',dpi=180)
plt.show()
# Присоединяем только диагностический показатель направления
diag = baseline_errors[baseline_errors.method.eq('linear_lat_lon')]
paired = paired.merge(diag[key+['endpoint_heading_change_deg']],on=key,validate='one_to_one')
paired['turn_group'] = pd.cut(paired.endpoint_heading_change_deg,[-.001,5,15,180],
                            labels=['До 5°','5–15°','Более 15°'])
turn = paired[paired.gap_s.eq(120)].groupby(['day','flight_id','turn_group'],observed=True).agg(
    tasks=('worse','size'),hermite_mean_m=('hermite','mean'),model_mean_m=('hermite_ml','mean'),
    fraction_worse=('worse','mean')).reset_index()
display(turn.round(3))
turn_macro = turn.groupby('turn_group',observed=True)[['hermite_mean_m','model_mean_m','fraction_worse']].mean()
display(turn_macro.round(3))
ax = turn_macro[['hermite_mean_m','model_mean_m']].rename(columns={
    'hermite_mean_m':'Эрмит','model_mean_m':'Эрмит + ML'}).plot.bar(color=['#00897B','#D65A4A'],figsize=(9,4))
ax.set(xlabel='Изменение направления',ylabel='Средняя ошибка, м',title='Интервалы 120 секунд · среднее по рейсовым группам')
ax.tick_params(axis='x',rotation=0)
plt.tight_layout()
plt.show()
# Сохраняем краткие итоги и полные ошибки независимо от знака улучшения
base = float(macro.loc['hermite','mean_m'])
learned = float(macro.loc['hermite_ml','mean_m'])
improvement = 100*(base-learned)/base
fractions = paired.groupby(['day','flight_id']).worse.mean()
display(Markdown(f'**Новые даты вместе:** Эрмит {base:.1f} м → Эрмит + ML {learned:.1f} м; '
                 f'изменение в пользу ML **{improvement:+.1f}%**. '
                 f'Средняя рейсовая доля целей с ухудшением: **{100*fractions.mean():.1f}%**.'))
assert hashlib.sha256(model11_path.read_bytes()).hexdigest() == plan['model11_sha256']
summary.to_csv(study/'reports/12_test_by_flight.csv',index=False)
macro.to_csv(study/'reports/12_test_macro.csv')
per_day.to_csv(study/'reports/12_test_by_day.csv')
gap.to_csv(study/'reports/12_test_by_gap.csv')
turn.to_csv(study/'reports/12_test_by_turn.csv',index=False)
fractions.to_csv(study/'reports/12_fraction_worse.csv')
evaluation.to_csv(study/'data/hermite_validation/12_test_errors.csv',index=False)
pd.DataFrame(provenance).to_csv(study/'reports/12_data_inventory.csv',index=False)
result = {'test_days':plan['test_days'],'train_flights':int(tasks.loc[train_mask,'flight_id'].nunique()),
          'validation_flights':int(tasks.loc[val_mask,'flight_id'].nunique()),
          'test_flights':int(tasks.loc[test_mask,'flight_id'].nunique()),'test_tasks':int(test_mask.sum()),
          'alpha':alpha,'hermite_mean_m':base,'hermite_ml_mean_m':learned,
          'model11_mean_m':float(macro.loc['ridge_11','mean_m']),
          'improvement_percent':improvement,'macro_fraction_worse':float(fractions.mean()),
          'model11_unchanged':True}
(study/'reports/12_hermite_audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\\n',encoding='utf8')''')
    md('''## Границы вывода

Поправка оценена на двух новых датах без их использования для настройки. Это все еще близкие аэропорты, часы и тот же источник. Validation уже применялась в 11 и является частью разработки. Разделение выполнено по датам и полетам, не по бортам.

Окна перекрываются; отдельные цели не независимы. Число рейсов ограничено, статистическая значимость не оценивалась. Преимущество или ухудшение относится к текущей модели с 19 признаками, а не к любым методам ML. При ухудшении нельзя объявлять поправку предпочтительной или исключать неудобные случаи.

Ответы – наблюдения OpenSky, не независимое истинное положение. Расхождение 43 км остается отдельной задачей. Любая дальнейшая настройка после просмотра этих дат потребует следующего проверочного набора.''')
    nb = nbf.v4.new_notebook(cells=cells)
    nb.metadata['kernelspec'] = dict(display_name='Python 3',language='python',name='python3')
    path = STUDY/'notebooks/experiments/12_hermite_correction.ipynb'
    nbf.write(nb,path)
    from nbclient import NotebookClient
    NotebookClient(nb,timeout=900,resources={'metadata':{'path':str(STUDY.parents[1])}}).execute()
    nbf.validate(nb)
    nbf.write(nb,path)
    print('Выполнен 12_hermite_correction.ipynb',flush=True)


if __name__ == '__main__': main()
