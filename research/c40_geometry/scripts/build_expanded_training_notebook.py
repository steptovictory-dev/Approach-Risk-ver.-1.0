"""Больше обучающих рейсов при неизменных признаках и новой проверочной дате."""
from pathlib import Path
import nbformat as nbf

STUDY = Path(__file__).resolve().parents[1]


def main():
    previous = nbf.read(STUDY/'notebooks/experiments/08_ml_correction.ipynb',as_version=4)
    feature_code = next(c.source for c in previous.cells if c.cell_type == 'code' and '# Берем точно те же цели' in c.source)
    cells = []
    def md(s): cells.append(nbf.v4.new_markdown_cell(s))
    def code(s): cells.append(nbf.v4.new_code_cell(s))
    md('''# 11 · Помогает ли больше обучающих данных?

**Вопрос:** улучшает ли расширение обучающей выборки результат при той же модели и тех же признаках?

Модель – Ridge с поправкой к линейной интерполяции. Добавляем изученные рейсы 28 октября и три новые даты ноября. Выбор регуляризации – 25 ноября, проверка – **2 декабря 2019 года**. Порядок добавления данных и даты зафиксированы до расчета ошибок.

Сравниваем исходную модель 08, модель на тех же шести train-рейсах с регуляризацией по новой validation-дате и расширенную модель с той же процедурой выбора. Это позволяет отличить влияние данных от влияния повторного выбора регуляризации. Дополнительно проверяем интерполяцию Эрмита, которая использует позиции и скорости обоих концов без обучения.''')
    code('''# Загружаем план дат и фиксируем целостность старой модели
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
plan = json.loads((study/'reports/11_expanded_protocol.json').read_text(encoding='utf8'))
old_model_path = study/'data/public_pilot/08_ridge_model.npz'
assert hashlib.sha256(old_model_path.read_bytes()).hexdigest() == plan['old_model_sha256']
old_model = np.load(old_model_path,allow_pickle=False)
old_split = json.loads((study/'reports/08_ml_protocol.json').read_text(encoding='utf8'))['split']
small_flights = [f for f,part in old_split.items() if part == 'train']
assert len(small_flights) == 6
display(plan)''')
    md('''## 1. Собираем данные разных дат и повторяем правила 07

Каждая дата хранится в своей локальной папке. Проверяем SHA256 подготовленной последовательности и ее календарную дату. Берем все найденные заранее выбранные рейсы; пропавшие случаи не заменяем по результатам модели. Аналитическая база лишь задает рейсы и временные окна; восстановление оценивается по наблюдениям OpenSky.

Прежние validation/test-рейсы из 08 в обучение не добавляем. День 28 октября уже изучен в 10 и теперь явно становится обучающим. Для нового test выделена другая дата.''')
    code('''# Формируем список источников с заранее назначенными ролями
datasets = [
    ('2019-10-21','train',study/'data/public_pilot/EPWA_control_geometry.csv',
     study/'data/public_pilot/07_hidden_point_errors.csv',None),
    ('2019-10-28','train',study/'data/external_day/external_geometry.csv',
     study/'data/external_day/10_external_hidden_point_errors.csv','10_external')]
for day in plan['train_days']+[plan['validation_day'],plan['test_day']]:
    part = 'train' if day in plan['train_days'] else ('validation' if day == plan['validation_day'] else 'test')
    folder = study/f'data/expanded_training/{day}'
    datasets.append((day,part,folder/'external_geometry.csv',folder/f'11_{day}_hidden_point_errors.csv','11_'+day))
frames, baseline_frames, provenance = [], [], []
split, flight_day = {}, {}
for day,part,path,error_path,prefix in datasets:
    frame = pd.read_csv(path)
    if day == '2019-10-21':
        frame = frame[frame.case_flight_id.isin(small_flights)].copy()
        old_audit = json.loads((study/'reports/07_hidden_point_audit.json').read_text(encoding='utf8'))
        assert hashlib.sha256(path.read_bytes()).hexdigest() == old_audit['source_sha256']
    else:
        manifest = json.loads((study/f'reports/{prefix}_download.json').read_text(encoding='utf8'))
        assert hashlib.sha256(path.read_bytes()).hexdigest() == manifest['selected_sha256']
    assert pd.to_datetime(frame.time,unit='s',utc=True).dt.strftime('%Y-%m-%d').eq(day).all()
    # Для новых дат применяем неизменные интервалы и критерии пригодности
    if day in plan['train_days']+[plan['validation_day'],plan['test_day']]:
        if not error_path.exists() or not (study/f'reports/{prefix}_hidden_point_audit.json').exists():
            with contextlib.redirect_stdout(io.StringIO()):
                evaluate_baselines(path,prefix=prefix)
        cached = json.loads((study/f'reports/{prefix}_hidden_point_audit.json').read_text(encoding='utf8'))
        assert cached['source_sha256'] == hashlib.sha256(path.read_bytes()).hexdigest()
        assert cached['gaps_s'] == [20,30,60,120] and cached['max_position_age_s'] == 2
    errors = pd.read_csv(error_path)
    errors = errors[errors.flight_id.isin(frame.case_flight_id.unique())].copy()
    for flight in frame.case_flight_id.unique():
        assert flight not in split, 'Повтор идентификатора между датами'
        split[flight],flight_day[flight] = part,day
    frames.append(frame)
    baseline_frames.append(errors)
    provenance.append({'Дата':day,'Часть':part,'Рейсов':frame.case_flight_id.nunique(),
                       'Наблюдений':len(frame),'Целей':len(errors)//3})
source = pd.concat(frames,ignore_index=True)
baseline_errors = pd.concat(baseline_frames,ignore_index=True)
display(pd.DataFrame(provenance))
# Проверяем непересечение дат обучения, выбора модели и проверки
assert not set(plan['train_days']+['2019-10-21','2019-10-28']).intersection([plan['validation_day'],plan['test_day']])''')
    md('''## 2. Формируем неизменные признаки

Используем код признаков 08. Скрытые координаты входят только в проверочный ответ. Каждому рейсу назначена одна часть. Все методы далее оцениваются на одной и той же подвыборке целей с конечными признаками.''')
    code(feature_code)
    md('''## 3. Измеряем влияние объема обучения на validation

Стандартизация оценивается на текущей обучающей подвыборке. Каждый рейс имеет одинаковый суммарный вес. Для каждой ступени выбираем регуляризацию по средней рейсовой ошибке 25 ноября из одинакового заранее заданного списка.

Ступени добавляют целые даты в зафиксированном порядке. Это описательная кривая, а не случайные независимые повторы. Большая выборка не гарантирует монотонного улучшения. Итоговую модель заранее определяем как модель на всех обучающих датах, не выбираем ступень по test.''')
    code('''train_mask = tasks.part.eq('train').to_numpy()
val_mask = tasks.part.eq('validation').to_numpy()
test_mask = tasks.part.eq('test').to_numpy()
assert set(tasks.loc[test_mask,'flight_id']).isdisjoint(tasks.loc[train_mask | val_mask,'flight_id'])

# Ошибка поправки вычисляется на эллипсоиде, одинаково для всех моделей
def correction_error(mask, correction):
    return 1000*ellipsoid_distance(tasks.loc[mask,'target_lat'].to_numpy(),tasks.loc[mask,'target_lon'].to_numpy(),
        linear_lat[mask]+correction[:,1]/north_scale,linear_lon[mask]+correction[:,0]/east_scale[mask])

def macro_mean(mask,error):
    return pd.DataFrame({'flight_id':tasks.loc[mask,'flight_id'].to_numpy(),'error':error}).groupby('flight_id').error.mean().mean()

def fit_ridge(mask, alpha):
    mean = features.loc[mask].mean().to_numpy()
    scale = features.loc[mask].std(ddof=0).to_numpy(copy=True)
    scale[scale < 1e-12] = 1
    x = np.column_stack([np.ones(len(features)),(features.to_numpy()-mean)/scale])
    a = x[mask]
    counts = tasks.loc[mask].groupby('flight_id').size()
    weights = tasks.loc[mask,'flight_id'].map(1/counts).to_numpy(copy=True)
    weights *= len(weights)/weights.sum()
    penalty = np.eye(a.shape[1])*alpha
    penalty[0,0] = 0
    coef = np.linalg.solve(a.T@(weights[:,None]*a)+penalty,a.T@(weights[:,None]*target[mask]))
    return {'mean':mean,'scale':scale,'coef':coef,'alpha':alpha},x

# Проверяем, что тот же код на прежних шести рейсах воспроизводит модель 08
original_mask = train_mask & tasks.flight_id.isin(small_flights).to_numpy()
reproduced, _ = fit_ridge(original_mask,float(old_model['alpha']))
assert np.allclose(reproduced['mean'],old_model['mean'])
assert np.allclose(reproduced['scale'],old_model['scale'])
assert np.allclose(reproduced['coef'],old_model['coef'],atol=1e-6,rtol=1e-6)
stage_days = [['2019-10-21'],['2019-10-21','2019-10-28']]
for i in range(len(plan['train_days'])):
    stage_days.append(['2019-10-21','2019-10-28']+plan['train_days'][:i+1])
selection, curve, models = [], [], []
for stage,days in enumerate(stage_days):
    mask = train_mask & tasks.flight_id.map(flight_day).isin(days).to_numpy()
    candidates = []
    for alpha in plan['alphas']:
        model,x = fit_ridge(mask,alpha)
        score = macro_mean(val_mask,correction_error(val_mask,x[val_mask]@model['coef']))
        candidates.append((score,alpha,model))
        selection.append({'stage':stage,'alpha':alpha,'validation_mean_m':score})
    score,alpha,model = min(candidates,key=lambda item:(item[0],item[1]))
    models.append(model)
    curve.append({'stage':stage,'train_flights':tasks.loc[mask,'flight_id'].nunique(),
                  'train_tasks':int(mask.sum()),'alpha':alpha,'validation_mean_m':score})
curve = pd.DataFrame(curve)
display(curve.round(3))
fig,ax = plt.subplots(figsize=(9,4))
ax.plot(curve.train_flights,curve.validation_mean_m,'o-',color='#D65A4A')
ax.set(xlabel='Обучающих рейсов',ylabel='Средняя рейсовая ошибка validation, м',title='Кривая увеличения объема данных')
ax.grid(alpha=.2)
plt.tight_layout()
plt.show()
# Сохраняем окончательную модель до просмотра нового test
final_model = models[-1]
np.savez(study/'data/expanded_training/11_ridge_model.npz',**final_model,
         feature_names=features.columns.to_numpy(dtype=str))
pd.DataFrame(selection).to_csv(study/'reports/11_validation_selection.csv',index=False)
curve.to_csv(study/'reports/11_learning_curve.csv',index=False)''')
    md('''## 4. Открываем новую дату проверки

Сравниваем исходную модель 08, модель на шести рейсах с новой validation и модель на всех обучающих датах. Коэффициенты старой модели не менялись. Сильный простой ориентир – кубическая интерполяция Эрмита: она согласует позиции и горизонтальные скорости обоих концов, но может давать выбросы при несогласованных наблюдениях.

Все ошибки считаются на общих целях test. Для длины пропуска показываем отдельные показатели: общий результат может скрывать ухудшение коротких интервалов.''')
    code('''# Применяем каждую зафиксированную модель к одним проверочным целям
def predict(model, mask):
    x = np.column_stack([np.ones(int(mask.sum())),(features.loc[mask].to_numpy()-model['mean'])/model['scale']])
    return x@model['coef']
assert list(features.columns) == old_model['feature_names'].tolist()
predictions = {'ridge_08':predict(old_model,test_mask),
               'ridge_six_new_validation':predict(models[0],test_mask),
               'ridge_expanded':predict(final_model,test_mask)}
# Эрмит использует только доступные концы и их горизонтальные скорости
wt = ((tasks.target_time-tasks.left_time)/tasks.gap_s).to_numpy()
dt = tasks.gap_s.to_numpy()
span_e = ((tasks.b_lon-tasks.a_lon+180)%360-180).to_numpy()*east_scale
span_n = (tasks.b_lat-tasks.a_lat).to_numpy()*north_scale
ha,hb = np.radians(tasks.a_heading),np.radians(tasks.b_heading)
h01 = -2*wt**3+3*wt**2
h10 = wt**3-2*wt**2+wt
h11 = wt**3-wt**2
hermite = np.column_stack([
    (h01-wt)*span_e+h10*dt*tasks.a_velocity*np.sin(ha)+h11*dt*tasks.b_velocity*np.sin(hb),
    (h01-wt)*span_n+h10*dt*tasks.a_velocity*np.cos(ha)+h11*dt*tasks.b_velocity*np.cos(hb)])
predictions['hermite'] = hermite[test_mask]
test_tasks = tasks.loc[test_mask,key].copy()
evaluation = baseline_errors.merge(test_tasks,on=key,how='inner',validate='many_to_one')
for method,correction in predictions.items():
    rows = test_tasks.copy()
    rows['method'],rows['error_m'] = method,correction_error(test_mask,correction)
    evaluation = pd.concat([evaluation,rows],ignore_index=True)
assert evaluation.groupby(key).method.nunique().eq(7).all()
assert np.isfinite(evaluation.error_m).all()
summary = evaluation.groupby(['flight_id','method']).error_m.agg(
    targets='size',mean_m='mean',median_m='median',p95_m=lambda s:s.quantile(.95),max_m='max').reset_index()
macro = summary.groupby('method')[['mean_m','median_m','p95_m']].mean()
display(macro.round(2))
display(summary.round(2))
by_gap = evaluation.groupby(['flight_id','gap_s','method']).error_m.mean().reset_index()
gap = by_gap.groupby(['gap_s','method']).error_m.mean().unstack()
display(gap.round(2))''')
    md('''## 5. Что дали дополнительные данные?

Зеленый показывает линейный метод, серый – исходную модель 08, красный – расширенную модель, фиолетовый – Эрмит. Вывод строится из фактической ошибки; больше данных может улучшить, оставить прежним или ухудшить результат. Две модели с одинаковой новой validation дают наиболее прямое сравнение объема обучения.''')
    code('''labels = {'linear_lat_lon':'Линейный метод','ridge_08':'Модель 08',
          'ridge_six_new_validation':'6 рейсов · новая validation',
          'ridge_expanded':'Больше данных','hermite':'Эрмит'}
colors = {'linear_lat_lon':'#00897B','ridge_08':'#85929E','ridge_expanded':'#D65A4A','hermite':'#7C3AED'}
fig,axes = plt.subplots(1,2,figsize=(14,4.8))
selected = ['linear_lat_lon','ridge_08','ridge_expanded','hermite']
macro.loc[selected,'mean_m'].rename(index=labels).plot.bar(ax=axes[0],color=[colors[m] for m in selected])
axes[0].set(ylabel='Средняя рейсовая ошибка, м',xlabel='',title='Новая дата · общий результат')
axes[0].tick_params(axis='x',rotation=20)
for method in selected:
    axes[1].plot(gap.index,gap[method],'o-',color=colors[method],label=labels[method])
axes[1].set(xlabel='Интервал, с',ylabel='Средняя ошибка, м',title='Преимущество зависит от пропуска',xticks=gap.index)
axes[1].legend(fontsize=8)
plt.tight_layout()
fig.savefig(study/'figures/11_expanded_training.png',dpi=180)
plt.show()
six = float(macro.loc['ridge_six_new_validation','mean_m'])
expanded = float(macro.loc['ridge_expanded','mean_m'])
old = float(macro.loc['ridge_08','mean_m'])
linear = float(macro.loc['linear_lat_lon','mean_m'])
change = 100*(six-expanded)/six
display(Markdown(f'**Влияние дополнительных данных при одинаковой процедуре выбора:** '
                 f'{six:.1f} → {expanded:.1f} м, изменение в пользу расширения **{change:+.1f}%**. '
                 f'Исходная модель 08: {old:.1f} м, линейный метод: {linear:.1f} м.'))
# Сохраняем все результаты, включая ухудшения и сильный простой ориентир
summary.to_csv(study/'reports/11_test_by_flight.csv',index=False)
macro.to_csv(study/'reports/11_test_macro.csv')
gap.to_csv(study/'reports/11_test_by_gap.csv')
evaluation.to_csv(study/'data/expanded_training/11_test_errors.csv',index=False)
pd.DataFrame(provenance).to_csv(study/'reports/11_data_inventory.csv',index=False)
assert hashlib.sha256(old_model_path.read_bytes()).hexdigest() == plan['old_model_sha256']
result = {'test_day':plan['test_day'],'train_flights':int(tasks.loc[train_mask,'flight_id'].nunique()),
          'validation_flights':int(tasks.loc[val_mask,'flight_id'].nunique()),
          'test_flights':int(tasks.loc[test_mask,'flight_id'].nunique()),
          'train_tasks':int(train_mask.sum()),'test_tasks':int(test_mask.sum()),
          'six_refit_mean_m':six,'expanded_mean_m':expanded,'old08_mean_m':old,
          'linear_mean_m':linear,'hermite_mean_m':float(macro.loc['hermite','mean_m']),
          'improvement_from_more_data_percent':change,'final_alpha':final_model['alpha'],
          'old_model_unchanged':True}
(study/'reports/11_expanded_audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\\n',encoding='utf8')''')
    md('''## Границы вывода

Это сравнение одной модели с разным объемом данных, при фиксированных признаках и одной новой дате проверки. Объем меняется вместе с разнообразием дат и рейсов; отделить эти эффекты полностью данный эксперимент не позволяет. Отдельные строки зависимы из-за перекрывающихся окон. Средние рейсовых медиан и P95 не являются показателями объединенного распределения.

Если простой метод Эрмита лучше, улучшение относительно линейной интерполяции само по себе не доказывает необходимость ML. Если расширение данных не помогает, это ограничение текущей модели и признаков, а не доказательство бесполезности данных вообще. Настройка после просмотра этого test потребует новой проверки.

Проверочный ответ – наблюдение OpenSky, а не независимая истинная позиция. Разделение по датам и полетам не является разделением по бортам. Аэропорты EGCC/EPWA и часы близки к прежним опытам. Расхождение 43 км остается самостоятельной нерешенной задачей.''')
    nb = nbf.v4.new_notebook(cells=cells)
    nb.metadata['kernelspec'] = dict(display_name='Python 3',language='python',name='python3')
    path = STUDY/'notebooks/experiments/11_expanded_training.ipynb'
    # Сохраняем исходник перед выполнением, чтобы эксперимент оставался доступным при сбое
    nbf.write(nb,path)
    from nbclient import NotebookClient
    NotebookClient(nb,timeout=900,resources={'metadata':{'path':str(STUDY.parents[1])}}).execute()
    nbf.validate(nb)
    nbf.write(nb,path)
    print('Выполнен 11_expanded_training.ipynb',flush=True)


if __name__ == '__main__': main()
