"""Train and select candidates inside an executed research notebook."""
from pathlib import Path
import nbformat as nb
from nbclient import NotebookClient
ROOT=Path(__file__).resolve().parents[3];BASE=ROOT/'research/boosting-weather-comparison'
cells=[]
def md(s):cells.append(nb.v4.new_markdown_cell(s))
def code(s):cells.append(nb.v4.new_code_cell(s))
md('''# 01. Бустинговые классификаторы GoA: обучение и оценка

## Вопрос и методика перед обучением

Задача – обучение с учителем, **бинарная классификация GoA на C40**.
Метод – CatBoostClassifier: градиентный бустинг деревьев с Logloss.
Прогноз – вероятность класса 1 по исходной разметке. Сравниваем варианты
без погоды (8 полей) и с погодой (34 поля) на одинаковых событиях.

Train 2019–2021: обучение категориальных преобразований и деревьев.
Validation 2022: выбор по AP и пороги. Test 2023: оценка после фиксации
выбора обоих вариантов. Test уже использовался прежде; оценка исследовательская.
Идентификаторы, полоса, число попыток и будущие исходы не являются признаками.
Историческая доступность условна, практическая пригодность не подтверждена.

Источник API: [официальная документация CatBoostClassifier](https://catboost.ai/docs/en/concepts/python-reference_catboostclassifier)
и [fit](https://catboost.ai/docs/en/concepts/python-reference_catboostclassifier_fit).
''')
code('''from pathlib import Path
import sys,json,hashlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
ROOT=Path.cwd();BASE=ROOT/'research/boosting-weather-comparison'
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(BASE/'scripts'))
from boosting_experiment import Experiment,CONFIG,DEST,write_json
from approach_risk.evaluation.classification import metrics
from approach_risk.evaluation.bootstrap import paired_intervals
experiment=Experiment()
display(pd.DataFrame([{'parameter':k,'value':str(v)} for k,v in CONFIG['parameters'].items()]))
display(json.loads((BASE/'reports/01_environment.json').read_text()))
''')
md('''## 1. Обучение непогодных классификаторов

Методика: бинарная классификация CatBoost, fit **только train**, без eval_set,
early stopping, весов классов и подвыборки. 500 деревьев каждого кандидата.
Фиксированный вариант depth=6, lr=0.05; четыре кандидата depth=4/8 × lr=0.03/0.1.
Числовые пропуски NaN, категориальные – отдельный код. CatBoost сохраняет
хронологический порядок train. Validation используется после каждого fit.
Метрики train описывают оптимизацию, а не обобщающую способность.
''')
code("baseline_candidates=experiment.train_variant('baseline',False)\ndisplay(baseline_candidates[['candidate','depth','learning_rate','AP','ROC_AUC','Brier','seconds']])")
md('''## 2. Обучение погодных классификаторов

Методика: та же бинарная классификация CatBoost и тот же бюджет из пяти
конфигураций. Используются одинаковые train/validation события, добавляются
только 26 принятых погодных полей. Данные не фильтруются по наличию погоды.
Флаги источника, давность и COR не входят в предикторы. Test еще не оценивается.
''')
code("weather_candidates=experiment.train_variant('weather',True)\ndisplay(weather_candidates[['candidate','depth','learning_rate','AP','ROC_AUC','Brier','seconds']])")
md('''## 3. Фиксация выбора по validation

Выбираем максимальный AP отдельно внутри каждого варианта. При равенстве –
меньшую глубину, затем learning_rate. Поиск не расширяется по test.
Выбранные конфигурации сохраняются до вызова оценки test.
''')
code('''selected=experiment.select();display(selected[['variant','candidate','depth','learning_rate','AP','ROC_AUC']])
fig,axes=plt.subplots(1,2,figsize=(12,4))
for ax,name in zip(axes,['baseline','weather']):
    f=pd.DataFrame([x for x in experiment.rows if x['variant']==name])
    ax.bar(f.candidate,f.AP,color='#2563eb');ax.set_title(name+' · validation AP');ax.tick_params(axis='x',rotation=25)
fig.tight_layout()
for ext in ['png','svg']:fig.savefig(BASE/f'figures/01_validation_candidates.{ext}',dpi=160,bbox_inches='tight')
plt.show()
fig,axes=plt.subplots(1,2,figsize=(12,4))
for ax,name in zip(axes,['baseline','weather']):
    for candidate,_,_ in experiment.candidates:
        f=pd.read_csv(BASE/f'reports/01_{name}_{candidate}_learning.csv')
        ax.plot(f.iteration,f.train_Logloss,label=candidate)
    ax.set_title(name+' · train Logloss');ax.set_xlabel('Итерация');ax.set_ylabel('Logloss')
handles,labels=axes[0].get_legend_handles_labels()
fig.legend(handles,labels,loc='upper center',ncol=5);fig.tight_layout(rect=(0,0,1,.85))
for ext in ['png','svg']:fig.savefig(BASE/f'figures/02_learning_curves.{ext}',dpi=160,bbox_inches='tight')
plt.show()
''')
md('''## 4. Оценка зафиксированной пары на test

Методика оценки классификации: AP, ROC-AUC, Brier и калибровка. Пороги
максимальной полноты при FPR<=0.1%, 0.5%, 1% выбираются только validation.
На test пороги не меняются, показываются фактическая FPR и полнота.
Включены срезы по аэропортам, месяцам и покрытию погоды. Логистический
baseline сравнивается на тех же событиях, без повторного обучения.
''')
code('''evaluation=experiment.test()
display(evaluation[evaluation.dimension.eq('all')])
comparison=[]
for name in ['baseline','weather']:
    pred=pd.read_parquet(DEST/f'{name}_test_predictions.parquet')
    old=pd.read_parquet(ROOT/f'research/weather-model-comparison/data/primary_models_v1/{name}_test_predictions.parquet').sort_values('source_row').reset_index(drop=True)
    assert pred.drop(columns='probability').equals(old.drop(columns='probability'))
    comparison.extend([{'family':'CatBoost','variant':name,**metrics(pred.GoA,pred.probability)},
                       {'family':'Logistic','variant':name,**metrics(old.GoA,old.probability)}])
comparison=pd.DataFrame(comparison);comparison.to_csv(BASE/'reports/03_family_comparison.csv',index=False);display(comparison)
fig,axes=plt.subplots(1,3,figsize=(13,4))
for ax,metric in zip(axes,['AP','ROC_AUC','Brier']):
    comparison.pivot(index='family',columns='variant',values=metric).plot.bar(ax=ax,rot=0,color=['#64748b','#2563eb'])
    ax.set_title(metric+' · test');ax.set_xlabel('');ax.get_legend().remove()
handles,_=axes[0].get_legend_handles_labels()
fig.legend(handles,['Без погоды','С погодой'],loc='upper center',ncol=2);fig.tight_layout(rect=(0,0,1,.85))
for ext in ['png','svg']:fig.savefig(BASE/f'figures/03_family_comparison.{ext}',dpi=160,bbox_inches='tight')
plt.show()
''')
md('''## 5. Парные интервалы по UTC-дням

2000 парных повторов, seed=20261011. В обоих вариантах выбираются одни и
те же дни с возвращением. Разность – с погодой минус без погоды.
Для Brier отрицательная разность лучше. Интервалы условны на выбранных
моделях, не включают неопределенность train и не устраняют всю междневную
зависимость. Взвешенные формулы AP/AUC с одинаковыми оценками проверяются
по sklearn перед вычислением интервалов.
''')
code('''a=pd.read_parquet(DEST/'baseline_test_predictions.parquet');b=pd.read_parquet(DEST/'weather_test_predictions.parquet')
assert a.drop(columns='probability').equals(b.drop(columns='probability'))
interval,bootstrap=paired_intervals(a.GoA,a.probability,b.probability,a.C40_CROSS_TIME)
interval.to_csv(BASE/'reports/04_day_intervals.csv',index=False);write_json(BASE/'reports/04_bootstrap_manifest.json',bootstrap)
display(interval);display(bootstrap)
''')
md('''## 6. Итоговые проверки и ограничения

Не объявляем систему эксплуатационно пригодной по росту AP. Оцениваем
абсолютную полноту, ложные тревоги и переносимость порогов. Важности признаков
сохранены, но не доказывают причинность. Проверки задержек/COR и независимый
период здесь не добавлены. Новые эксперименты должны иметь отдельный вопрос
и протокол; повторный подбор по просмотренному test не допускается.
''')
code('''saved=[]
for path in sorted(DEST.rglob('*')):
    if path.is_file() and path.suffix in ['.cbm','.parquet','.json'] and 'logs' not in path.parts:
        with path.open('rb') as f: saved.append({'path':path.relative_to(ROOT).as_posix(),'sha256':hashlib.file_digest(f,'sha256').hexdigest()})
write_json(BASE/'reports/05_artifact_manifest.json',{'verified':True,'fingerprint':experiment.fingerprint,'selected':experiment.selected,'candidate_count':10,'test_after_selection':True,'files':saved})
for path in (BASE/'figures').glob('*.svg'):
    path.write_text('\\n'.join(line.rstrip() for line in path.read_text(encoding='utf8').splitlines())+'\\n',encoding='utf8')
print('Все 10 кандидатов и выбранная пара проверены. Test не использовался для выбора.')
''')
n=nb.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}})
path=BASE/'notebooks/01_catboost_training.ipynb';nb.write(n,path)
class SavingClient(NotebookClient):
    async def async_execute_cell(self,*args,**kwargs):
        try:return await super().async_execute_cell(*args,**kwargs)
        finally:nb.write(self.nb,path)
try:SavingClient(n,timeout=14400,resources={'metadata':{'path':str(ROOT)}}).execute()
finally:nb.write(n,path)
print('Executed',path)
