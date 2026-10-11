"""Render scientific diagnostics from fixed predictions; never fit a model."""
from pathlib import Path
import nbformat as nb
from nbclient import NotebookClient

ROOT=Path(__file__).resolve().parents[3]
BASE=ROOT/'research/boosting-weather-comparison'
cells=[]
def md(s):cells.append(nb.v4.new_markdown_cell(s))
def code(s):cells.append(nb.v4.new_code_cell(s))
md('''# CatBoost: диагностика выбранной пары

Методика – оценка бинарной классификации GoA по сохраненным прогнозам.
Обучения и подбора параметров в этом ноутбуке нет. Пороги зафиксированы по
validation. Test уже просмотрен в предыдущем исследовании, поэтому сравнение
семейств имеет исследовательский статус. Результаты не подтверждают готовность
к эксплуатации. AP зависит от распространенности GoA; важность признака
не является причинным эффектом погоды.
''')
code('''from pathlib import Path
import json,sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
from catboost import CatBoostClassifier
from sklearn.metrics import precision_recall_curve
ROOT=Path.cwd();BASE=ROOT/'research/boosting-weather-comparison';DEST=ROOT/'artifacts/boosting-weather-comparison'
sys.path.insert(0,str(ROOT/'src'))
from approach_risk.evaluation.classification import metrics
plt.rcParams.update({'figure.dpi':120,'font.size':10})
pred={name:pd.read_parquet(DEST/f'{name}_test_predictions.parquet') for name in ['baseline','weather']}
assert pred['baseline'].drop(columns='probability').equals(pred['weather'].drop(columns='probability'))
selection=json.loads((BASE/'reports/02_selection.json').read_text(encoding='utf8'))
evaluation=pd.read_csv(BASE/'reports/03_test_evaluation.csv')
display(pd.read_csv(BASE/'reports/01_validation_candidates.csv')[['variant','candidate','depth','learning_rate','AP','ROC_AUC','Brier']])
display(selection)
def save(fig,name):
    for ext in ['png','svg']:
        path=BASE/f'figures/{name}.{ext}'
        fig.savefig(path,dpi=160,bbox_inches='tight')
        if ext=='svg':path.write_text('\\n'.join(line.rstrip() for line in path.read_text(encoding='utf8').splitlines())+'\\n',encoding='utf8')
''')
md('''## Precision–recall на test

Пунктирная линия показывает долю GoA. Левый график показывает весь диапазон,
правый – первые 20% полноты. График не служит для выбора рабочего порога.
''')
code('''fig,axes=plt.subplots(1,2,figsize=(12,4))
for name,label,color in [('baseline','Без погоды','#64748b'),('weather','С погодой','#2563eb')]:
    f=pred[name];precision,recall,_=precision_recall_curve(f.GoA,f.probability)
    for ax in axes:ax.plot(recall,precision,label=label,color=color,linewidth=1)
for ax in axes:
    ax.axhline(pred['baseline'].GoA.mean(),color='black',linestyle=':',label='Доля GoA')
    ax.set_xlabel('Полнота');ax.set_ylabel('Точность');ax.set_yscale('log');ax.grid(alpha=.2)
axes[0].set_title('Полный диапазон');axes[1].set_title('Полнота до 20%');axes[1].set_xlim(0,.2)
handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc='upper center',ncol=3)
fig.tight_layout(rect=(0,0,1,.85));save(fig,'04_precision_recall');plt.show()
''')
md('''## Калибровка

Используем десять групп приблизительно одинакового размера по оценке риска.
Показываем средний прогноз и фактическую долю GoA, а также число наблюдений
и событий. Это описательная проверка; калибратор по test не обучаем.
''')
code('''fig,ax=plt.subplots(figsize=(7,5));tables=[]
for name,label,color in [('baseline','Без погоды','#64748b'),('weather','С погодой','#2563eb')]:
    f=pred[name].copy();f['bin']=pd.qcut(f.probability,10,duplicates='drop')
    table=f.groupby('bin',observed=True).agg(n=('GoA','size'),events=('GoA','sum'),predicted=('probability','mean'),observed=('GoA','mean')).reset_index()
    table['bin']=table['bin'].astype(str);table['variant']=name;tables.append(table)
    ax.plot(table.predicted,table.observed,'o-',label=label,color=color)
limit=max(t[['predicted','observed']].to_numpy().max() for t in tables)*1.1
ax.plot([0,limit],[0,limit],':',color='black',label='Идеальная калибровка')
ax.set(xlabel='Средняя предсказанная вероятность',ylabel='Фактическая доля GoA',title='Калибровка по децилям · test')
ax.legend(loc='best');ax.grid(alpha=.2);fig.tight_layout();save(fig,'05_calibration');plt.show()
calibration=pd.concat(tables,ignore_index=True);calibration.to_csv(BASE/'reports/07_calibration_deciles.csv',index=False);display(calibration)
''')
md('''## Перенос порогов validation → test

Для каждого заданного ограничения FPR порог выбирается только на validation.
На test отдельно считаем фактический FPR, полноту, точность, TP и FP.
Одинаковое ограничение при выборе не гарантирует одинаковый FPR в новом периоде.
''')
code('''thresholds=json.loads((BASE/'reports/03_thresholds.json').read_text(encoding='utf8'));rows=[]
for name,f in pred.items():
    for budget,threshold in thresholds[name].items():
        alarm=f.probability>=threshold;y=f.GoA.astype(bool)
        tp=int((alarm&y).sum());fp=int((alarm&~y).sum())
        rows.append({'variant':name,'validation_FPR_budget':float(budget),'threshold':threshold,'test_recall':tp/y.sum(),'test_FPR':fp/(~y).sum(),'test_precision':tp/(tp+fp) if tp+fp else np.nan,'TP':tp,'FP':fp})
operating=pd.DataFrame(rows);operating.to_csv(BASE/'reports/07_operating_points.csv',index=False);display(operating)
fig,axes=plt.subplots(1,2,figsize=(12,4))
for ax,metric,title in zip(axes,['test_recall','test_FPR'],['Полнота на test','Фактический FPR на test']):
    operating.pivot(index='validation_FPR_budget',columns='variant',values=metric).plot.bar(ax=ax,rot=0,color=['#64748b','#2563eb'])
    ax.set_title(title);ax.set_xlabel('Ограничение FPR на validation');ax.get_legend().remove();ax.grid(axis='y',alpha=.2)
handles,_=axes[0].get_legend_handles_labels();fig.legend(handles,['Без погоды','С погодой'],loc='upper center',ncol=2)
fig.tight_layout(rect=(0,0,1,.85));save(fig,'06_operating_points');plt.show()
''')
md('''## Срезы по аэропортам, месяцам и покрытию погоды

Это описательные срезы заранее определенных групп. Показываем объем и число
GoA, чтобы не сравнивать только точечные метрики при разном числе событий.
Поиск лучшего аэропорта или месяца по test не является валидацией модели.
''')
code('''for dimension in ['airport','period','weather_coverage']:
    display(evaluation[evaluation.dimension==dimension])
airport=evaluation[(evaluation.dimension=='airport') & evaluation.AP.notna()]
fig,ax=plt.subplots(figsize=(10,4))
airport.pivot(index='group',columns='model',values='AP').plot.bar(ax=ax,rot=30,color=['#64748b','#2563eb'])
ax.set(title='AP по аэропортам · test',xlabel='Аэропорт',ylabel='AP');ax.grid(axis='y',alpha=.2)
handles,_=ax.get_legend_handles_labels();ax.get_legend().remove();fig.legend(handles,['Без погоды','С погодой'],loc='upper center',ncol=2)
fig.tight_layout(rect=(0,0,1,.85));save(fig,'08_airport_AP');plt.show()
''')
md('''## Важности признаков и ограничения интерпретации

Важности CatBoost относятся к конкретной обученной модели и ее набору
признаков. Коррелирующие признаки могут делить важность. Ни ранжирование,
ни прирост AP не доказывают причинность; прогноз может отражать аэропорт,
сезон, покрытие METAR и другие зависимости.
''')
code('''fig,axes=plt.subplots(1,2,figsize=(12,6))
for ax,name,title in zip(axes,['baseline','weather'],['Без погоды','С погодой']):
    model=CatBoostClassifier();model.load_model(str(DEST/name/selection['selected'][name]/'model.cbm'))
    values=model.get_feature_importance(type='PredictionValuesChange')
    assert len(values)==len(model.feature_names_) and np.isfinite(values).all() and values.sum()>0
    full=pd.DataFrame({'feature':model.feature_names_,'importance':values})
    full.to_csv(BASE/f'reports/03_{name}_importance.csv',index=False)
    display(full.sort_values('importance',ascending=False))
    importance=full.nlargest(12,'importance').sort_values('importance')
    ax.barh(importance.feature,importance.importance,color='#64748b' if name=='baseline' else '#2563eb')
    ax.set_title(title);ax.set_xlabel('Важность CatBoost');ax.grid(axis='x',alpha=.2)
fig.tight_layout();save(fig,'07_feature_importance');plt.show()
display(pd.read_csv(BASE/'reports/04_day_intervals.csv'))
''')
n=nb.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}})
path=BASE/'notebooks/02_catboost_diagnostics.ipynb';nb.write(n,path)
try:NotebookClient(n,timeout=600,resources={'metadata':{'path':str(ROOT)}}).execute()
finally:nb.write(n,path)
print('Executed',path)
