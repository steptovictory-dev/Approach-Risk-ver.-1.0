"""Executed report for binary GoA classification sensitivity runs; no refit."""
from pathlib import Path
import nbformat as nb
from nbclient import NotebookClient
ROOT=Path(__file__).resolve().parents[3];BASE=ROOT/'research/weather-model-comparison'
cells=[nb.v4.new_markdown_cell('''# 05. Устойчивость основной пары

**Методика перед обучением:** обучение с учителем, бинарная классификация
GoA логистическим классификатором L2, C=1. Преобразования fit только на train;
пороги только validation. В run_model_sensitivity.py обучены пары без
typecode и с зазором 24 часа. Здесь читаем сохраненные результаты без refit.
Test ранее использовался. Сравнение разных масок не является парным
сравнением на разных строках: внутри каждой пары популяция одинакова.

Для основной пары bootstrap 2000 раз выбирает UTC-дни с возвращением;
в обоих вариантах используются одинаковые дни. AP и AUC рассчитываются
точно с весами кратности дня и одинаковыми оценками. Формулы проверены
независимо по sklearn. Brier: отрицательная разность означает улучшение.
Интервалы условны на обученных моделях и не включают неопределенность train.
'''),nb.v4.new_code_cell('''from pathlib import Path
import hashlib,json
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
BASE=Path.cwd()/'research/weather-model-comparison'
rows=[]
for name,prefix,folder in [('Основная пара','04','primary_models_v1'),('Без типа самолета','05_no_aircraft','no_aircraft_models_v1'),('Зазор 24 часа','05_gap','gap_models_v1')]:
    m=json.loads((BASE/f'reports/{prefix}_training_manifest.json').read_text(encoding='utf8'))
    assert all(x['converged'] for x in m['models'])
    for entry in m['files']:
        with (BASE/'data'/folder/entry['file']).open('rb') as f: assert hashlib.file_digest(f,'sha256').hexdigest()==entry['sha256']
    f=pd.read_csv(BASE/f'reports/{prefix}_test_evaluation.csv')
    q=f[f.dimension.eq('all')&f.AP.notna()].copy();q['scenario']=name;rows.append(q)
summary=pd.concat(rows,ignore_index=True)
summary[['scenario','model','rows','GoA','AP','ROC_AUC','Brier']].to_csv(BASE/'reports/05_sensitivity_summary.csv',index=False)
display(summary[['scenario','model','rows','GoA','AP','ROC_AUC','Brier']])
interval=pd.read_csv(BASE/'reports/05_primary_day_bootstrap.csv');display(interval)
display(json.loads((BASE/'reports/05_bootstrap_manifest.json').read_text()))
'''),nb.v4.new_code_cell('''fig,axes=plt.subplots(1,2,figsize=(12,4))
for ax,metric in zip(axes,['AP','ROC_AUC']):
    summary.pivot(index='scenario',columns='model',values=metric).plot.bar(ax=ax,rot=0,color=['#64748b','#2563eb'])
    ax.set_title(metric+' · test 2023');ax.set_xlabel('')
fig.tight_layout()
for ext in ['png','svg']:
    path=BASE/f'figures/05_sensitivity_comparison.{ext}'
    fig.savefig(path,dpi=160,bbox_inches='tight')
    if ext=='svg':path.write_text('\\n'.join(s.rstrip() for s in path.read_text(encoding='utf8').splitlines())+'\\n',encoding='utf8')
plt.show()
'''),nb.v4.new_markdown_cell('''## Границы вывода

Эти проверки характеризуют устойчивость внутри заданного семейства моделей
и ретроспективной выборки. Они не подтверждают причинность, эксплуатационную
пригодность или доступность METAR в реальном времени. Зависимость между днями
bootstrap полностью не устраняет. Следующая проверка — изменение задержек
0/30 минут и исключение COR при фиксированном классификаторе и порогах.
''')]
n=nb.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}})
p=BASE/'notebooks/05_model_sensitivity.ipynb';nb.write(n,p)
try:NotebookClient(n,timeout=600,resources={'metadata':{'path':str(ROOT)}}).execute()
finally:nb.write(n,p)
print('Executed',p)
