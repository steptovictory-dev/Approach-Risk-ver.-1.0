"""Present the completed binary-classification training run without refitting."""
from pathlib import Path
import nbformat as nb
from nbclient import NotebookClient
ROOT=Path(__file__).resolve().parents[3]; BASE=ROOT/'research/weather-model-comparison'
method=(BASE/'methodology.md').read_text(encoding='utf8')
cells=[nb.v4.new_markdown_cell('# 04. Основная пара: обучение и оценка\n\n'+method),
nb.v4.new_markdown_cell('''## Выполненный запуск

Обучение выполнено scripts/train_primary.py после разрешения пользователя.
Этот ноутбук читает сохраненные модели и прогнозы, проверяет артефакты и
показывает результаты без повторного fit. Код запуска и методика доступны
по ссылкам [train_primary.py](../scripts/train_primary.py) и
[methodology.md](../methodology.md). Результаты относятся к основной паре.
'''),nb.v4.new_code_cell('''from pathlib import Path
import hashlib,json
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
ROOT=Path.cwd(); BASE=ROOT/'research/weather-model-comparison'
m=json.loads((BASE/'reports/04_training_manifest.json').read_text(encoding='utf8'))
assert m['training_executed'] and all(x['converged'] for x in m['models'])
for entry in m['files']:
    with (BASE/'data/primary_models_v1'/entry['file']).open('rb') as f:
        assert hashlib.file_digest(f,'sha256').hexdigest()==entry['sha256']
display(pd.DataFrame(m['models']))
frames=[]
for split in ['validation','test']:
    f=pd.read_csv(BASE/f'reports/04_{split}_evaluation.csv')
    f['split']=split;frames.append(f)
evaluation=pd.concat(frames,ignore_index=True)
overall=evaluation[evaluation.dimension.eq('all')]
display(overall[overall.AP.notna()][['split','model','rows','GoA','AP','ROC_AUC','Brier']])
display(overall[overall.target_validation_FPR.notna()][['split','model','target_validation_FPR','recall','precision','FPR','TP','FP']])
'''),nb.v4.new_markdown_cell('''## Основные метрики и калибровка

AP оценивает качество ранжирования редкого класса, ROC-AUC — разделение
классов, Brier — ошибку вероятностей. Доля GoA является ориентиром AP
случайного ранжирования. Рост метрик еще не означает практическую пригодность.
'''),nb.v4.new_code_cell('''table=overall[(overall.split=='test')&overall.AP.notna()].set_index('model')
fig,axes=plt.subplots(1,3,figsize=(13,4))
for ax,metric in zip(axes,['AP','ROC_AUC','Brier']):
    table[metric].plot.bar(ax=ax,color=['#64748b','#2563eb'],rot=0)
    ax.set_title(metric+' · test 2023');ax.set_xlabel('')
    if metric=='AP': ax.axhline(1739/546577,color='red',linestyle='--',label='Доля GoA');ax.legend()
    if metric=='ROC_AUC': ax.axhline(.5,color='red',linestyle='--')
fig.tight_layout()
for ext in ['png','svg']:
    path=BASE/f'figures/04_primary_test_metrics.{ext}'
    fig.savefig(path,dpi=160,bbox_inches='tight')
    if ext=='svg': path.write_text('\\n'.join(s.rstrip() for s in path.read_text(encoding='utf8').splitlines())+'\\n',encoding='utf8')
plt.show()
print('Пока не выполнены:',m['pending'])
'''),nb.v4.new_markdown_cell('''## Ограничения и продолжение

Test уже использовался ранее; признаки имеют условную доступность.
Дневные интервалы, абляция типа самолета, маска зазора и проверки задержки/COR
еще не выполнены. Полный протокол не завершен. Выигрыш основной пары
следует трактовать как ретроспективный результат для этой выборки.
''')]
n=nb.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}})
path=BASE/'notebooks/04_primary_training.ipynb';nb.write(n,path)
try: NotebookClient(n,timeout=600,resources={'metadata':{'path':str(ROOT)}}).execute()
finally: nb.write(n,path)
print('Executed',path)
