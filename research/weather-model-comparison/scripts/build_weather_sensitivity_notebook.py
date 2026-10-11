"""Execute report of fixed binary-classifier delay/COR sensitivity; no fit."""
from pathlib import Path
import nbformat as nb
from nbclient import NotebookClient
ROOT=Path(__file__).resolve().parents[3];BASE=ROOT/'research/weather-model-comparison'
cells=[nb.v4.new_markdown_cell('''# 06. Задержки METAR и исключение COR

Методика: бинарная классификация GoA фиксированным логистическим
классификатором L2 C=1, обученным ранее на сценарии 15 минут с COR.
Повторного fit нет. Преобразования и пороги validation заморожены.
Меняется только погодный вход для тех же 546577 test-событий.

Сценарии: задержки 0, 15, 30 минут; без COR при 15 минутах. Максимальная
давность 60 минут, доступность строго до C40. При исключении COR выбирается
последнее оставшееся сообщение, а не просто блокируется прежняя связь.
Неоднозначность времени определяется по исходному архиву до исключения COR.
Все 26 полей получают те же числовые маски и правила проверенного ядра.
Отсутствие сообщения не удаляет событие. Это допущения доступности,
а не подтвержденные времена получения исторической погоды.
'''),nb.v4.new_code_cell('''from pathlib import Path
import hashlib,json
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
BASE=Path.cwd()/'research/weather-model-comparison'
m=json.loads((BASE/'reports/06_weather_scenario_manifest.json').read_text())
assert m['verified'] and m['primary_reproduced'] and not m['fit_executed'] and not m['thresholds_retuned']
for entry in m['files']:
    with (BASE/'data/weather_sensitivity_v1'/entry['file']).open('rb') as f:assert hashlib.file_digest(f,'sha256').hexdigest()==entry['sha256']
summary=pd.read_csv(BASE/'reports/06_weather_scenario_summary.csv')
thresholds=pd.read_csv(BASE/'reports/06_weather_scenario_thresholds.csv')
display(summary);display(thresholds)
assert summary.rows.eq(546577).all() and summary.GoA.eq(1739).all()
assert summary.loc[summary.scenario.eq('primary_reproduction'),'information_changed'].eq(0).all()
'''),nb.v4.new_markdown_cell('''## Проверка воспроизведения и изменения качества

Основной сценарий 15 минут заново выбран из архива. Все 26 модельных
полей совпали с исходным модельным снимком, вероятности — с сохраненными
прогнозами до 1e-12. Это проверяет согласованность альтернативной сборки.
Доли изменения информации и вероятностей используют все test-события.
Пороги не выбираются заново; фактическая FPR может отличаться между сценариями.
'''),nb.v4.new_code_cell('''order=['delay0','primary_reproduction','delay30','no_cor']
plot=summary.set_index('scenario').loc[order]
labels=['0 минут','15 минут','30 минут','Без COR']
fig,axes=plt.subplots(1,3,figsize=(14,4))
for ax,metric in zip(axes,['AP','ROC_AUC','information_changed']):
    values=plot[metric].to_numpy()
    if metric=='information_changed':values=100*values/546577
    ax.bar(labels,values,color=['#64748b','#2563eb','#64748b','#64748b'])
    ax.set_title(metric if metric!='information_changed' else 'Изменение информации, %')
fig.tight_layout()
for ext in ['png','svg']:
    p=BASE/f'figures/06_delay_cor_sensitivity.{ext}'
    fig.savefig(p,dpi=160,bbox_inches='tight')
    if ext=='svg':p.write_text('\\n'.join(s.rstrip() for s in p.read_text(encoding='utf8').splitlines())+'\\n',encoding='utf8')
plt.show()
'''),nb.v4.new_markdown_cell('''## Ограничения

Это проверка фиксированной модели при изменении входа, а не сравнение
моделей, заново обученных на каждом сценарии. Test уже использовался;
выбирать по этим результатам лучший сценарий для подтверждающей оценки нельзя.
Основная пара, дневные интервалы, абляция типа и зазор рассмотрены отдельно.
Низкая полнота GoA и переносимость порогов ограничивают практический вывод.
''')]
n=nb.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}})
p=BASE/'notebooks/06_delay_cor_sensitivity.ipynb';nb.write(n,p)
try:NotebookClient(n,timeout=600,resources={'metadata':{'path':str(ROOT)}}).execute()
finally:nb.write(n,p)
print('Executed',p)
