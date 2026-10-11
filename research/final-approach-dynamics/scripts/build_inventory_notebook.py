"""Execute the local trajectory readiness audit; no training or downloads."""
from pathlib import Path
import nbformat as nb
from nbclient import NotebookClient
ROOT=Path(__file__).resolve().parents[3];BASE=ROOT/'research/final-approach-dynamics'
for folder in ['notebooks','reports','figures','data','Checkpoint']:(BASE/folder).mkdir(parents=True,exist_ok=True)
cells=[]
def md(s):cells.append(nb.v4.new_markdown_cell(s))
def code(s):cells.append(nb.v4.new_code_cell(s))
md('''# Динамика финального захода: первый аудит данных

Методика этого ноутбука – описательный аудит локальных траекторий, **без
обучения**. Будущая задача – бинарная классификация последующего GoA по
динамике до момента прогноза, с парой без погоды / с погодой.
Сначала проверяем, какие наблюдения действительно доступны. Существующие
пилоты создавались для геометрии C40 и интерполяции, поэтому их нельзя
автоматически считать выборкой финальных заходов с надежными метками.
''')
code('''from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display
ROOT=Path.cwd();BASE=ROOT/'research/final-approach-dynamics';OLD=ROOT/'research/c40_geometry/data'
files=sorted(OLD.rglob('*state_vectors*.csv'))
assert files
rows=[];columns=[]
for path in files:
    f=pd.read_csv(path,dtype={'icao24':str})
    assert {'time','icao24','lat','lon'}.issubset(f.columns)
    utc=pd.to_datetime(f.time,unit='s',utc=True)
    ordered=f.sort_values(['icao24','time'])
    gaps=ordered.groupby('icao24').time.diff();positive=gaps[gaps>0]
    age=f.time-f.lastposupdate if 'lastposupdate' in f else pd.Series(dtype=float)
    with path.open('rb') as stream:digest=hashlib.file_digest(stream,'sha256').hexdigest()
    rows.append({'source':path.relative_to(ROOT).as_posix(),'sha256':digest,'bytes':path.stat().st_size,'rows':len(f),'aircraft':f.icao24.nunique(),'utc_min':str(utc.min()),'utc_max':str(utc.max()),'median_positive_spacing_s':positive.median(),'p95_positive_spacing_s':positive.quantile(.95),'p95_position_age_s':age.quantile(.95),'negative_position_age_rows':int((age<0).sum()),'duplicate_aircraft_time_rows':int(f.duplicated(['icao24','time']).sum()),'has_GoA_label':'GoA' in f})
    for field in f:
        columns.append({'source':path.relative_to(ROOT).as_posix(),'field':field,'nonmissing_rows':int(f[field].notna().sum()),'nonmissing_fraction':f[field].notna().mean()})
inventory=pd.DataFrame(rows);fields=pd.DataFrame(columns)
inventory.to_csv(BASE/'reports/00_local_sources.csv',index=False);fields.to_csv(BASE/'reports/00_field_coverage.csv',index=False)
display(inventory)
display(fields.groupby('field').agg(files=('source','nunique'),mean_nonmissing_fraction=('nonmissing_fraction','mean')))
''')
md('''## Интервалы наблюдений и свежесть позиций

Время строки и время обновления позиции различаются. Медиана и 95-й
процентиль интервала вычисляются внутри ICAO24 по положительным разностям.
Дубли отдельно отмечены, отрицательный возраст позиции не исправляется.
Число строк разных файлов не является числом независимых событий: источники
могут перекрываться. Оценки относятся только к этим локальным пилотам.
''')
code('''labels=[Path(x).parent.name+'/'+Path(x).name.replace('_state_vectors','') for x in inventory.source]
fig,axes=plt.subplots(1,2,figsize=(14,6))
axes[0].barh(labels,inventory.rows,color='#64748b');axes[0].set(title='Строки в локальных источниках',xlabel='Число строк')
axes[1].barh(labels,inventory.p95_positive_spacing_s,color='#2563eb');axes[1].set(title='95-й процентиль интервалов',xlabel='Секунды')
for ax in axes:ax.grid(axis='x',alpha=.2)
fig.tight_layout()
for ext in ['png','svg']:fig.savefig(BASE/f'figures/00_source_inventory.{ext}',dpi=160,bbox_inches='tight')
plt.show()
fig,ax=plt.subplots(figsize=(12,6))
ax.barh(labels,inventory.p95_position_age_s,color='#d97706')
ax.set(title='95-й процентиль возраста позиции',xlabel='Секунды от lastposupdate до времени строки')
ax.grid(axis='x',alpha=.2);fig.tight_layout()
for ext in ['png','svg']:fig.savefig(BASE/f'figures/02_position_freshness.{ext}',dpi=160,bbox_inches='tight')
plt.show()
''')
md('''## Пример последовательности движения

Показываем наблюдения одного самолета из локального пилота. Горизонтальная
ось – время относительно первого показанного наблюдения, без выравнивания
по будущему касанию или уходу. Окно не считается установленным финальным
участком или моментом захвата ILS. Поля показаны в исходных единицах файла;
их происхождение и преобразования должны быть проверены до расчета признаков.
Не отождествляем velocity с IAS, heading с курсом носа и baroaltitude с AGL.
''')
code('''path=OLD/'external_day/state_vectors.csv';raw=pd.read_csv(path,dtype={'icao24':str})
aircraft=raw.groupby('icao24').size().idxmax()
example=raw[raw.icao24==aircraft].sort_values('time').drop_duplicates('time')
minutes=(example.time-example.time.min())/60
fig,axes=plt.subplots(3,1,figsize=(10,8),sharex=True)
for ax,field in zip(axes,['velocity','baroaltitude','vertrate']):
    ax.plot(minutes,example[field],'.-',color='#2563eb',markersize=3)
    ax.set_ylabel(field+' · исходные единицы');ax.grid(alpha=.2)
axes[0].set_title('Один локальный траекторный пример; стадия захода не подтверждена')
axes[-1].set_xlabel('Минуты от первого наблюдения');fig.tight_layout()
for ext in ['png','svg']:fig.savefig(BASE/f'figures/01_local_sequence.{ext}',dpi=160,bbox_inches='tight')
plt.show()
display(pd.DataFrame([{'rows':len(example),'duration_minutes':float(minutes.max()),'GoA_label_present':'GoA' in example,'ILS_capture_observed':False,'final_segment_verified':False}]))
''')
md('''## Готовность к новому исследованию

Наличие рядов подтверждает возможность анализа динамики, но не готовность
выборки. Пилоты предыдущего исследования не заменяют проверку покрытий и
связи с исходами. До обучения нужны обе группы, единицы и момент прогноза.
''')
code('''readiness=pd.DataFrame([
 {'requirement':'Локальные временные последовательности','status':'есть','next':'Проверить покрытие финальных участков'},
 {'requirement':'Прямая метка GoA в файлах state vectors','status':'есть' if inventory.has_GoA_label.any() else 'нет','next':'Связать с исходной разметкой и проверить попытки'},
 {'requirement':'Наблюдаемый захват ILS','status':'не подтвержден','next':'Отделить факт захвата от геометрического приближения'},
 {'requirement':'Единицы и история преобразований','status':'нужна проверка','next':'Проверить источник каждого поля'},
 {'requirement':'Момент, окно и горизонт прогноза','status':'не заданы','next':'Зафиксировать правило из доступных к моменту данных'},
 {'requirement':'Достаточное покрытие GoA и обычных посадок','status':'не проверено','next':'Построить аудит связанных финальных участков'},
 {'requirement':'Погода до нового момента прогноза','status':'новый join не построен','next':'Повторно привязать METAR к этому моменту, не брать C40 join'},
 {'requirement':'Разделение train/validation/test','status':'не задано','next':'Разделить по времени с группировкой попыток одного полета'}])
readiness.to_csv(BASE/'reports/00_readiness.csv',index=False);display(readiness)
manifest={'method':'descriptive local trajectory inventory; no fitting','source_files':len(inventory),'source_files_with_GoA':int(inventory.has_GoA_label.sum()),'training_started':False,'final_segment_verified':False,'weather_join_built':False,'sources':inventory[['source','sha256']].to_dict('records')}
(BASE/'reports/00_inventory_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\\n',encoding='utf8')
for path in (BASE/'figures').glob('*.svg'):path.write_text('\\n'.join(x.rstrip() for x in path.read_text(encoding='utf8').splitlines())+'\\n',encoding='utf8')
''')
n=nb.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}})
path=BASE/'notebooks/00_data_inventory.ipynb';nb.write(n,path)
try:NotebookClient(n,timeout=600,resources={'metadata':{'path':str(ROOT)}}).execute()
finally:nb.write(n,path)
print('Executed local trajectory inventory; no training.')
