"""Девятый ноутбук: диагностика фиксированных test-прогнозов."""
from pathlib import Path
import nbformat as nbf
root=Path(__file__).resolve().parents[1];cells=[]
def md(s):cells.append(nbf.v4.new_markdown_cell(s))
def code(s):cells.append(nbf.v4.new_code_cell(s))
md('''# Разбор ошибок baseline

## 1. Границы диагностики

Читаем сохраненные test-прогнозы 08 и те же validation-пороги. Модели не
переобучаем, пороги не подбираем, новые исключения не вводим. Разрезы заранее
выбраны для описания: аэропорт, месяц C40 UTC, missing / unseen typecode.
Январь–май 2023 не составляют полный год; последний месяц наблюдается частично.

Сравниваем две модели при разных фиксированных validation-порогах. Поэтому
разница TP / FP не является изолированным эффектом добавления типов самолета.
AP описывает ранжирование без порога, Brier — вероятности. Частота GoA влияет
на уровень AP; внутри аэропорта показываем собственную частоту метки.
Небольшие группы не получают отдельного статистического вывода по одной цифре.

Это анализ уже просмотренного test. Его выводы не используются для настройки;
следующий подбор потребует нового независимого финального периода. Историческая
доступность признаков остается условной, EGCC в test отсутствует.''')
code('''from pathlib import Path
import os,hashlib,json
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.metrics import average_precision_score,roc_auc_score
import matplotlib.pyplot as plt
from IPython.display import display,Markdown
ROOT=Path.cwd().resolve()
while not (ROOT/'reports/baseline_verification.json').exists():
    if ROOT==ROOT.parent:raise FileNotFoundError('Сначала выполните 08_baseline.ipynb')
    ROOT=ROOT.parent
R=ROOT/'reports';F=ROOT/'figures';LOCAL=ROOT/'data'
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda:stream.read(8*1024*1024),b''):h.update(block)
    return h.hexdigest()
audit=json.loads((R/'baseline_verification.json').read_text(encoding='utf-8'))
selection_path=R/'baseline_frozen_selection.json';selection_hash=sha(selection_path)
selection=json.loads(selection_path.read_text(encoding='utf-8'))
prediction_path=LOCAL/'08_test_predictions.parquet'
assert sha(prediction_path)==audit['prediction_sha256']
d=pq.read_table(prediction_path).to_pandas()
assert d.source_row.is_unique and d.source_row.is_monotonic_increasing
assert len(d)==audit['rows_test'] and int(d.GoA.sum())==audit['goa_test']
names=['geometry','geometry_aircraft'];colors={'geometry':'#00897B','geometry_aircraft':'#D65A4A'}
thresholds={n:selection['chosen'][n]['threshold'] for n in names}
for n in names:
    d[n+'_alert']=d[n].ge(thresholds[n]);d[n+'_sqerror']=(d[n]-d.GoA.astype(int))**2
d['month']=d.C40_CROSS_TIME.dt.strftime('%Y-%m')
plt.rcParams.update({'figure.dpi':120,'font.size':11,'axes.spines.top':False,
 'axes.spines.right':False,'axes.grid':True,'grid.alpha':.18})
def save(name):
    plt.tight_layout()
    for ext in ['png','svg']:plt.savefig(F/f'{name}.{ext}',bbox_inches='tight')
    plt.show()
''')
md('''## 2. Метаданные тех же test-записей

Порядковые номера строк привязаны к SHA256 исходного финального parquet.
Сопоставляем их точно, без предположения об уникальности flight_id. Сведения
используются только для диагностики групп; исходные значения не изменяются.''')
code('''entry=next(x for x in json.loads((R/'verification.json').read_text(encoding='utf-8'))['inputs'] if x['stage']=='final')
DATA=Path(os.environ.get('SAMPLE_SELECTION_DATA_ROOT',ROOT.parents[1]))/entry['file'].replace(chr(92),'/')
if not DATA.exists():DATA=LOCAL/'06_05_cases_excluded.parquet'
assert sha(DATA)==audit['input_sha256']
target=d.source_row.to_numpy();pieces=[];offset=0
reader=pq.ParquetFile(DATA)
for batch in reader.iter_batches(batch_size=200000,columns=['airport','C40_CROSS_TIME','GoA','typecode','icaoaircrafttype']):
    left=np.searchsorted(target,offset);right=np.searchsorted(target,offset+batch.num_rows)
    if right>left:
        a=batch.to_pandas().iloc[target[left:right]-offset].copy()
        a['source_row']=target[left:right];pieces.append(a)
    offset+=batch.num_rows
del reader
a=pd.concat(pieces,ignore_index=True);del pieces
assert np.array_equal(a.source_row,d.source_row)
assert a.airport.equals(d.airport) and a.C40_CROSS_TIME.equals(d.C40_CROSS_TIME) and a.GoA.equals(d.GoA)
vocab=json.loads((R/'preparation_train_vocabulary.json').read_text(encoding='utf-8'))['vocabulary']
for field in ['typecode','icaoaircrafttype']:
    raw=a[field];missing=raw.isna()|raw.eq('').fillna(False)
    d[field+'_state']=np.select([missing,raw.isin(vocab[field])],['missing','seen'],default='unseen')
del a
def measure(frame,n):
    y=frame.GoA.to_numpy(dtype=bool);p=frame[n].to_numpy();alert=frame[n+'_alert'].to_numpy()
    tp=int((alert&y).sum());fp=int((alert&~y).sum());fn=int((~alert&y).sum());tn=int((~alert&~y).sum());P=int(y.sum())
    return {'rows':len(frame),'goa':P,'prevalence':P/len(frame),
            'AP':float(average_precision_score(y,p)) if P else np.nan,
            'ROC_AUC':float(roc_auc_score(y,p)) if 0<P<len(frame) else np.nan,
            'Brier':float(frame[n+'_sqerror'].mean()),'mean_probability':float(p.mean()),
            'tp':tp,'fp':fp,'fn':fn,'tn':tn,'recall':tp/P if P else np.nan,
            'precision':tp/(tp+fp) if tp+fp else np.nan,'alert_rate':(tp+fp)/len(frame),
            'fp_per_1000':1000*fp/len(frame),'false_share_of_alerts':fp/(tp+fp) if tp+fp else np.nan,
            'fp_per_tp':fp/tp if tp else np.nan,'threshold':thresholds[n]}
tables={}
for field in ['airport','month','typecode_state','icaoaircrafttype_state']:
    rows=[]
    for value,frame in d.groupby(field,observed=True):
        for n in names:rows.append({'group':value,'model':n,**measure(frame,n)})
    tables[field]=pd.DataFrame(rows);tables[field].to_csv(R/f'errors_by_{field}.csv',index=False)
display(tables['month'].style.format(precision=4))
display(tables['typecode_state'].style.format(precision=4))
''')
md('''## 3. Совпадение тревог и пропущенные GoA

Совместные состояния дают полный баланс: обе модели, только geometry,
только geometry_aircraft, ни одна. Они описывают текущие validation-пороги,
не перебор порогов на test. Публикуем только агрегаты, не идентификаторы рейсов.''')
code('''d['alert_overlap']=np.select([d.geometry_alert&d.geometry_aircraft_alert,
 d.geometry_alert&~d.geometry_aircraft_alert,~d.geometry_alert&d.geometry_aircraft_alert],
 ['both','geometry_only','aircraft_only'],default='neither')
joint=d.groupby(['GoA','alert_overlap']).size().rename('rows').reset_index()
joint.to_csv(R/'errors_alert_overlap.csv',index=False);display(joint)
positive=d[d.GoA].groupby(['airport','month','alert_overlap']).size().rename('rows').reset_index()
positive.to_csv(R/'errors_positive_overlap_groups.csv',index=False)
airport=tables['airport'];month=tables['month']
fig,axes=plt.subplots(1,2,figsize=(12,5))
for n in names:
    q=month[month.model.eq(n)].sort_values('group')
    axes[0].plot(q.group,100*q.recall,'o-',label=n,color=colors[n])
    axes[1].plot(q.group,q.fp_per_1000,'o-',label=n,color=colors[n])
axes[0].set(ylabel='Recall, %',title='GoA, найденные при фиксированных порогах')
axes[1].set(ylabel='Ошибки FP на 1000 записей',title='Ошибки: GoA предсказан, но метка отрицательная')
for ax in axes:ax.set_xlabel('Месяц C40 UTC');ax.tick_params(axis='x',rotation=20);ax.legend()
save('28_errors_by_month')
q=airport.pivot(index='group',columns='model',values='fp').sort_values('geometry')
q.plot.barh(figsize=(11,8),color=[colors[n] for n in q.columns])
plt.xlabel('Ошибки FP');plt.ylabel('Аэропорт');plt.title('Где сосредоточены ошибки FP')
save('29_errors_false_alerts_airports')
''')
md('''## 4. Разница ранжирования и вероятностей по аэропортам

AP внутри аэропорта не складывается в pooled AP. Не приписываем глобальное
изменение AP одному аэропорту арифметически. Для Brier такое разложение точное:
вес аэропорта в test умножается на разницу средних квадратов ошибок.
Отрицательный вклад означает меньший Brier у geometry_aircraft. Это описание,
не причинное объяснение. Отдельных доверительных интервалов по аэропортам
здесь нет; сравнение по знаку не объявляется статистически значимым.''')
code('''p=airport.pivot(index='group',columns='model',values=['AP','Brier','rows','goa','prevalence'])
comparison=pd.DataFrame({'airport':p.index,'rows':p[('rows','geometry')].to_numpy(),
 'goa':p[('goa','geometry')].to_numpy(),'prevalence':p[('prevalence','geometry')].to_numpy(),
 'geometry_AP':p[('AP','geometry')].to_numpy(),'aircraft_AP':p[('AP','geometry_aircraft')].to_numpy(),
 'delta_AP':(p[('AP','geometry_aircraft')]-p[('AP','geometry')]).to_numpy(),
 'delta_Brier':(p[('Brier','geometry_aircraft')]-p[('Brier','geometry')]).to_numpy()})
comparison['test_weight']=comparison.rows/len(d)
comparison['Brier_contribution']=comparison.test_weight*comparison.delta_Brier
comparison.to_csv(R/'errors_airport_comparison.csv',index=False);display(comparison.style.format(precision=6))
global_delta=float(d.geometry_aircraft_sqerror.mean()-d.geometry_sqerror.mean())
assert np.isclose(comparison.Brier_contribution.sum(),global_delta,atol=1e-14,rtol=1e-7)
q=comparison.sort_values('delta_AP')
fig,axes=plt.subplots(1,2,figsize=(12,8))
axes[0].barh(q.airport,q.delta_AP,color=np.where(q.delta_AP>=0,'#D65A4A','#00897B'))
axes[0].axvline(0,color='#777777',lw=.8);axes[0].set_title('Разница AP внутри аэропорта');axes[0].set_xlabel('geometry_aircraft − geometry')
axes[1].barh(q.airport,q.Brier_contribution,color=np.where(q.Brier_contribution>=0,'#D65A4A','#00897B'))
axes[1].axvline(0,color='#777777',lw=.8);axes[1].set_title('Вклад аэропорта в общую разницу Brier');axes[1].set_xlabel('Вес × разница Brier')
save('30_errors_airport_differences')
q=tables['typecode_state']
fig,axes=plt.subplots(1,2,figsize=(11,4))
for ax,metric,label in [(axes[0],'recall','Recall, %'),(axes[1],'alert_rate','Доля тревог, %')]:
    q.pivot(index='group',columns='model',values=metric).mul(100).plot.bar(ax=ax,color=[colors[n] for n in names])
    ax.set_ylabel(label);ax.set_xlabel('Исходный typecode');ax.tick_params(axis='x',rotation=0)
    order=sorted(q.group.unique())
    support=q[q.model.eq('geometry')].set_index('group')
    ax.set_xticks(range(len(order)),[f'{g}\\nN={int(support.loc[g,"rows"]):,}' for g in order])
    if metric=='recall':
        for i,g in enumerate(order):
            if support.loc[g,'goa']==0:ax.text(i,0,'Нет GoA',ha='center',va='bottom',fontsize=9)
fig.suptitle('Пропуски и новые категории: описательные группы')
save('31_errors_typecode_states')
''')
md('''## 5. Проверки и выводы

Баланс ошибок в каждом разрезе должен совпадать с 08. При нуле TP отношение
FP/TP остается NaN, а не бесконечным или нулевым. Для небольшого unseen-набора
не делаем вывод об эффективности обнаружения. Данные, прогнозы и выбор порогов
проверяются на неизменность до и после анализа.''')
code('''baseline=pd.read_csv(R/'baseline_metrics.csv')
for field,table in tables.items():
    for n in names:
        a=table[table.model.eq(n)];b=baseline[baseline.split.eq('test')&baseline.model.eq(n)].iloc[0]
        for column in ['rows','goa','tp','fp','fn','tn']:assert int(a[column].sum())==int(b[column])
        assert np.isclose(np.average(a.Brier,weights=a.rows),b.Brier)
assert joint.rows.sum()==len(d) and positive.rows.sum()==int(d.GoA.sum())
assert sha(DATA)==audit['input_sha256'] and sha(prediction_path)==audit['prediction_sha256']
assert sha(selection_path)==selection_hash
verification={'input_sha256':audit['input_sha256'],'prediction_sha256':audit['prediction_sha256'],
 'selection_sha256':selection_hash,'rows_test':len(d),'goa_test':int(d.GoA.sum()),
 'group_balances_passed':True,'Brier_decomposition_passed':True,'input_unchanged':True,
 'prediction_unchanged':True,'thresholds_unchanged':True,'retraining_performed':False,
 'threshold_search_performed':False,'diagnostic_groups':['airport','month','typecode_state','icaoaircrafttype_state']}
(R/'errors_verification.json').write_text(json.dumps(verification,indent=2),encoding='utf-8')
improved=int(comparison.delta_AP.gt(0).sum());worse=int(comparison.delta_AP.lt(0).sum())
type_counts=d.typecode_state.value_counts()
text=f"""# Ошибки фиксированных baseline-моделей

Проверены {len(d):,} test-записей / {int(d.GoA.sum()):,} GoA.
Модели, прогнозы и validation-пороги не изменялись.
Внутри аэропортов AP выше у geometry_aircraft в {improved} группах, ниже в {worse}.
Это описательные знаки без отдельных интервалов; pooled AP не является их суммой.
Общая разница Brier: {global_delta:.12g}; сумма вкладов аэропортов совпадает.

Пересечение тревог:

{joint.to_string(index=False)}

Состояния typecode:

{type_counts.to_string()}

Все балансы ошибок совпадают с 08. Неполные месяцы, разное число событий и
смена состава аэропортов ограничивают сравнение периодов. Unseen-группа мала,
по ней нельзя утверждать надежное качество. Нулевая AP при отсутствии GoA
не подставляется. Историческая доступность остается условной, EGCC отсутствует.
Диагностика не выбирает новый порог или модель. Показатели при разных
validation-порогах не изолируют эффект типов самолетов. Выводы после просмотра
test могут мотивировать только отдельный будущий протокол с независимой
финальной оценкой. Публичные отчеты содержат агрегаты без идентификаторов.
"""
(R/'errors_findings.md').write_text(text,encoding='utf-8');display(Markdown(text))
''')
nb=nbf.v4.new_notebook(cells=cells,metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'}})
nbf.write(nb,root/'notebooks/09_errors.ipynb')
