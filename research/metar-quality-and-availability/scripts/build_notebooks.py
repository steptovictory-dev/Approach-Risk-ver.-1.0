"""Выполненные научные ноутбуки: методика, чувствительность и валидация."""
from pathlib import Path
import json
import sys
import nbformat as nbf
from nbclient import NotebookClient
from jupyter_client.manager import KernelManager

BASE=Path(__file__).resolve().parents[1]
R=BASE/'reports'
SETUP="""from pathlib import Path
import hashlib,json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display,Markdown
BASE=Path.cwd().resolve().parent
PROJECT=BASE.parents[1]
R,F=BASE/'reports',BASE/'figures'
GREEN,RED='#00897B','#D65A4A'
plt.rcParams.update({'figure.dpi':120,'font.size':11,'axes.grid':True,'grid.alpha':.15,'axes.spines.top':False,'axes.spines.right':False})
def sha(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
def save(name):
    plt.tight_layout()
    for ext in ['png','svg']:plt.savefig(F/f'{name}.{ext}',bbox_inches='tight')
    plt.show()
    plt.close()
manifest=json.loads((R/'02_study_manifest.json').read_text(encoding='utf8'))
assert sha(PROJECT/'europe/data/analytical/06_05_cases_excluded.parquet')==manifest['input_sha256']
assert sha(BASE/'protocol.md')==manifest['protocol_sha256']
assert manifest['primary_reproduced'] and manifest['all_rows_once'] and not manifest['training_executed']
summary=pd.read_csv(R/'02_scenario_summary.csv')
strata=pd.read_csv(R/'02_scenario_strata.csv')
pd.set_option('display.max_columns',20)
pd.set_option('display.max_rows',50)
"""

def create():
    cells=[]
    def md(text):cells.append(nbf.v4.new_markdown_cell(text))
    def code(text):cells.append(nbf.v4.new_code_cell(text))
    return cells,md,code

def methods():
    cells,md,code=create()
    md(r'''# 01. Научный вопрос, выборка и методика

**Насколько правила условной доступности и качества METAR изменяют покрытие
и состав информации на C40 в фиксированной европейской выборке?**

Весь корпус: 6 331 485 событий и 18 349 меток GoA, 19 аэропортов.
Ретроспективный анализ свойств данных, без обучения модели. Основной сценарий
согласован ранее; исследовательские сценарии зафиксированы до этих расчетов.

## Временной выбор

Для события i время C40 обозначено t_i, время наблюдения сообщения j - o_j.
Предполагаемая задержка d не равна измеренному времени получения.

$$j^*(i;d)=\arg\max_j\{o_j:o_j+d<t_i\}.$$

Для предельной давности a покрытие наблюдениями:

$$A_i(a,d)=\mathbf1\{j^*\text{ однозначен},\ t_i-o_{j^*}\leq a\}.$$

Проверка основного текста Q_j и числовая маска M_jk дают:

$$U_i=A_iQ_{j^*},\qquad
x^{model}_{ik}=\begin{cases}x_{j^*k},&U_i=1, M_{j^*k}=0;\\
\mathrm{NA},&\text{иначе}.\end{cases}$$

Числовая маска не уничтожает все поля строки; непроверенный основной блок
оставляет все модельные погодные поля неизвестными. Отсутствующая порывная
группа не означает нулевой порыв. COR не является погодным явлением.
''')
    code(SETUP)
    code("""display(pd.DataFrame(manifest['scenarios']))
primary=strata.loc[strata.scenario.eq('primary')]
display(primary.groupby('GoA')[['rows','observation','usable','cor_usable']].sum().reset_index())
airport=primary.groupby(['airport','GoA'])[['rows','usable']].sum().reset_index()
airport['usable_percent']=100*airport.usable/airport.rows
display(airport)
fig,ax=plt.subplots(figsize=(11,5))
for label,color,title in [(False,GREEN,'Обычные посадки'),(True,RED,'GoA')]:
    group=airport.loc[airport.GoA.eq(label)].set_index('airport').sort_index()
    ax.plot(group.index,group.usable_percent,marker='o',color=color,label=title)
ax.set(ylabel='Допустимый погодный блок, %',title='Покрытие основного сценария по аэропортам')
ax.tick_params(axis='x',rotation=60)
ax.legend()
save('01_primary_airport_coverage')
""")
    md('''## Причины недопустимости основного блока

Знаменатель - все события. Причины взаимоисключающие; это не классификация
причин отсутствия каждого отдельного числового поля.
''')
    code("""reasons=pd.read_csv(R/'03_missing_reasons.csv')
counts=reasons.loc[reasons.scenario.eq('primary')].groupby(['GoA','reason']).rows.sum().reset_index()
display(counts)
excluded=counts.loc[counts.reason.ne('usable')]
fig,ax=plt.subplots(figsize=(10,5))
pivot=excluded.pivot(index='reason',columns='GoA',values='rows').fillna(0)
for label in pivot.columns:
    pivot[label]=100*pivot[label]/primary.loc[primary.GoA.eq(label),'rows'].sum()
pivot.rename(index={'no_previous_report':'Нет предыдущего сообщения','ambiguous_time':'Неоднозначное время',
    'older_than_limit':'Превышена давность','unverified_core':'Непроверенный основной текст'}).plot.barh(ax=ax,color=[GREEN,RED])
ax.set(xlabel='Доля событий своей метки, %',ylabel='',title='Недопустимость погодного блока')
ax.legend(['Обычные посадки','GoA'])
save('02_primary_missing_reasons')
season=primary.groupby(['season','GoA'])[['rows','usable']].sum().reset_index()
season['usable_percent']=100*season.usable/season.rows
display(season)
""")
    md('''Источник и популяция уже отобраны предыдущим исследованием. Это
покрытие внутри фиксированного аналитического входа, не всех реальных посадок.
Сезоны DJF/MAM/JJA/SON - календарные группы месяцев по UTC. Изменение состава
аэропортов и лет не позволяет трактовать различия как причинный сезонный эффект.
Полные расчеты по событиям воспроизводятся scripts/run_study.py.
''')
    return nbf.v4.new_notebook(cells=cells)

def sensitivity():
    cells,md,code=create()
    md(r'''# 02. Чувствительность покрытия и погодных признаков

Для сценария s доля допустимых погодных блоков:
$$C_s=\frac{1}{N}\sum_i U_{is}.$$
Смена выбранного сообщения и изменение информации различаются:
$$S_s=\frac{1}{N}\sum_i\mathbf1\{j^*_{is}\ne j^*_{i0}\},$$
$$I_s=\frac{1}{N}\sum_i\mathbf1\{\exists k: x_{isk}\ne x_{i0k}
\text{ или изменился статус пропуска}\}.$$

I считается по 13 полям, перечисленным в таблицах ниже, а не по всем
26 погодным полям. Разница чисел при двух пропусках не считается изменением.
Для направления ветра используется круговая разность:
$$D_{dir}=\min(|\theta_s-\theta_0|,360-|\theta_s-\theta_0|).$$
''')
    code(SETUP)
    code("""display(summary[['scenario','GoA','rows','usable_percent','message_changed_percent','information_changed_percent']])
overall=summary.groupby('scenario')[['rows','usable','message_changed','information_changed']].sum()
for field in ['usable','message_changed','information_changed']:overall[field+'_percent']=100*overall[field]/overall.rows
display(overall.reset_index())
fig,axes=plt.subplots(1,2,figsize=(13,5))
overall.usable_percent.plot.bar(ax=axes[0],color=GREEN,label='Допустимый погодный блок')
axes[0].set(ylabel='Доля всех событий, %',title='Покрытие при разных правилах',ylim=(0,100))
overall[['message_changed_percent','information_changed_percent']].plot.bar(ax=axes[1],color=[RED,GREEN])
axes[1].set(ylabel='Доля всех событий, %',title='Изменения относительно primary')
axes[1].legend(['Сменилось сообщение','Изменились значения или пропуски'],fontsize=8)
axes[0].legend(fontsize=8)
for ax in axes:ax.set_xlabel('Сценарий')
save('03_coverage_and_information')
""")
    md('''## Неоднородность по аэропортам

Проценты рассчитаны внутри аэропорта. Они не стандартизованы по годам,
сезонам или метке. В no_cor исправления исключаются до поиска: исходная
неисправленная версия того же сообщения не восстанавливается.
''')
    code("""airport=strata.groupby(['airport','scenario'])[['rows','information_changed']].sum()
airport['percent']=100*airport.information_changed/airport.rows
pivot=airport.percent.unstack('scenario')[['delay0','delay5','delay30','age30','age90','no_cor']]
display(pivot)
fig,ax=plt.subplots(figsize=(9,9))
im=ax.imshow(pivot.to_numpy(),aspect='auto',cmap='YlOrRd',vmin=0,vmax=100)
ax.set_xticks(range(len(pivot.columns)),pivot.columns)
ax.set_yticks(range(len(pivot.index)),pivot.index)
ax.set(title='Изменение информации относительно primary')
fig.colorbar(im,ax=ax,label='Доля событий аэропорта, %')
save('04_airport_information_sensitivity')
fields=pd.read_csv(R/'03_field_differences.csv')
totals=fields.groupby(['scenario','field'])[['rows','paired_present','missing_changed','value_changed','threshold_changed','absolute_difference_sum']].sum().reset_index()
totals['value_changed_percent_of_pairs']=100*totals.value_changed/totals.paired_present.replace(0,np.nan)
totals['missing_changed_percent_all']=100*totals.missing_changed/totals.rows
totals['mean_absolute_difference']=totals.absolute_difference_sum/totals.paired_present.replace(0,np.nan)
totals.loc[totals.field.isin(['present_weather_tokens','cumulonimbus_reported','towering_cumulus_reported','cavok']),'mean_absolute_difference']=np.nan
display(totals.loc[totals.scenario.isin(['delay0','delay30','no_cor'])])
selected=totals.loc[totals.scenario.eq('delay0')].set_index('field')
fig,ax=plt.subplots(figsize=(11,6))
selected[['value_changed_percent_of_pairs','missing_changed_percent_all']].plot.barh(ax=ax,color=[RED,GREEN])
ax.set(xlabel='Доля, %',ylabel='',title='Чувствительность полей к нулевой задержке')
ax.legend(['Изменение числа/категории среди заполненных пар','Смена пропуска среди всех событий'],fontsize=8)
save('05_field_value_and_missingness')
""")
    md(r'''## Устойчивость к временной неоднородности

Круговой блочный bootstrap календарных дней, парно для сценария и primary.
Все аэропорты одного дня переносятся совместно. Для каждого повторения b:
$$\Delta C_s^{(b)}=100\frac{\sum_{i\in b}(U_{is}-U_{i0})}{N_b}.$$
Интервалы - 2,5 и 97,5 процентили 2000 повторов, seed 20261011.
Это не ошибка измеренной доли полного корпуса и не интервал переносимости
на будущие годы. Длины блоков 7/14/28 дней проверяют зависимость результата
от принятой временной структуры.
''')
    code("""boot=pd.read_csv(R/'04_block_bootstrap.csv')
display(boot)
subset=boot.loc[boot.block_days.eq(14)&boot.GoA.eq(True)&boot.scenario.ne('primary')].reset_index(drop=True)
fig,ax=plt.subplots(figsize=(10,5))
ax.errorbar(subset.difference_pp,np.arange(len(subset)),xerr=[subset.difference_pp-subset.lower_pp,subset.upper_pp-subset.difference_pp],fmt='o',color=RED,capsize=4,label='GoA, блок 14 дней')
ax.set_yticks(range(len(subset)),subset.scenario)
ax.axvline(0,color='gray',linewidth=1)
ax.set(xlabel='Разность покрытия с primary, процентные пункты',title='Парная временная устойчивость покрытия GoA')
ax.legend()
save('06_block_bootstrap_goa')
""")
    md('''Изменение закодированной видимости или высоты не всегда равно
изменению физической величины: специальные коды задают границы. Пороговые
различия в CSV являются описательными: 5 узлов, 30 градусов, 1000 м,
500 ft, 2 °C и 2 гПа; это не критерии опасности или очистки.
Модельная польза погоды и причинное влияние на GoA здесь не измерены.
''')
    return nbf.v4.new_notebook(cells=cells)

def validation():
    cells,md,code=create()
    md('''# 03. Проверка токенов и границы валидации

До 12 сообщений каждой группы аэропорт × год плюс редкие облачные форматы,
остаточные сообщения и числовые аномалии. Детерминированная выборка;
обогащение редкими случаями не допускает чтения частоты замечаний как оценки
частоты ошибок в полном архиве.

Отдельный лексический алгоритм читает простые числовые токены, без импорта
рабочего декодера. Сравнение проверяет преобразование кодов, а не реальную
погоду. IEM также использует python-metar: согласие с рабочим строгим парсером
не было бы полностью независимой проверкой.
''')
    code(SETUP)
    code("""validation=json.loads((R/'05_validation_manifest.json').read_text(encoding='utf8'))
display(pd.DataFrame([validation]))
registry=pd.read_csv(R/'05_audit_sample_registry.csv')
display(registry.groupby('station')[['sample_regular','sample_enriched']].sum().reset_index())
audit=pd.read_csv(R/'05_lexical_validation.csv')
counts=audit.groupby(['field','agrees']).size().unstack(fill_value=0)
display(counts)
fig,ax=plt.subplots(figsize=(10,5))
counts.rename(columns={True:'Согласие с токеном',False:'Расхождение'}).plot.barh(stacked=True,ax=ax,color=[RED,GREEN] if False in counts.columns else [GREEN])
ax.set(xlabel='Число проверок',ylabel='',title='Проверка прямого чтения числовых токенов')
for position,field in enumerate(counts.index):
    errors=int(counts.loc[field,False]) if False in counts else 0
    ax.text(counts.loc[field].sum()+150,position,f'Расхождений: {errors}',va='center',fontsize=9)
ax.set_xlim(0,float(counts.sum(axis=1).max())*1.22)
ax.legend()
save('07_lexical_validation')
display(audit.loc[~audit.agrees].head(50))
adjudication=pd.read_csv(R/'05_disagreement_adjudication.csv')
assert (~adjudication.decoded_candidate & ~adjudication.core_validated).all()
display(adjudication)
fixtures=pd.read_csv(R/'05_semantic_fixtures.csv')
assert fixtures.agrees.all()
display(fixtures)
""")
    md('''Ожидания для восьми примеров заданы явно по руководствам и
консервативной политике неизвестных облачных компонент. Это авторские
ожидания, не разметка независимого метеоролога. Девять слешей проверяют
политику проекта, а не нормативность группы. Вертикальная видимость
сохраняется отдельно от нижней границы облаков.

Восемь расхождений относятся к четырем сообщениям с неподдержанной
структурой перед читаемым 9999. Все четыре основные части заблокированы;
исходные тексты не исправляются. Полный разбор приведен в таблице выше.

Не установлены: фактическая физическая правильность всех наблюдений,
время исторического получения и происхождение каждой числовой аномалии.
Полная внешняя ручная валидация остается ограничением публикации.
Источники и разграничение вклада: literature/01_positioning_and_sources.md.
''')
    return nbf.v4.new_notebook(cells=cells)

for name,nb in [('01_question_and_methods.ipynb',methods()),('02_availability_sensitivity.ipynb',sensitivity()),('03_decode_validation.ipynb',validation())]:
    nb.metadata={'kernelspec':{'name':'python3','display_name':'Python 3','language':'python'}}
    manager=KernelManager(kernel_name='python3')
    manager.kernel_spec.argv=[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}']
    client=NotebookClient(nb,km=manager,timeout=600,resources={'metadata':{'path':str(BASE/'notebooks')}},allow_errors=False)
    path=BASE/'notebooks'/name
    try:client.execute()
    finally:nbf.write(nb,path)
    nbf.validate(nb)
    assert all(c.execution_count is not None for c in nb.cells if c.cell_type=='code')
    outputs=[o for c in nb.cells if c.cell_type=='code' for o in c.outputs]
    assert not any(o.output_type=='error' for o in outputs)
    count=sum('image/png' in o.get('data',{}) for o in outputs)
    (R/f'{name[:2]}_notebook_verification.json').write_text(json.dumps(dict(notebook=name,executed=True,embedded_figures=count),indent=2)+'\n',encoding='utf8')
    print(name,count,flush=True)
