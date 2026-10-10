"""Выполненный ноутбук отдельного аудита числовых аномалий."""
import json
import sys
import nbformat as nbf
from nbclient import NotebookClient
from jupyter_client.manager import KernelManager
from build_weather_notebooks import BASE, NOTEBOOKS, SETUP, make

cells=[]
def md(text): cells.append(nbf.v4.new_markdown_cell(text))
def code(text): cells.append(nbf.v4.new_code_cell(text))
md('''# 09. Отдельный аудит числовых аномалий METAR

## 1. Протокол и границы проверки

Аудит выполнен по согласованному решению до исправления значений или исключения
записей. Проверены все 93 блока станция–год. Кандидаты: четыре наблюдаемых
экстремума, QNH вне диагностического интервала 850–1100 гПа и превышение точки
росы над температурой. Этот интервал — средство поиска, а не норматив или
правило очистки. Проверка не является исчерпывающей физической валидацией.

Для четырех экстремумов показаны сообщения той же станции в пределах ±3 часов.
Последующие сообщения используются только для диагностики и не становятся
признаками на C40. Отбор не зависит от исхода посадки.
''')
code(SETUP)
code('''audit=json.loads((R/'15_numeric_audit.json').read_text(encoding='utf8'))
cases=pd.read_csv(R/'12_full_extreme_value_examples.csv')
candidates=pd.read_csv(R/'15_numeric_candidates.csv')
neighbors=pd.read_csv(R/'15_extreme_neighbors.csv')
checks=pd.read_csv(R/'15_source_archive_checks.csv')
assert audit['rows_changed']==audit['rows_deleted']==0
assert candidates.metar_id.nunique()==audit['unique_candidates']
display(pd.DataFrame([audit]))
readable_table(cases[['station','metar_utc','metar','review_reason','extreme_value','linked_rows','linked_go_arounds']])
readable_table(candidates)
''')
md('''## 2. Сопоставление с соседними наблюдениями

Красным обозначено проверяемое сообщение, зеленым — соседние наблюдения.
Линия соединяет дискретные сообщения и не означает интерполяцию измерений.
Единицы и шкалы различаются между панелями. Резкое отклонение требует проверки
происхождения; оно само по себе не устанавливает правильное значение.
''')
code('''neighbors['metar_utc']=pd.to_datetime(neighbors.metar_utc,utc=True)
fig,axes=plt.subplots(2,2,figsize=(13,8))
units={'qnh_hpa':'QNH, гПа','dewpoint_c':'Точка росы, °C','cloud_ceiling_ft':'Нижняя граница BKN/OVC, футы'}
for ax,case in zip(axes.flat,cases.itertuples()):
    field=case.review_reason.rsplit('_',1)[0]
    part=neighbors.loc[neighbors.case_id.eq(case.metar_id)].sort_values('metar_utc')
    ax.plot(part.offset_minutes,part[field],'-',color='0.65',linewidth=1)
    adjacent=part.loc[part.offset_minutes.ne(0)]
    ax.scatter(adjacent.offset_minutes,adjacent[field],color=GREEN,label='Соседние наблюдения',s=20)
    ax.scatter([0],[case.extreme_value],color=CORAL,s=65,zorder=3,label='Проверяемое сообщение')
    ax.set(title=f'{case.station}: {pd.Timestamp(case.metar_utc):%Y-%m-%d %H:%M} UTC',
           xlabel='Время относительно сообщения, мин',ylabel=units[field])
    ax.set_xlim(-190,190)
    if adjacent[field].notna().sum()==0:
        ax.text(.04,.07,'В соседних сообщениях BKN/OVC отсутствуют',transform=ax.transAxes,fontsize=9)
    ax.legend(fontsize=8)
save('27_numeric_extreme_neighbors')
for case in cases.itertuples():
    display(Markdown(f'### {case.station}, {case.metar_utc}'))
    readable_table(neighbors.loc[neighbors.case_id.eq(case.metar_id),['metar_utc','metar','qnh_hpa','temperature_c','dewpoint_c','cloud_ceiling_ft']])
other=candidates.loc[candidates.reason.eq('dewpoint_above_temperature')].drop_duplicates('metar_id')
for case in other.itertuples():
    display(Markdown(f'### Проверка согласованности температуры: {case.station}, {case.metar_utc}'))
    readable_table(neighbors.loc[neighbors.case_id.eq(case.metar_id),['metar_utc','metar','temperature_c','dewpoint_c']])
''')
md('''## 3. Проверка почасового архива источника

[Почасовой архив IEM](https://mesonet-longterm.agron.iastate.edu/archive/raw/sao/)
содержит сообщения, собранные IEM из потока National Weather Service. Для каждого
экстремума проверены час наблюдения и два следующих часа. Сжатые файлы сохранены
локально с URL, временем загрузки и SHA-256. Поиск проверяет точное совпадение
текста после нормализации пробелов; отсутствие совпадения в трех файлах не
доказывает отсутствие сообщения в источнике вообще. Поставщик тот же, поэтому
это не независимое подтверждение измерения и не доказательство ошибки станции.
''')
code('''import gzip
for row in checks.loc[checks.status.eq('retrieved')].itertuples():
    path=DATA/'anomaly_source_audit'/row.url.rsplit('/',1)[1]
    assert sha(path)==row.sha256
    raw=' '.join(gzip.decompress(path.read_bytes()).decode('utf8',errors='replace').split())
    original=cases.set_index('metar_id').loc[row.case_id,'metar']
    assert (' '.join(original.split()) in raw)==bool(row.exact_text_present)
readable_table(checks)
display(checks.groupby('case_id').agg(files_retrieved=('status',lambda x:int(x.eq('retrieved').sum())),
    files_with_exact_text=('exact_text_present','sum')))
''')
md('''## 4. Исследовательский вывод

Q0122, Q1202, 10/M80 и BKN900 присутствуют в исходных текстах, поэтому это
не результат подстановки чисел нашим декодером. Синтаксическая валидность не
означает достоверность величин. Соседние сообщения приведены для оценки
временной согласованности; исправление по ним не выполнялось.

Четыре сообщения связаны с семью обычными посадками и ни с одним GoA при
основном сопоставлении до 60 минут. Это не основание сохранять или удалять
аномалии в зависимости от метки. Числовая проверка должна учитываться отдельно
от флагов синтаксического декодирования. Независимый источник измерений и
согласованное правило обработки остаются необходимыми для исправления.
''')
code('''assert cases.linked_rows.sum()==7 and cases.linked_go_arounds.sum()==0
assert not audit['independent_source_verified']
assert sha(PROJECT/'europe/data/analytical/06_05_cases_excluded.parquet')==manifest['input_sha256']
''')
notebook=make(cells)
path=NOTEBOOKS/'09_numeric_anomaly_audit.ipynb'
manager=KernelManager(kernel_name='python3')
manager.kernel_spec.argv=[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}']
client=NotebookClient(notebook,km=manager,timeout=300,resources={'metadata':{'path':str(NOTEBOOKS)}},allow_errors=False)
try: client.execute()
finally: nbf.write(notebook,path)
nbf.validate(notebook)
assert all(c.execution_count is not None for c in notebook.cells if c.cell_type=='code')
assert not any(o.output_type=='error' for c in notebook.cells if c.cell_type=='code' for o in c.outputs)
(BASE/'reports/15_notebook_verification.json').write_text(json.dumps(dict(executed=True,notebook=path.name,embedded_figures=1),indent=2)+'\n',encoding='utf8')
print(path,flush=True)
