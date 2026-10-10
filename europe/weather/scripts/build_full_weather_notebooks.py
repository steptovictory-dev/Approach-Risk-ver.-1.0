"""Протокол полного охвата и выполненный итог обработки METAR."""
import json
import sys
import nbformat as nbf
from nbclient import NotebookClient
from jupyter_client.manager import KernelManager
from build_weather_notebooks import BASE, NOTEBOOKS, SETUP, make

def protocol_notebook():
    cells=[]
    def md(s):cells.append(nbf.v4.new_markdown_cell(s))
    def code(s):cells.append(nbf.v4.new_code_cell(s))
    md('''# 07. Протокол погодного покрытия актуальной аналитической базы

## 1. Область исследования

Включаются все непустые группы аэропорт–месяц текущей аналитической базы по
времени C40 UTC, без отбора по исходу или погодным характеристикам. Дизайн
зафиксирован до новой загрузки METAR. Сохраняются все строки и обе метки.
Покрытие относится к текущей аналитической выборке; оно не подтверждает
репрезентативность исходного авиационного движения. Обучение не входит в этап.
''')
    code(SETUP)
    code('''protocol=json.loads((R/'11_full_protocol.json').read_text(encoding='utf8'))
counts=pd.read_csv(R/'07_c40_month_counts.csv')
assert protocol['input_sha256']==manifest['input_sha256']
assert counts.rows.sum()==protocol['rows'] and counts.go_arounds.sum()==protocol['go_arounds']
assert len(counts)==protocol['groups']==968
assert counts.airport.nunique()==protocol['airports']==19
display(pd.DataFrame([{'Записи':protocol['rows'],'GoA':protocol['go_arounds'],
    'Аэропорты':protocol['airports'],'Группы аэропорт–месяц':protocol['groups'],
    'Запросы станция–год':len(protocol['jobs'])}]))
registry=counts.groupby('airport').agg(first_month=('month','min'),last_month=('month','max'),
    months=('month','size'),rows=('rows','sum'),go_arounds=('go_arounds','sum')).reset_index()
readable_table(registry)
''')
    md('''## 2. Временные границы и объем аналитических данных

Периоды определяются только присутствием записей в актуальном входе. Общий
интервал — январь 2019 — май 2023 года; границы аэропортов различаются.
EGCC заканчивается февралем 2021 года в данной аналитической базе. Поздние
записи исходного архива без C40 не добавляются в погодную выборку.
Пустые месяцы не требуют запросов ради искусственного выравнивания периодов.
''')
    code('''table=counts.pivot(index='airport',columns='month',values='rows').sort_index()
fig,ax=plt.subplots(figsize=(13,6))
im=ax.imshow(table.fillna(0).values/1000,cmap='YlGn',aspect='auto',vmin=0)
ticks=list(range(0,len(table.columns),6))
ax.set(yticks=range(len(table)),yticklabels=table.index,xticks=ticks,xticklabels=table.columns[ticks],
    xlabel='Месяц C40, UTC',title='Число записей аналитической базы по аэропорту и месяцу')
ax.tick_params(axis='x',rotation=45)
fig.colorbar(im,ax=ax,label='Число записей, тыс.')
save('21_full_input_monthly_registry')
''')
    md('''## 3. Метаданные и идентичность станции

Для каждого аэропорта проверено наличие единственного ICAO-идентификатора
в сети IEM. Сохранены исходные GeoJSON, URL, SHA256 и время загрузки. Названия
и координаты отражают текущий каталог; историческое положение станции и
непрерывность METAR не считаются подтвержденными. Это особенно существенно
для изменений инфраструктуры аэропорта в течение исследуемого периода.
Код станции и временная отметка внутри каждого сырого сообщения проверяются
при загрузке. Выбор другой станции при отсутствии сообщения не применяется.
''')
    code('''stations=[]
for source in protocol['station_metadata']:
    path=DATA/'station_metadata'/f"{source['network']}.geojson"
    assert sha(path)==source['sha256']
    matches=[f for f in json.loads(path.read_text(encoding='utf8'))['features'] if f['id']==source['airport']]
    assert len(matches)==1
    p=matches[0]['properties']
    stations.append({'airport':source['airport'],'network':source['network'],'name':p['sname'],
        'latitude':source['coordinates'][1],'longitude':source['coordinates'][0],
        'archive_begin':p.get('archive_begin'),'source_url':source['url']})
readable_table(pd.DataFrame(stations))
jobs=pd.DataFrame(protocol['jobs'])
requests_by_airport=jobs.groupby('airport').size().sort_values()
fig,ax=plt.subplots(figsize=(9,6))
ax.barh(requests_by_airport.index,requests_by_airport.values,color=CORAL)
ax.set(xlabel='Число запросов станция–год',title='Размер плана загрузки METAR',xticks=range(6))
save('22_full_request_plan')
display(jobs[['airport','year','start_utc','end_utc','rows','go_arounds']])
''')
    md('''## 4. Загрузка, сохранение прогресса и сопоставление

Запрос охватывает присутствующий период станции внутри года, включая
предшествующие сутки. Routine и special сохраняются в исходных байтах.
Годовой блок — техническая единица загрузки; отчеты строятся по месяцам C40.
Контрольные SHA256 связывают вход, сырой ответ и результат. Завершенный блок
не обрабатывается повторно при возобновлении без проверки версии кода и файлов.

Для предполагаемых задержек 0/5/10/15 минут выбирается последнее наблюдение
строго до C40 с учетом задержки. Затем проверяются давность 30/60/90 минут,
неоднозначность, полный разбор или основные поля и наличие COR. Замечание к
качеству не вызывает переход к более старой погоде. Основной предел — 60 минут.
Ни отсутствие погоды, ни плохой разбор не удаляют аналитическую запись.
''')
    code('''annual=counts.assign(year=counts.month.str[:4]).groupby('year')[['rows','go_arounds']].sum()
fig,axes=plt.subplots(1,2,figsize=(10,4))
axes[0].bar(annual.index,annual.rows/1000000,color=CORAL)
axes[0].set(ylabel='Число записей, млн',xlabel='Год C40, UTC',title='Объем аналитических данных')
axes[1].bar(annual.index,annual.go_arounds,color=GREEN)
axes[1].set(ylabel='Число GoA',xlabel='Год C40, UTC',title='Распределение метки GoA')
save('23_full_input_annual_counts')
display(annual)
''')
    md('''## 5. Правила аудита и ограничения интерпретации

Все неизвестные коды, пропуски, различные тексты на одной отметке и COR
сохраняются в диагностике. Проверка основных полей не исправляет полный текст.
Интервал между архивными строками не доказывает отсутствие передачи со станции.
При отличии годовой выгрузки от прежнего месячного снимка изменение учитывается
отдельно. Перекрытие предшествующих суток не должно удваивать аналитические строки.

Время наблюдения не является доказательством времени получения METAR.
Доступность целевого аэропорта к C40, обработка пропусков и временное разделение
данных должны быть определены до обучения. Исторические координаты станции и
изменения инфраструктуры требуют отдельной проверки при интерпретации результатов.
''')
    code('''import unittest,test_decode_metar,test_full_weather_pipeline
suite=unittest.TestSuite([unittest.defaultTestLoader.loadTestsFromModule(m) for m in [test_decode_metar,test_full_weather_pipeline]])
result=unittest.TextTestRunner(stream=sys.stdout,verbosity=1).run(suite)
assert result.wasSuccessful()
assert sha(PROJECT/'europe/data/analytical/06_05_cases_excluded.parquet')==protocol['input_sha256']
''')
    return make(cells)

def results_notebook():
    cells=[]
    def md(s):cells.append(nbf.v4.new_markdown_cell(s))
    def code(s):cells.append(nbf.v4.new_code_cell(s))
    md('''# 08. Погодное покрытие актуальной аналитической базы

## 1. Завершенность обработки и происхождение

Результаты полного технического охвата по [протоколу](07_full_coverage_protocol.ipynb).
Все исходные аналитические записи сохранены. Флаги качества являются
диагностическими сценариями и не означают принятого исключения записей.
Обучение модели не выполнялось.
''')
    code(SETUP)
    code('''protocol=json.loads((R/'11_full_protocol.json').read_text(encoding='utf8'))
progress=json.loads((R/'12_full_progress.json').read_text(encoding='utf8'))
integrity=json.loads((R/'12_full_integrity.json').read_text(encoding='utf8'))
assert integrity['all_source_rows_present_once'] and integrity['rows']==protocol['rows']
assert progress['complete'] and progress['completed']==progress['planned']==len(protocol['jobs'])
assert progress['rows_processed']==protocol['rows']
assert sum(j['go_arounds'] for j in progress['jobs'])==protocol['go_arounds']
for job in progress['jobs']:
    assert job['input_sha256']==protocol['input_sha256']
    path=DATA/'full/raw'/f"{job['airport']}_{job['year']}.csv"
    assert sha(path)==job['source']['sha256']
    for file in job['files']:
        assert sha(DATA/'full/results'/file['name'])==file['sha256']
display(pd.DataFrame(progress['jobs'])[['airport','year','rows','go_arounds','raw_reports','unique_reports','identical_duplicates','ambiguous_times','residual_reports','core_failed_reports']])
scenario=pd.read_csv(R/'12_full_scenarios.csv')
residual=pd.read_csv(R/'12_full_residuals.csv',keep_default_na=False)
summary=scenario.groupby(['latency_minutes','maximum_age_minutes','quality','exclude_cor','GoA'])[['total','candidates']].sum().reset_index()
summary['coverage_percent']=100*summary.candidates/summary.total
display(summary.loc[summary.maximum_age_minutes.eq(60)&summary.latency_minutes.eq(0)])
summary.to_csv(R/'12_full_summary.csv',index=False)
''')
    md('''## 2. Покрытие по аэропортам и месяцам

Покрытие наблюдением означает наличие однозначного предыдущего METAR в пределах
порога давности. Оно не требует полного декодирования и не подтверждает время
фактического получения сообщения. Различия периодов и состава аэропортов
ограничивают сравнение долей.
Белые ячейки обозначают отсутствие аналитических записей соответствующей
метки в месяце; они не означают установленное отсутствие METAR.
''')
    code('''base=scenario.loc[scenario.latency_minutes.eq(0)&scenario.maximum_age_minutes.eq(60)&scenario.quality.eq('observation')&scenario.exclude_cor.eq(False)]
fig,axes=plt.subplots(1,2,figsize=(15,7))
for ax,label in zip(axes,[False,True]):
    part=base.loc[base.GoA.eq(label)].copy()
    part['coverage']=100*part.candidates/part.total
    table=part.pivot(index='airport',columns='month',values='coverage').sort_index()
    im=ax.imshow(np.ma.masked_invalid(table.values),vmin=0,vmax=100,cmap='YlGn',aspect='auto')
    ticks=list(range(0,len(table.columns),12))
    ax.set(yticks=range(len(table)),yticklabels=table.index,xticks=ticks,xticklabels=table.columns[ticks],title='GoA' if label else 'Обычные посадки',xlabel='Месяц C40, UTC')
    ax.tick_params(axis='x',rotation=45)
    fig.colorbar(im,ax=ax,label='Покрытие METAR, %')
save('24_full_monthly_coverage')
airport=base.groupby(['airport','GoA'])[['total','candidates']].sum().reset_index()
airport['coverage_percent']=100*airport.candidates/airport.total
display(airport)
monthly=base.assign(coverage_percent=100*base.candidates/base.total)
display(monthly.sort_values(['coverage_percent','airport','month']).head(20)[
    ['airport','month','GoA','total','candidates','coverage_percent']])
display(pd.read_csv(R/'12_full_age_and_missingness.csv').groupby('airport')[
    ['total','no_previous','older_than_60','older_than_90','ambiguous']].sum())
''')
    md('''## 3. Чувствительность к давности и предполагаемой задержке

Повторный выбор при задержке оценивает предположение о доступности, а не
фактическое время публикации. Пропуски и COR не заменяются более старой погодой
ради улучшения качества. Основные поля и полный разбор представлены отдельно.
''')
    code('''fig,axes=plt.subplots(1,2,figsize=(11,4))
for ax,label in zip(axes,[False,True]):
    for age,color in [(30,CORAL),(60,GREEN),(90,LIGHT_CORAL)]:
        part=summary.loc[summary.GoA.eq(label)&summary.maximum_age_minutes.eq(age)&summary.quality.eq('core_validated')&summary.exclude_cor.eq(True)].sort_values('latency_minutes')
        ax.plot(part.latency_minutes,part.coverage_percent,marker='o',color=color,label=f'Давность ≤ {age} мин')
    ax.set(title='GoA' if label else 'Обычные посадки',xlabel='Предполагаемая задержка, мин',ylabel='Основные поля без COR, %',xticks=[0,5,10,15])
axes[0].legend(loc='lower left')
save('25_full_latency_age_sensitivity')
''')
    md('''## 4. Замечания декодирования и исходные сообщения

Полный реестр сохранен в [CSV](../reports/12_full_residuals.csv). Здесь показаны
числа по станциям и группам замечаний, а также первые 30 сообщений по станции
и времени без отбора по исходу. Положительная проверка префикса до QNH не
исправляет исходный текст и не подтверждает валидность дополнительных групп.
''')
    code('''display(residual.groupby(['station','decode_issues']).agg(reports=('metar_id','size'),
    matched_rows=('matched_rows','sum'),matched_go_arounds=('matched_go_arounds','sum')).reset_index())
readable_table(residual.sort_values(['station','metar_utc']).head(30)[['station','metar_utc','metar','decode_issues','core_issues','matched_rows','matched_go_arounds']])
groups=residual.groupby('station').size().sort_values()
fig,ax=plt.subplots(figsize=(9,6))
ax.barh(groups.index,groups.values,color=CORAL)
ax.set(xlabel='Число уникальных сообщений с замечаниями',title='Остаточные замечания декодирования по станциям')
save('26_full_decode_residuals')
comparisons=[dict(airport=j['airport'],year=j['year'],**c) for j in progress['jobs'] for c in j['pilot_snapshot_comparison']]
display(pd.DataFrame(comparisons))
completeness=pd.read_csv(R/'12_full_field_completeness.csv')
filled=completeness.groupby(['airport','field'])[['total','present']].sum().reset_index()
filled['present_percent']=100*filled.present/filled.total
display(filled.pivot(index='airport',columns='field',values='present_percent').round(2))
display(completeness.groupby('field').agg(minimum=('minimum','min'),maximum=('maximum','max')))
readable_table(pd.read_csv(R/'12_full_extreme_value_examples.csv'))
formats=pd.read_csv(R/'12_full_formats.csv',keep_default_na=False)
display(formats.groupby(['wind_unit','qnh_unit','visibility_unit'])[['reports','auto_reports','cor_reports']].sum().reset_index())
''')
    md('''## 5. Итоговые границы применимости

Завершение загрузки не означает готовность модели к обучению. Историческая
доступность METAR и целевого аэропорта к C40 не подтверждена. Пропуски,
замечания качества и изменение состава движения требуют отдельных решений.
Новые исключения и автоматическое исправление источника не применялись.
Числовая достоверность не подтверждается синтаксической проверкой: QNH 122 и
1202 гПа, точка росы −80 °C и BKN900 рассмотрены отдельно в
[ноутбуке 09](09_numeric_anomaly_audit.ipynb). Значения сохранены без исправления.
Остаточные форматы и правило использования погодных признаков требуют отдельного решения.

Аэропорт назначения принят заранее известным к C40 как исследовательское
допущение; фактический внешний канал должен быть подтвержден для применения.
[Допущения доступности](../reports/13_feature_availability_assumptions.md).
''')
    code('''import unittest,test_decode_metar,test_full_weather_pipeline
outcome=unittest.TextTestRunner(stream=sys.stdout,verbosity=1).run(unittest.TestSuite([
    unittest.defaultTestLoader.loadTestsFromModule(m) for m in [test_decode_metar,test_full_weather_pipeline]]))
assert outcome.wasSuccessful()
assert sha(PROJECT/'europe/data/analytical/06_05_cases_excluded.parquet')==protocol['input_sha256']
''')
    return make(cells)

def main():
    mode=sys.argv[1] if len(sys.argv)>1 else 'protocol'
    assert mode in ['protocol','results']
    notebook=protocol_notebook() if mode=='protocol' else results_notebook()
    name='07_full_coverage_protocol.ipynb' if mode=='protocol' else '08_full_weather_coverage.ipynb'
    path=NOTEBOOKS/name
    manager=KernelManager(kernel_name='python3')
    manager.kernel_spec.argv=[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}']
    client=NotebookClient(notebook,km=manager,timeout=300,resources={'metadata':{'path':str(NOTEBOOKS)}},allow_errors=False)
    client.on_cell_start=lambda cell,cell_index,**kwargs:print(f'{name} cell {cell_index+1}/{len(notebook.cells)}',flush=True)
    try:client.execute()
    finally:nbf.write(notebook,path)
    nbf.validate(notebook)
    code_cells=[c for c in notebook.cells if c.cell_type=='code']
    assert all(c.execution_count is not None for c in code_cells)
    outputs=[o for c in code_cells for o in c.outputs]
    assert not any(o.output_type=='error' for o in outputs)
    verified=dict(notebook=name,executed=True,embedded_figures=sum('image/png' in o.get('data',{}) for o in outputs))
    assert verified['embedded_figures']==3
    (BASE/'reports'/('11_notebook_verification.json' if mode=='protocol' else '12_notebook_verification.json')).write_text(json.dumps(verified,indent=2)+'\n',encoding='utf8')
    print(verified,flush=True)

if __name__=='__main__':main()
