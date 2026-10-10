"""Общий аудит шести станций: основные поля, интервалы и задержка публикации."""
import copy
import json
import sys
import nbformat as nbf
from nbclient import NotebookClient
from jupyter_client.manager import KernelManager
from build_weather_notebooks import BASE, NOTEBOOKS, make
from build_quality_notebook import cells as template

def build():
    cells=copy.deepcopy(template)
    cells[0].source='''# 06. Качество и чувствительность погодного покрытия шести аэропортов

## 1. Область анализа и воспроизводимость

Объединены первый и расширенный пилоты: EGCC/EGLL/LFPG и EDDF/EHAM/LEMD,
полные январь и июль 2019 года по C40 UTC. Исследуются основные поля
сообщений с замечаниями, интервалы наблюдений и чувствительность к задержке.
Предполагаемые задержки 0/5/10/15 минут не подтверждают фактическую доступность.
Состав станций различается; сравнение этапов описательное.
[Протокол расширения](04_expansion_protocol.ipynb) ·
[Результаты расширения](05_expansion_coverage_and_decoding.ipynb).
'''
    cells[1].source+='''
expanded=json.loads((R/'08_pilot_manifest.json').read_text(encoding='utf8'))
assert expanded['input_sha256']==manifest['input_sha256']
for source in expanded['downloads']:
    assert sha(DATA/'raw'/f"{source['airport']}_{source['month']}.csv")==source['sha256']
total_rows=manifest['rows']+expanded['rows']
total_go_arounds=manifest['go_arounds']+expanded['go_arounds']
assert total_rows==171125 and total_go_arounds==508
'''
    cells[2].source=cells[2].source.replace(
        "reports = pd.read_parquet(DATA / '04_decoded_reports.parquet')",
        "reports = pd.concat([pd.read_parquet(DATA/'04_decoded_reports.parquet'),pd.read_parquet(DATA/'09_decoded_reports.parquet')],ignore_index=True)")
    cells[2].source=cells[2].source.replace(
        "events = pd.read_parquet(DATA / '03_pilot_matches.parquet')",
        "events = pd.concat([pd.read_parquet(DATA/'03_pilot_matches.parquet'),pd.read_parquet(DATA/'08_pilot_matches.parquet')],ignore_index=True)\nassert events.source_row.is_unique")
    cells[2].source=cells[2].source.replace(
        "DATA/'03_pilot_matches.parquet']}",
        "DATA/'03_pilot_matches.parquet',DATA/'08_pilot_matches.parquet',DATA/'09_decoded_reports.parquet',DATA/'09_pilot_features.parquet']}")
    cells[2].source=cells[2].source.replace('len(reports)==9223','len(reports)==18439').replace(
        '06_core_residual_audit.csv','10_core_residual_audit.csv')
    cells[2].source+='''
fields_table=pd.DataFrame({'Проверяемое поле':fields})
display(fields_table)
affected=events.merge(reports[['metar_id','station','metar_utc','metar','core_validated']].rename(columns={'station':'airport'}),
    on=['airport','metar_utc','metar'],how='left',validate='many_to_one')
assert len(affected)==total_rows
effect=affected.groupby('metar_id').agg(matched_rows=('source_row','size'),matched_go_arounds=('GoA','sum')).reset_index()
residual=residual.merge(effect,on='metar_id',how='left',validate='one_to_one')
residual[['matched_rows','matched_go_arounds']]=residual[['matched_rows','matched_go_arounds']].fillna(0).astype(int)
residual.to_csv(R/'10_core_residual_audit.csv',index=False)
display(residual[['station','metar_utc','core_validated','core_issues','matched_rows','matched_go_arounds']])
'''
    # Новый реестр хранит те же флаги stage04/stage09; основной префикс
    # остается отдельной диагностикой, а не исправленным исходным текстом.
    cells[3].source='''## 4. Повторный выбор METAR при предполагаемой задержке публикации

Выбирается последнее наблюдение с `observation_time + latency < C40`.
Давность считается от наблюдения. Последнее сообщение не пропускается из-за
замечаний качества или COR. После выбора проверяются предел давности и флаги.
При этом COR может смениться более старым сообщением без COR, поэтому покрытие
без COR не обязано монотонно уменьшаться при увеличении задержки.
'''
    cells[4].source=cells[4].source.replace('06_availability_scenarios.csv','10_availability_scenarios.csv')
    cells[4].source+='''
scenarios['stage']=np.where(scenarios.airport.isin(['EGCC','EGLL','LFPG']),'Первый пилот','Расширенный пилот')
scenarios.to_csv(R/'10_availability_scenarios.csv',index=False)
by_stage=scenarios.groupby(['stage','latency_minutes','maximum_age_minutes','quality','exclude_cor','GoA'])[['total','candidates']].sum().reset_index()
by_stage['coverage_percent']=100*by_stage.candidates/by_stage.total
by_stage.to_csv(R/'10_stage_comparison.csv',index=False)
display(by_stage.loc[by_stage.maximum_age_minutes.eq(60)&by_stage.quality.eq('core_validated')&by_stage.exclude_cor.eq(True)])
'''
    cells[5].source=cells[5].source.replace('08_latency_quality_coverage','18_combined_latency_quality')
    cells[5].source=cells[5].source.replace("axes[0].legend(fontsize=9)","axes[0].legend(fontsize=8,loc='upper left',bbox_to_anchor=(0,-.32),ncol=2)")
    cells[6].source=cells[6].source.replace('09_latency_airport_coverage','19_combined_latency_airports').replace(
        "for airport in sorted(events.airport.unique()):",
        "for airport,color in zip(sorted(events.airport.unique()),['#A51C30','#D65A4A','#E69A90','#005B50','#00897B','#80CBC4']):").replace(
        "marker='o',label=airport)","marker='o',label=airport,color=color)")
    cells[6].source=cells[6].source.replace("axes[0].legend()","axes[0].legend(loc='upper left',bbox_to_anchor=(0,-.32),ncol=3,fontsize=8)")
    cells[7].source=cells[7].source.replace('10_latency_age_limits','20_combined_latency_age_limits')
    cells[8].source='''## 5. Сравнение этапов и условия дальнейшего расширения

Основные поля оцениваются отдельно от полного текста: успешный разбор префикса
не исправляет группы состояния ВПП или сдвига ветра. Пропуски остаются пропусками.
Сравнение этапов не оценивает эффект аэропорта или сезонности.

Перед расширением на остальные станции требуется фиксированный перечень
станций и периодов, проверка идентичности станции, сохранение сырого ответа и
SHA256, контроль кратности и меток, а также полный учет неизвестных кодов,
неоднозначных временных отметок и пробелов покрытия. Новая неподдерживаемая
единица или код требует отдельной проверки до использования признака.
Порог 60 минут остается исследовательским; результаты 30/90 минут и задержек
представляются отдельно, без подбора порога по сохранению GoA.

Для обучения отдельно должны быть определены доступность аэропорта к C40,
обработка пропусков и временное разделение данных. Историческое время получения
METAR пока неизвестно. Массовая загрузка и обучение в этом ноутбуке не выполняются.
'''
    cells[9].source=cells[9].source.replace("manifest['rows']",'total_rows').replace("manifest['go_arounds']",'total_go_arounds')
    cells[9].source+='''
# Неподдержанная группа внутри основных полей должна блокировать префикс.
bad='EHAM 180455Z 17003KT INVALID CAVOK M02/M02 Q1017'
original=decode(bad,'EHAM',pd.Timestamp('2019-01-18T04:55Z').to_pydatetime())
assert not audit_core(bad,'EHAM',pd.Timestamp('2019-01-18T04:55Z'),original)[0]
good='EHAM 180455Z 17003KT CAVOK M02/M02 Q1017 06/1///95 NOSIG'
original=decode(good,'EHAM',pd.Timestamp('2019-01-18T04:55Z').to_pydatetime())
assert audit_core(good,'EHAM',pd.Timestamp('2019-01-18T04:55Z'),original)[0]
changed=dict(original,wind_speed_kt=999)
assert not audit_core(good,'EHAM',pd.Timestamp('2019-01-18T04:55Z'),changed)[0]
import unittest,test_decode_metar
outcome=unittest.TextTestRunner(stream=sys.stdout,verbosity=1).run(unittest.defaultTestLoader.loadTestsFromModule(test_decode_metar))
assert outcome.wasSuccessful()
assert sha(PROJECT/'europe/data/analytical/06_05_cases_excluded.parquet')==manifest['input_sha256']
overview={'rows':total_rows,'go_arounds':total_go_arounds,'reports':len(reports),
    'residual_reports':len(residual),'core_validated_residual_reports':int(residual.core_validated.sum()),
    'aged_expansion_rows':len(aged),'rows_deleted':0,'input_unchanged':True,
    'previous_files_unchanged':all(sha(p)==d for p,d in protected.items()),
    'availability_verified':False,'training_executed':False}
(R/'10_quality_manifest.json').write_text(json.dumps(overview,indent=2)+'\\n',encoding='utf8')
display(pd.DataFrame([overview]).T)
'''
    age_md=nbf.v4.new_markdown_cell('''## 3. Анализ записей LEMD с давностью METAR 60–90 минут

Для каждого C40 приводятся предыдущая и следующая временные отметки архива
и интервал между ними. Следующее сообщение используется только для диагностики
интервала и не участвует в погодных признаках на C40. Интервал между строками
архива не доказывает отсутствие передачи наблюдения метеостанцией.
''')
    age_code=nbf.v4.new_code_cell('''aged=events.loc[events.airport.isin(['EDDF','EHAM','LEMD'])&events.age_minutes.gt(60)&events.age_minutes.le(90)].copy()
assert len(aged)==14 and aged.airport.eq('LEMD').all() and not aged.GoA.any()
times=reports.loc[reports.station.eq('LEMD')&reports.archive_month.eq('2019-01'),'metar_utc'].drop_duplicates().sort_values()
following=[]
for stamp in aged.C40_CROSS_TIME:
    later=times.loc[times.gt(stamp)]
    following.append(later.iloc[0] if len(later) else pd.NaT)
aged['next_archive_observation_utc']=following
aged['archive_interval_minutes']=(aged.next_archive_observation_utc-aged.metar_utc).dt.total_seconds()/60
columns=['source_row','airport','month','GoA','C40_CROSS_TIME','metar_utc','age_minutes',
    'next_archive_observation_utc','archive_interval_minutes','status']
readable_table(aged[columns].sort_values('C40_CROSS_TIME'))
aged[columns].to_csv(R/'10_lemd_aged_records.csv',index=False)
intervals=aged.groupby(['metar_utc','next_archive_observation_utc','archive_interval_minutes']).agg(
    affected_rows=('source_row','size'),first_c40=('C40_CROSS_TIME','min'),last_c40=('C40_CROSS_TIME','max')).reset_index()
display(intervals)
fig,ax=plt.subplots(figsize=(9,4))
for i,row in enumerate(aged.sort_values('C40_CROSS_TIME').itertuples()):
    ax.plot([0,row.archive_interval_minutes],[i,i],color=LIGHT_GREEN,linewidth=2)
    ax.scatter(row.age_minutes,i,color=CORAL,zorder=3)
ax.scatter([],[],color=CORAL,label='C40 записи')
ax.plot([],[],color=LIGHT_GREEN,label='Интервал между строками архива')
ax.axvline(60,color=GREEN,linestyle='--',label='Основной предел давности')
ax.set(xlabel='Минуты после предыдущего архивного наблюдения',ylabel='Запись, порядок по C40',
    title='LEMD: положение C40 внутри интервалов наблюдений')
ax.legend(loc='upper left',bbox_to_anchor=(1.02,1),fontsize=9)
save('17_lemd_archive_intervals')
''')
    cells[3:3]=[age_md,age_code]
    cells[2:2]=[nbf.v4.new_markdown_cell('''## 2. Верификация основных полей сообщений с замечаниями

Для каждого сообщения отдельно проверяется префикс до единственной Q/A-группы
и согласованность основных погодных полей с сохраненным разбором. Все исходные
тексты и прежние флаги качества сохраняются. Флаг проверки префикса не означает
валидность полного сообщения или наличие каждого числового значения.
''')]
    return make(cells)

def main():
    notebook=build()
    path=NOTEBOOKS/'06_combined_quality_and_availability.ipynb'
    manager=KernelManager(kernel_name='python3')
    manager.kernel_spec.argv=[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}']
    client=NotebookClient(notebook,km=manager,timeout=300,resources={'metadata':{'path':str(NOTEBOOKS)}},allow_errors=False)
    client.on_cell_start=lambda cell,cell_index,**kwargs: print(f'cell {cell_index+1}/{len(notebook.cells)}',flush=True)
    try:client.execute()
    finally:nbf.write(notebook,path)
    nbf.validate(notebook)
    code_cells=[c for c in notebook.cells if c.cell_type=='code']
    outputs=[o for c in code_cells for o in c.outputs]
    assert all(c.execution_count is not None for c in code_cells)
    assert not any(o.output_type=='error' for o in outputs)
    verified=dict(notebook=path.name,executed=True,code_cells=len(code_cells),
        embedded_figures=sum('image/png' in o.get('data',{}) for o in outputs))
    assert verified['embedded_figures']==4
    (BASE/'reports/10_notebook_verification.json').write_text(json.dumps(verified,indent=2)+'\n',encoding='utf8')
    print(verified,flush=True)

if __name__=='__main__':main()
