"""Выполненные ноутбуки остаточных решений и итоговой таблицы C40."""
import json
import sys
import nbformat as nbf
from nbclient import NotebookClient
from jupyter_client.manager import KernelManager
from build_weather_notebooks import BASE, NOTEBOOKS, SETUP, make

def residual_notebook():
    cells=[]
    def md(text):cells.append(nbf.v4.new_markdown_cell(text))
    def code(text):cells.append(nbf.v4.new_code_cell(text))
    md('''# 13. Правила использования остаточных форматов METAR

Все 1102 остаточных сообщения после cloud-v2 получают явное решение об
использовании. Это завершение правил обработки, а не утверждение, что все
исходные тексты удалось полностью расшифровать. Поврежденные, неоднозначные
и неподдержанные группы остаются сохраненными без догадок и исправления.

Если проверен основной префикс, допускается его наблюдаемая погода; хвост
после QNH не используется для реконструкции признаков. Если основной блок
не проверен, его модельные погодные поля остаются неизвестными. Посадка
сохраняется. Предположение о задержке, числовые маски и пропуски действуют
дополнительно к этому правилу.
''')
    code(SETUP)
    code('''cases=pd.read_csv(R/'22_residual_decision_register.csv',keep_default_na=False)
groups=pd.read_csv(R/'22_residual_decision_groups.csv')
assert cases.metar_id.is_unique and len(cases)==groups.messages.sum()==1102
assert cases.source_action.eq('preserve_without_repair').all()
assert cases.loc[cases.core_validated,'model_action'].eq('use_validated_core_only').all()
assert cases.loc[~cases.core_validated,'model_action'].eq('model_weather_unknown').all()
readable_table(groups)
readable_table(cases.groupby(['decode_issues','model_action']).size().rename('messages').reset_index())
fig,ax=plt.subplots(figsize=(10,5))
count=cases.groupby('model_action').size()
labels={'use_validated_core_only':'Использовать проверенный префикс',
    'model_weather_unknown':'Оставить погодные поля неизвестными'}
ax.barh([labels[x] for x in count.index],count.values,color=[CORAL,GREEN])
ax.set(xlabel='Число уникальных сообщений',title='Решения по остаточным форматам')
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color=CORAL,label='Непроверенный основной блок'),
    Patch(color=GREEN,label='Проверенный основной префикс')],fontsize=9)
save('33_residual_decisions')
for category in groups.decision_category:
    display(Markdown('### '+category))
    readable_table(cases.loc[cases.decision_category.eq(category)].sort_values(['station','metar_utc']).head(5)[
        ['station','metar_utc','metar','decode_issues','core_validated','model_action']])
''')
    md('''Нераспознанные погодные сочетания вроде FZBCFG/BCFZFG не преобразуются
в новые составные признаки без отдельной семантической проверки. Группы ВПП,
RE// и нетипичные обозначения тренда не исправляются автоматически. При
непроверенном префиксе консервативно блокируется весь модельный погодный блок;
это исследовательское ограничение использования, не доказательство ошибки
каждого числового поля. COR сохраняется как флаг исправленного сообщения.
[Протокол](../reports/22_final_weather_protocol.md).
''')
    return make(cells)

def final_notebook():
    cells=[]
    def md(text):cells.append(nbf.v4.new_markdown_cell(text))
    def code(text):cells.append(nbf.v4.new_code_cell(text))
    md('''# 14. Итоговая таблица погодных признаков на C40

Основной сценарий согласован пользователем: давность METAR до 60 минут,
предполагаемая задержка 15 минут, COR сохранены с флагом, без импутации.
Выбирается последнее сообщение со строго предшествующим условным временем
доступности. Выбор не зависит от качества или исхода, более старое «хорошее»
сообщение не подставляется. Фактическое время получения исторических METAR,
включая исправления COR, не подтверждено.

Все строки посадок и исходные числа сохраняются. Модельные копии `_model`
ограничены временной доступностью, проверкой основного текста и принятыми
числовыми масками. Наличие таблицы не означает завершенное обучение или
готовый эксплуатационный прогноз.
''')
    code(SETUP)
    code('''protocol=json.loads((R/'22_final_weather_protocol.json').read_text(encoding='utf8'))
application=json.loads((R/'23_final_weather_application.json').read_text(encoding='utf8'))
coverage=pd.read_csv(R/'23_final_weather_coverage.csv')
fields=pd.read_csv(R/'23_final_weather_fields.csv')
assert protocol['assumed_latency_minutes']==application['assumed_latency_minutes']==15
assert application['rows']==coverage.rows.sum()==6331485
assert coverage.loc[coverage.GoA.eq(True),'rows'].sum()==application['go_arounds']==18349
assert application['all_source_rows_present_once'] and application['events_and_labels_unchanged']
assert application['source_reports_unchanged'] and application['numeric_mask_registry_preserved']
assert application['strict_assumed_availability_before_c40']
assert not application['training_executed'] and application['rows_deleted']==0
display(pd.DataFrame([{k:v for k,v in application.items() if k!='files'}]))
readable_table(coverage.groupby('GoA')[['rows','observation_usable','weather_core_usable','cor_usable','numeric_flagged_rows']].sum().reset_index())
filled=fields.groupby('field')[['rows','present_model']].sum()
filled['present_percent']=100*filled.present_model/filled.rows
readable_table(filled.reset_index())
fig,ax=plt.subplots(figsize=(11,8))
ordered=filled.present_percent.sort_values()
ax.barh(ordered.index,ordered.values,color=GREEN,label='Заполненные модельные значения')
ax.set(xlabel='Доля строк, %',xlim=(0,100),title='Заполненность погодных модельных полей')
ax.legend(fontsize=9)
save('34_final_weather_completeness')
''')
    md('''## Независимая проверка времени выбора

Для всех файлов последний допустимый отчет повторно определяется через
поиск позиции в массиве времен наблюдения (numpy.searchsorted), независимо
от использованного при сборке merge_asof. Проверяются источники, SHA-256,
кратность строк, ограничения доступности и заполненность вне допустимого
основного блока. Неоднозначные времена проверены отдельно.
''')
    code('''seen=np.zeros(application['rows'],dtype=bool)
labels=0
examples=[]
for item in application['files']:
    path=DATA/'final_c40_weather_v1'/item['name']
    assert sha(path)==item['sha256']
    frame=pd.read_parquet(path)
    key=item['name'].removesuffix('.parquet')
    reports=pd.read_parquet(DATA/'full_cloud_v2/reports'/f'{key}_reports.parquet')
    assert sha(DATA/'full_cloud_v2/reports'/f'{key}_reports.parquet')==item['source_reports_sha256']
    assert not reports.duplicated('metar_utc',keep=False).any()
    ordered=reports.sort_values('metar_utc')
    available_ns=(ordered.metar_utc+pd.Timedelta(minutes=15)).astype('int64').to_numpy()
    positions=np.searchsorted(available_ns,frame.C40_CROSS_TIME.astype('int64').to_numpy(),side='left')-1
    has=positions>=0
    assert frame.metar_utc.notna().to_numpy().tolist()==has.tolist()
    expected=ordered.metar_id.to_numpy()[positions[has]]
    assert np.array_equal(frame.loc[has,'metar_id'].to_numpy(),expected)
    ids=frame.source_row.to_numpy()
    assert frame.source_row.is_unique and not seen[ids].any()
    seen[ids]=True
    labels+=int(frame.GoA.sum())
    assert frame.loc[has,'assumed_available_utc'].lt(frame.loc[has,'C40_CROSS_TIME']).all()
    assert frame.loc[frame.weather_core_usable,'age_minutes'].between(15,60,inclusive='right').all()
    model_fields=[f for f in frame if f.endswith('_model')]
    assert frame.loc[~frame.weather_core_usable,model_fields].isna().all().all()
    for name in model_fields:
        source=name.removesuffix('_model')
        mask=frame.numeric_mask_fields.str.split(';').apply(lambda values:source in values)
        assert frame.loc[mask,name].isna().all()
    if len(examples)<5:
        examples.append(frame[['source_row','airport','GoA','C40_CROSS_TIME','metar_utc','assumed_available_utc',
            'age_minutes','correction_reported','weather_core_usable','weather_missing_reason','qnh_hpa','qnh_hpa_model']].head(2))
assert seen.all() and labels==18349
assert sha(PROJECT/'europe/data/analytical/06_05_cases_excluded.parquet')==application['input_sha256']
readable_table(pd.concat(examples,ignore_index=True))
display(pd.DataFrame([{'Независимо проверено файлов':len(application['files']),
    'Строк сохранено':len(seen),'GoA сохранено':labels,'Выбор последнего допустимого METAR проверен':True}]))
''')
    md('''## Пропуски, флаги и границы применения

Причины пропуска различаются: нет предшествующего METAR, сообщение слишком
старое, неоднозначное время, непроверенный основной текст, числовая маска,
отсутствующая группа или неизвестная характеристика в самом сообщении.
COR означает исправленную версию; True у correction_reported не доказывает
фактическую доступность исправления на C40. Правила объяснены в README.

В модельных полях остаются отношения ге/меньше для специальных кодов видимости,
отдельная вертикальная видимость и статусы пропусков. Ветер относительно ВПП
не рассчитывается: выбор ВПП к C40 не установлен. Аэропорт назначения заранее
известен по исследовательскому допущению пользователя, не по доказанному
свойству ретроспективного поля airport.

Метка GoA и служебные поля происхождения не входят в список погодных
предикторов. Импутация, статистические преобразования, подбор признаков и
оценка модели выполняются только по отдельному протоколу моделирования.
[Итоговый отчет](../reports/23_final_weather_review.md).
''')
    md('''## Временное разделение

Согласовано разделение по C40 в UTC: train до 2022 года, validation за 2022
год, test за 2023 год. Карта хранится отдельно в data/temporal_split_v1/.
Поле temporal_split в погодном снимке содержит прежний служебный заполнитель;
перед присоединением карты его необходимо удалить. Присоединение выполняется
по source_row с проверкой one_to_one и неизменного хеша входной базы.
Test уже использовался в предыдущих исследованиях и не является новой
независимой подтверждающей оценкой. Обучение не выполнялось.
''')
    code('''split_application=json.loads((R/'24_temporal_split_application.json').read_text(encoding='utf8'))
split_counts=pd.read_csv(R/'24_temporal_split_counts.csv')
split_seen=np.zeros(application['rows'],dtype=bool)
for item in split_application['files']:
    path=DATA/'temporal_split_v1'/item['name']
    assert sha(path)==item['sha256']
    mapping=pd.read_parquet(path)
    years=mapping.C40_CROSS_TIME.dt.year
    expected=np.select([years.lt(2022),years.eq(2022)],['train','validation'],default='test')
    assert np.array_equal(mapping.temporal_split.to_numpy(),expected)
    assert mapping.source_row.is_unique and not split_seen[mapping.source_row].any()
    split_seen[mapping.source_row]=True
assert split_seen.all() and split_counts.rows.sum()==application['rows']
readable_table(split_counts.groupby('temporal_split').rows.sum().reset_index())
readable_table(split_counts.groupby(['temporal_split','airport']).rows.sum().reset_index())
''')
    return make(cells)

for name,notebook,count in [('13_residual_usage_protocol.ipynb',residual_notebook(),1),
                            ('14_final_c40_weather_features.ipynb',final_notebook(),1)]:
    manager=KernelManager(kernel_name='python3')
    manager.kernel_spec.argv=[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}']
    client=NotebookClient(notebook,km=manager,timeout=600,resources={'metadata':{'path':str(NOTEBOOKS)}},allow_errors=False)
    path=NOTEBOOKS/name
    try:client.execute()
    finally:nbf.write(notebook,path)
    nbf.validate(notebook)
    outputs=[o for c in notebook.cells if c.cell_type=='code' for o in c.outputs]
    assert all(c.execution_count is not None for c in notebook.cells if c.cell_type=='code')
    assert not any(o.output_type=='error' for o in outputs)
    assert sum('image/png' in o.get('data',{}) for o in outputs)==count
    (BASE/'reports'/f'{name[:2]}_final_notebook_verification.json').write_text(json.dumps(dict(notebook=name,executed=True,embedded_figures=count),indent=2)+'\n',encoding='utf8')
    print(path,flush=True)
