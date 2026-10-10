"""Выполненный отчет сравнения облачного декодирования cloud-v2."""
import json
import sys
import nbformat as nbf
from nbclient import NotebookClient
from jupyter_client.manager import KernelManager
from build_weather_notebooks import BASE, NOTEBOOKS, SETUP, make

cells=[]
def md(text):cells.append(nbf.v4.new_markdown_cell(text))
def code(text):cells.append(nbf.v4.new_code_cell(text))
md('''# 12. Проверка обновленного представления облачных групп METAR

## 1. Область изменения и сохраненные версии

Версия `cloud-v2` поддерживает `///CB` и `///TCU`, сохраняя известный тип и
неизвестные количество/высоту, и согласует сравнение полностью отсутствующей
облачной группы `/////////` с python-metar. Значение коротких групп описано в
[Météo-France, Guide aviation 2021, с. 17](https://meteofrance.fr/sites/meteofrance.fr/files/files/editorial/guide_aviation_2021.pdf).

Предыдущие модули и файлы остаются на месте, а новая версия хранится в
`data/full_cloud_v2/`: `reports/`, `features/`, `numeric_features/`.
Числовые маски CP-10 сохраняются. Наличие типа облака не подтверждает
его количественные характеристики или время получения METAR к C40.
''')
code(SETUP)
code('''application=json.loads((R/'21_cloud_v2_application.json').read_text(encoding='utf8'))
cases=pd.read_csv(R/'21_cloud_v2_cases.csv',keep_default_na=False)
changes=pd.read_csv(R/'21_cloud_v2_field_changes.csv',keep_default_na=False)
coverage=pd.read_csv(R/'21_cloud_v2_coverage.csv')
remaining=pd.read_csv(R/'21_cloud_v2_residuals.csv',keep_default_na=False)
assert application['all_source_rows_present_once'] and application['previous_files_unchanged']
assert application['numeric_masks_preserved'] and application['non_cloud_numeric_values_unchanged']
assert application['station_time_and_metar_links_unchanged']
assert application['rows']==coverage.rows.sum()==6331485 and application['go_arounds']==18349
assert cases.metar_id.is_unique and len(cases)==application['unique_target_messages']
assert len(remaining)==application['remaining_residual_messages']
assert application['rows_deleted']==0 and not application['training_executed']
display(pd.DataFrame([{key:value for key,value in application.items() if key!='jobs'}]))
readable_table(coverage)
''')
md('''## 2. Что изменилось в представлении

Для короткой группы количество записывается неизвестным, высота — пропуском,
тип CB/TCU сохраняется. При неизвестном количестве хотя бы одного слоя
потолок BKN/OVC не считается полностью установленным. Высота известного слоя
может сохраняться как `lowest_known_cloud_base_ft`; это сводка известных
высот, а не доказательство высоты самого низкого возможного облака.

Для девяти слешей исходный токен и неизвестные компоненты остаются в данных.
Исправляется сравнение с парсером, который не создает слой для полностью
отсутствующих характеристик. Неизвестная облачность не превращается в CAVOK.
''')
code('''field_counts=changes.groupby('field').metar_id.nunique().sort_values()
display(field_counts.rename('Измененных сообщений').to_frame())
fig,ax=plt.subplots(figsize=(11,7))
ax.barh(field_counts.index,field_counts.values,color=CORAL,label='Измененные поля сообщений')
ax.set(xlabel='Число уникальных сообщений',title='Изменения полей при обработке облачных групп')
ax.legend(fontsize=9)
save('31_cloud_v2_field_changes')
readable_table(cases.groupby(['old_decoded_candidate','new_decoded_candidate','old_core_validated','new_core_validated']).size().rename('messages').reset_index())
for token in ['///TCU','///CB','/////////']:
    display(Markdown('### Исходная группа '+token))
    examples=cases.loc[cases.metar.str.split().apply(lambda values:token in values)].sort_values(['station','metar_utc']).head(3)
    readable_table(examples)
    readable_table(changes.loc[changes.metar_id.isin(examples.metar_id)])
''')
md('''## 3. Изменение покрытия и остаточные замечания

Число сообщений и число посадок подсчитываются отдельно. Здесь используется
тот же основной сценарий связи: давность до 60 минут, задержка 0 минут,
COR сохранены. Улучшение синтаксического покрытия не подтверждает числовую
достоверность, полноту измерений или готовность набора к обучению.
''')
code('''old_audit=json.loads((R/'19_residual_format_audit.json').read_text(encoding='utf8'))
old_count=old_audit['messages']
new_count=len(remaining)
fig,ax=plt.subplots(figsize=(8,4))
ax.bar(['Предыдущая версия','cloud-v2'],[old_count,new_count],color=[CORAL,GREEN])
for x,value in enumerate([old_count,new_count]):ax.text(x,value+20,str(value),ha='center')
ax.set(ylabel='Число уникальных сообщений',title='Остаточные замечания полного декодирования')
from matplotlib.patches import Patch
ax.legend(handles=[Patch(color=CORAL,label='Предыдущая версия'),Patch(color=GREEN,label='Обновленная версия')])
save('32_cloud_v2_residual_comparison')
display(coverage[['old_decoded_rows','new_decoded_rows','old_core_rows','new_core_rows']].sum().to_frame('Строки до 60 минут'))
readable_table(remaining.groupby(['station','decode_issues']).agg(messages=('metar_id','size'),
    matched_60min_rows=('matched_60min_rows','sum'),matched_60min_go_arounds=('matched_60min_go_arounds','sum')).reset_index())
readable_table(remaining.sort_values(['station','metar_utc']).head(30))
''')
md('''## 4. Проверки совместимости и сохранности

Все 18 439 сообщений первоначальных пилотов проверены по всем ранее
сохраненным полям. Полный архив просмотрен для определения области изменения;
повторно декодированы только сообщения с целевыми группами в наблюдаемой
части до температуры/QNH. Для остальных сообщений модуль v2 вызывает прежний
декодер. При сохранении каждого файла проверены неизменность строк, связей и
полей вне облачной области, а также сохранность всех масок числового качества.

Ниже повторяются 26 семантических тестов и проверяются контрольные суммы
старых и новых файлов. Прогнозные группы и хвост после QNH не используются
для восстановления наблюдаемой облачности.
''')
code('''import unittest,test_decode_metar,test_full_weather_pipeline,test_cloud_v2
outcome=unittest.TextTestRunner(stream=sys.stdout,verbosity=1).run(unittest.TestSuite([
    unittest.defaultTestLoader.loadTestsFromModule(module)
    for module in [test_decode_metar,test_full_weather_pipeline,test_cloud_v2]]))
assert outcome.wasSuccessful() and outcome.testsRun==26
assert application['pilot_rows_unchanged']==18439
assert sha(BASE/'scripts/decode_metar_cloud_v2.py')==application['decoder_sha256']
assert sha(BASE/'scripts/decode_metar.py')==application['original_decoder_sha256']
assert sha(BASE/'scripts/rebuild_cloud_v2.py')==application['implementation_sha256']
for job in application['jobs']:
    for item in job['files']:
        assert sha(DATA/'full_cloud_v2'/item['path'])==item['sha256']
original=json.loads((R/'12_full_integrity.json').read_text(encoding='utf8'))
for item in original['feature_files']:assert sha(DATA/'full/results'/item['name'])==item['sha256']
numeric=json.loads((R/'18_numeric_mask_application.json').read_text(encoding='utf8'))
for item in numeric['files']:assert sha(DATA/'numeric_quality_v1'/item['name'])==item['output_sha256']
display(pd.DataFrame([{'Тестов':outcome.testsRun,'Сообщений пилотов без изменения':18439,
    'Строк базы сохранено':application['rows'],'Удалено строк':0,'Числовые маски сохранены':True}]))
''')
md('''## 5. Границы результата

Исправлено представление облачных групп, не сами наблюдения. Не выполнены
замена исходных строк, восстановление высоты/количества облаков, импутация,
обучение или оценка влияния на качество модели. Старые версии сохранены для
воспроизводимости аудита. Следующие форматы — погодные сочетания, группы
состояния ВПП, нетипичные хвосты — требуют собственных проверок.

После их разбора необходимо согласовать основной сценарий доступности погоды
на C40. Использовать новые погодные поля вместе с числовыми модельными копиями
из одной версии `full_cloud_v2`, соединяя по `source_row` и SHA-256 входа.
[Отчет версии](../reports/21_cloud_v2_review.md).
''')
notebook=make(cells)
path=NOTEBOOKS/'12_cloud_format_update.ipynb'
manager=KernelManager(kernel_name='python3')
manager.kernel_spec.argv=[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}']
client=NotebookClient(notebook,km=manager,timeout=300,resources={'metadata':{'path':str(NOTEBOOKS)}},allow_errors=False)
try:client.execute()
finally:nbf.write(notebook,path)
nbf.validate(notebook)
outputs=[o for c in notebook.cells if c.cell_type=='code' for o in c.outputs]
assert all(c.execution_count is not None for c in notebook.cells if c.cell_type=='code')
assert not any(o.output_type=='error' for o in outputs)
count=sum('image/png' in o.get('data',{}) for o in outputs)
assert count==2
(BASE/'reports/21_notebook_verification.json').write_text(json.dumps(dict(notebook=path.name,executed=True,embedded_figures=count),indent=2)+'\n',encoding='utf8')
print(path,flush=True)
