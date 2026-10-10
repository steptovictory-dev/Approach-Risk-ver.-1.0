"""Выполненный отчет о согласованной маскировке числовых полей."""
import json
import sys
import nbformat as nbf
from nbclient import NotebookClient
from jupyter_client.manager import KernelManager
from build_weather_notebooks import BASE, NOTEBOOKS, SETUP, make

cells=[]
def md(text):cells.append(nbf.v4.new_markdown_cell(text))
def code(text):cells.append(nbf.v4.new_code_cell(text))
md('''# 10. Применение согласованных правил числового качества

## 1. Решение и область применения

Пользователь принял маскировку отдельных сомнительных полей четырех экстремумов
и обеих компонент 14 несогласованных температурных пар. Все посадки, исходные
сообщения и декодированные числа сохраняются. Изменения относятся только к
полям с суффиксом `_model` в отдельной числовой копии признаков.

Эта копия не является готовым набором для обучения: временная доступность,
пропуски, COR и остаточные замечания декодера требуют отдельных решений.
Исходные файлы `data/full/results/` не изменяются. Обучение не выполняется.
''')
code(SETUP)
code('''decision=json.loads((R/'17_numeric_quality_protocol.json').read_text(encoding='utf8'))
application=json.loads((R/'18_numeric_mask_application.json').read_text(encoding='utf8'))
assert decision['status']=='accepted'
assert application['all_source_rows_present_once'] and application['raw_fields_unchanged']
assert application['source_files_unchanged'] and application['unmasked_model_values_unchanged']
assert application['rows_deleted']==0 and not application['training_executed']
assert application['input_sha256']==manifest['input_sha256']
display(pd.DataFrame([{key:value for key,value in application.items() if key!='files'}]))
registry=pd.read_csv(R/'17_numeric_quality_preview.csv')
readable_table(registry[['station','metar_utc','numeric_quality_reasons','proposed_mask_fields','linked_rows']])
''')
md('''## 2. Воздействие на строки и поля

Счетчик ячеек и счетчик посадок различаются: одна строка может иметь несколько
замаскированных полей. В журнал включены и выбранные поля, уже отсутствовавшие
в исходных числах; столбец `previously_present` показывает фактические числовые
значения, замененные пропуском. Красный цвет обозначает выбранные ячейки,
зеленый — среди них ранее заполненные. Категории перекрываются.
''')
code('''journal=pd.read_csv(R/'18_numeric_mask_journal.csv')
coverage=pd.read_csv(R/'18_numeric_mask_coverage.csv')
fields=pd.read_csv(R/'18_numeric_mask_fields.csv')
assert coverage.rows.sum()==application['rows']==6331485
assert coverage.loc[coverage.GoA.eq(True),'rows'].sum()==application['go_arounds']==18349
assert not journal.duplicated(['source_row','field']).any()
assert journal.model_value.isna().all()
assert journal.loc[journal.status.eq('matched')].source_row.nunique()==44
display(coverage.groupby('GoA')[['rows','flagged_rows','flagged_rows_matched_60min']].sum())
readable_table(fields)
fig,ax=plt.subplots(figsize=(11,5))
x=np.arange(len(fields))
ax.bar(x-.18,fields.masked_cells,.36,color=CORAL,label='Ячейки, выбранные для маскировки')
ax.bar(x+.18,fields.previously_present,.36,color=GREEN,label='Из них ранее заполненные')
ax.set_xticks(x,fields.field,rotation=25,ha='right')
ax.set(ylabel='Число ячеек',title='Маскировка отдельных числовых полей')
ax.legend()
save('28_numeric_mask_fields')
readable_table(journal)
''')
md('''## 3. Проверки сохраненных файлов

Скрипт применения повторно прочитал каждую сохраненную часть: все исходные
поля совпали, выбранные модельные поля стали пропусками, остальные модельные
значения совпали с исходными. Идентификаторы строк покрывают вход ровно один
раз, обе метки сохранены. Ниже повторно проверяются SHA-256 всех 93 исходных
файлов и всех 93 файлов копии относительно журнала применения.
''')
code('''assert len(application['files'])==93
for item in application['files']:
    assert sha(DATA/'full/results'/item['name'])==item['source_sha256']
    assert sha(DATA/'numeric_quality_v1'/item['name'])==item['output_sha256']
assert sha(BASE/'scripts/apply_numeric_quality.py')==application['implementation_sha256']
assert sha(R/'17_numeric_quality_preview.csv')==application['registry_sha256']
display(pd.DataFrame([{'Проверено исходных файлов':93,'Проверено файлов копии':93,
    'Удалено строк':0,'Исходные числовые значения сохранены':True}]))
sample=journal.loc[journal.status.eq('matched')].drop_duplicates('source_row').head(6)
parts=[]
for item in application['files']:
    if item['flagged_rows']==0:continue
    frame=pd.read_parquet(DATA/'numeric_quality_v1'/item['name'])
    part=frame.loc[frame.source_row.isin(sample.source_row)]
    if not part.empty:parts.append(part)
examples=pd.concat(parts,ignore_index=True)
readable_table(examples[['source_row','airport','metar','numeric_quality_reasons',
    'temperature_c','temperature_c_model','dewpoint_c','dewpoint_c_model',
    'qnh_hpa','qnh_hpa_model','cloud_ceiling_ft','cloud_ceiling_ft_model']])
''')
md('''## 4. Ограничения и продолжение

Причины пропуска по числовому качеству сохранены отдельно. Отсутствие флага
не означает физическую достоверность. Для температурных пар замаскированы обе
компоненты, поскольку источник ошибочного компонента не установлен.

Не выполнены импутация, расчет влажности и других зависимых признаков,
обучение, замена METAR более старым сообщением или восстановление по будущим
наблюдениям. Копия содержит все статусы сопоставления; доступность и давность
не обходятся числовой проверкой. Регистр четырех экстремумов — результат
ретроспективного аудита, а не правило обнаружения ошибок на новых данных.

Следующий этап — отдельный аудит остаточных форматов декодирования и затем
согласование основного сценария доступности погодных признаков на C40.
[Принятый протокол](../reports/17_numeric_quality_protocol.md).
''')
notebook=make(cells)
path=NOTEBOOKS/'10_numeric_quality_application.ipynb'
manager=KernelManager(kernel_name='python3')
manager.kernel_spec.argv=[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}']
client=NotebookClient(notebook,km=manager,timeout=300,resources={'metadata':{'path':str(NOTEBOOKS)}},allow_errors=False)
try:client.execute()
finally:nbf.write(notebook,path)
nbf.validate(notebook)
outputs=[o for c in notebook.cells if c.cell_type=='code' for o in c.outputs]
assert all(c.execution_count is not None for c in notebook.cells if c.cell_type=='code')
assert not any(o.output_type=='error' for o in outputs)
figures=sum('image/png' in o.get('data',{}) for o in outputs)
assert figures==1
(BASE/'reports/18_notebook_verification.json').write_text(json.dumps(dict(notebook=path.name,executed=True,embedded_figures=figures),indent=2)+'\n',encoding='utf8')
print(path,flush=True)
