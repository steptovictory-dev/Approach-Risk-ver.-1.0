"""Выполненный ноутбук аудита остаточных форматов METAR."""
import json
import sys
import nbformat as nbf
from nbclient import NotebookClient
from jupyter_client.manager import KernelManager
from build_weather_notebooks import BASE, NOTEBOOKS, SETUP, make

cells=[]
def md(text):cells.append(nbf.v4.new_markdown_cell(text))
def code(text):cells.append(nbf.v4.new_code_cell(text))
md('''# 11. Аудит остаточных форматов METAR

## 1. Область и способ проверки

Проверены все 2425 уникальных сообщений с замечаниями в полном погодном
наборе. Отбор задан прежним результатом декодера, а не исходом посадки.
Декодер и проверка основного префикса повторно воспроизведены для каждого
сообщения. Исходные тексты, значения и модельная числовая копия не изменяются.

Основная категория назначена по заранее указанному приоритету наблюдаемых
форматов: конвективный тип без количества/высоты, девять слешей, шесть слешей,
расхождение по явлениям, хвост после QNH, видимость, ветер, облачность, ошибка
строгого парсера, прочее. Это организация аудита, а не установленная причина
ошибки станции. Вторичные флаги пересекаются; основные категории — нет.
''')
code(SETUP)
code('''audit=json.loads((R/'19_residual_format_audit.json').read_text(encoding='utf8'))
cases=pd.read_csv(R/'19_residual_format_cases.csv',keep_default_na=False)
groups=pd.read_csv(R/'19_residual_format_groups.csv')
issues=pd.read_csv(R/'19_residual_issue_counts.csv')
unparsed=pd.read_csv(R/'19_residual_unparsed_groups.csv',keep_default_na=False)
assert cases.metar_id.is_unique and len(cases)==groups.messages.sum()==2425
assert audit['all_saved_issue_strings_reproduced'] and audit['all_saved_core_flags_reproduced']
assert sha(R/'12_full_residuals.csv')==audit['source_sha256']
assert sha(BASE/'scripts/decode_metar.py')==audit['decoder_sha256']
assert sha(BASE/'scripts/audit_residual_formats.py')==audit['audit_script_sha256']
display(pd.DataFrame([audit]))
readable_table(groups)
readable_table(issues)
''')
md('''## 2. Частота форматов и воздействие на аналитические строки

Число сообщений и число связанных посадок — разные единицы. Основной
сценарий: предшествующее сообщение до 60 минут, задержка 0 минут, COR сохранены.
Связи пересчитаны по полным файлам признаков. Более старые сообщения не
включены в столбцы `matched_60min_*`; исходные `linked_*` сохраняют все статусы.
''')
code('''names={
'convective_type_without_amount_height':'Тип CB/TCU без количества и высоты',
'nine_slash_cloud_placeholder':'Группа облачности /////////',
'six_slash_placeholder':'Группа //////',
'weather_parser_disagreement':'Расхождение по погодным явлениям',
'unclassified_tail_after_qnh':'Нераспознанный хвост после QNH',
'visibility_outside_current_scope':'Замечание к разбору видимости',
'wind_missing_or_multiple':'Замечание к группе ветра',
'other_cloud_parser_disagreement':'Другие расхождения по облакам',
'strict_parser_error':'Прочая ошибка строгого парсера',
'other_issue_requires_individual_review':'Другие замечания'}
ordered=groups.sort_values('messages')
fig,axes=plt.subplots(1,2,figsize=(15,7),sharey=True)
y=np.arange(len(ordered))
axes[0].barh(y,ordered.messages,color=CORAL,label='Сообщения с замечаниями')
axes[1].barh(y,ordered.matched_60min_rows,color=GREEN,label='Связанные строки, давность до 60 мин')
axes[0].set_yticks(y,ordered.primary_category.map(names))
axes[0].set(xlabel='Число уникальных сообщений',title='Остаточные форматы')
axes[1].set(xlabel='Число строк аналитической базы',title='Воздействие при основном сопоставлении')
for ax in axes:ax.legend(fontsize=8)
save('29_residual_formats_and_links')
readable_table(cases.groupby(['primary_category','core_validated']).agg(messages=('metar_id','size'),
    matched_60min_rows=('matched_60min_rows','sum'),matched_60min_go_arounds=('matched_60min_go_arounds','sum')).reset_index())
''')
md('''## 3. Облачные группы с отсутствующими компонентами

В [руководстве Météo-France 2021, печатная страница 17](https://meteofrance.fr/sites/meteofrance.fr/files/files/editorial/guide_aviation_2021.pdf)
группы `///CB` и `///TCU` обозначают обнаруженный тип конвективного облака при
неизвестных количестве и высоте. Это не отсутствие конвекции и не нулевая
высота. В нашем регулярном выражении группа требует отдельные количество
и высоту, поэтому короткая форма не распознается как слой: признаки типа
могут ошибочно оставаться False. Подтверждение кода не подтверждает измерение
в конкретном сообщении и его историческую доступность.

Группа `/////////` встречается в 472 остаточных сообщениях. Ручной разбор
сохраняет неизвестные количество, высоту и тип, но количество слоев отличается
от python-metar. В отличие от короткого CB/TCU, это расхождение представления
пропусков. Оно не разрешает превращать неизвестную облачность в ясное небо.
''')
code('''conv=cases.loc[cases.primary_category.eq('convective_type_without_amount_height')].copy()
conv['raw_tcu']=conv.observed_text.str.split().apply(lambda ts:'///TCU' in ts)
conv['raw_cb']=conv.observed_text.str.split().apply(lambda ts:'///CB' in ts)
display(pd.DataFrame([{
    'Сообщений с коротким CB/TCU':len(conv),
    'Короткая TCU есть, декодированный флаг False':int((conv.raw_tcu & ~conv.towering_cumulus_reported).sum()),
    'Короткая CB есть, декодированный флаг False':int((conv.raw_cb & ~conv.cumulonimbus_reported).sum())}]))
station=cases.groupby(['station','primary_category']).size().unstack(fill_value=0)
typed=station.get('convective_type_without_amount_height',pd.Series(0,index=station.index))
missing=station.get('nine_slash_cloud_placeholder',pd.Series(0,index=station.index))
other=station.sum(axis=1)-typed-missing
fig,ax=plt.subplots(figsize=(11,7))
ax.barh(station.index,typed,color=CORAL,label='Короткая группа CB/TCU')
ax.barh(station.index,missing,left=typed,color=GREEN,label='Группа /////////')
ax.barh(station.index,other,left=typed+missing,color='0.7',label='Остальные основные категории')
ax.set(xlabel='Число уникальных сообщений',title='Остаточные форматы по станциям')
ax.legend(fontsize=9)
save('30_residual_formats_by_station')
''')
md('''## 4. Остальные группы и исходные примеры

Ошибки строгого парсера группируются по нераспознанным фрагментам, а не
полным строкам ошибок с разным временем. Среди примеров встречаются группы
состояния ВПП, RE//, разделенные пробелами обозначения сдвига ветра,
нетипичные погодные сочетания и хвосты с похожими на TREND обозначениями.
Такие фрагменты требуют индивидуального правила; автоматическое исправление
порядка символов и добавление BECMG/TEMPO не выполнены.

Замечание к видимости не всегда означает неизвестную видимость: нетипичная
группа изменения ветра перед видимостью может нарушить позиционный разбор.
Дефект представления, неподдержанный формат и вероятное повреждение текста
нельзя объединять в одно утверждение об ошибке источника.
''')
code('''readable_table(unparsed.head(30))
for category in groups.primary_category:
    display(Markdown('### '+names[category]))
    part=cases.loc[cases.primary_category.eq(category)].sort_values(['station','metar_utc','metar_id'])
    readable_table(part.head(5)[['station','metar_utc','metar','decode_issues','unparsed_groups',
        'core_validated','matched_60min_rows','matched_60min_go_arounds']])
small=cases.loc[cases.primary_category.isin(['other_issue_requires_individual_review','other_cloud_parser_disagreement','six_slash_placeholder'])]
display(Markdown('### Все случаи трех малых категорий'))
readable_table(small[['station','metar_utc','metar','decode_issues']])
''')
md('''## 5. Повторное воспроизведение и сохранность данных

Декодирование и проверка основного префикса повторяются для всех остаточных
сообщений. Отдельно проверяются контрольные суммы погодных файлов и числовой
копии после CP-10. Флаг основного префикса не исправляет полный текст и не
подтверждает корректность остальных групп.
''')
code('''from decode_metar import decode
from core_audit import audit_core
for row in cases.itertuples():
    stamp=pd.Timestamp(row.metar_utc)
    decoded=decode(row.metar,row.station,stamp.to_pydatetime())
    assert decoded['metar_id']==row.metar_id
    assert decoded['decode_issues']==row.decode_issues and not decoded['decoded_candidate']
    validated,_,_=audit_core(row.metar,row.station,stamp,decoded)
    assert validated==row.core_validated
integrity=json.loads((R/'12_full_integrity.json').read_text(encoding='utf8'))
for item in integrity['feature_files']:
    assert sha(DATA/'full/results'/item['name'])==item['sha256']
numeric=json.loads((R/'18_numeric_mask_application.json').read_text(encoding='utf8'))
for item in numeric['files']:
    assert sha(DATA/'numeric_quality_v1'/item['name'])==item['output_sha256']
assert not audit['decoder_changed'] and not audit['weather_features_changed']
assert audit['rows_deleted']==0 and not audit['training_executed']
display(pd.DataFrame([{'Повторно проверено сообщений':len(cases),
    'Погодных файлов без изменения':93,'Файлов числовой копии без изменения':93}]))
''')
md('''## 6. Итог и следующий технический шаг

Проверены все остаточные сообщения, но семантические решения для всех форматов
не объявляются завершенными. Первый обоснованный шаг — поддержать короткие
CB/TCU с сохранением неизвестных количества и высоты, затем согласовать
сравнение облачных пропусков с python-metar. Изменение декодера должно быть
проверено на исходных пилотах и полном архиве, сохраняя числовые маски CP-10.

В этом ноутбуке исправления не применялись. Остаточные сообщения связаны с
6808 строками и 37 GoA в основном сценарии; для 2429 этих строк проходит
проверка основного префикса. Эти числа описывают покрытие, а не готовность
признаков к обучению. [Отчет аудита](../reports/19_residual_format_review.md).
''')
notebook=make(cells)
path=NOTEBOOKS/'11_residual_format_audit.ipynb'
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
(BASE/'reports/19_notebook_verification.json').write_text(json.dumps(dict(notebook=path.name,executed=True,embedded_figures=count),indent=2)+'\n',encoding='utf8')
print(path,flush=True)
