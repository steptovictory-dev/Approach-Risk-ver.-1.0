"""Ограниченный пилот EDDF/EHAM/LEMD по зафиксированному протоколу."""
import json
import sys
import pandas as pd
import nbformat as nbf
from nbclient import NotebookClient
from jupyter_client.manager import KernelManager
from build_weather_notebooks import BASE, NOTEBOOKS, SETUP, make
from audit_pilot_input import INPUT, digest
from run_coverage_pilot import main as coverage
from decode_metar import decode

def main():
    protected={p:digest(p) for p in [INPUT,BASE/'data/03_pilot_matches.parquet',
        BASE/'data/04_decoded_reports.parquet',BASE/'data/04_pilot_features.parquet']}
    coverage(expansion=True)
    manifest=json.loads((BASE/'reports/08_pilot_manifest.json').read_text(encoding='utf8'))
    records=[]
    for source in manifest['downloads']:
        path=BASE/'data/raw'/f"{source['airport']}_{source['month']}.csv"
        assert digest(path)==source['sha256']
        frame=pd.read_csv(path,dtype=str,keep_default_na=False)
        for row in frame.itertuples(index=False):
            result=decode(row.metar,row.station,pd.Timestamp(row.valid,tz='UTC').to_pydatetime())
            result['archive_month']=source['month']
            records.append(result)
    reports=pd.DataFrame(records)
    # Исходные дубликаты сохраняются; одинаковые декодированные сообщения
    # сворачиваются только для присоединения признаков.
    reports['metar_utc']=pd.to_datetime(reports.metar_utc,utc=True).astype('datetime64[ns, UTC]')
    reports.to_parquet(BASE/'data/09_decoded_reports.parquet',index=False)
    matches=pd.read_parquet(BASE/'data/08_pilot_matches.parquet')
    right=reports.drop_duplicates('metar_id').rename(columns={'station':'airport'}).drop(columns='archive_month')
    keys=['airport','metar_utc','metar']
    assert not right.duplicated(keys).any()
    features=matches.merge(right,on=keys,how='left',validate='many_to_one',suffixes=('','_decoded'))
    assert len(features)==manifest['rows'] and features.source_row.is_unique
    assert int(features.GoA.sum())==manifest['go_arounds']
    features['weather_candidate_60min']=features.status.eq('matched')&features.decoded_candidate.eq(True)
    features['weather_candidate_60min_without_cor']=features.weather_candidate_60min&~features.correction_reported.eq(True)
    features.to_parquet(BASE/'data/09_pilot_features.parquet',index=False)
    residual=reports.loc[~reports.decoded_candidate].drop_duplicates('metar_id')
    affected=features.groupby('metar_id').agg(matched_rows=('source_row','size'),matched_go_arounds=('GoA','sum')).reset_index()
    residual=residual.merge(affected,on='metar_id',how='left',validate='one_to_one')
    residual[['matched_rows','matched_go_arounds']]=residual[['matched_rows','matched_go_arounds']].fillna(0).astype(int)
    residual.to_csv(BASE/'reports/09_decode_residuals.csv',index=False)
    summary=features.groupby(['airport','month','GoA']).agg(rows=('source_row','size'),
        candidates=('weather_candidate_60min','sum'),candidates_without_cor=('weather_candidate_60min_without_cor','sum')).reset_index()
    summary.to_csv(BASE/'reports/09_quality_summary.csv',index=False)
    for path,value in protected.items():
        assert digest(path)==value
    journal=dict(rows=len(features),go_arounds=int(features.GoA.sum()),raw_reports=len(reports),
        unique_reports=int(reports.metar_id.nunique()),reports_with_issues=len(residual),
        matched_residual_rows=int(residual.matched_rows.sum()),matched_residual_go_arounds=int(residual.matched_go_arounds.sum()),
        candidates=int(features.weather_candidate_60min.sum()),candidate_go_arounds=int(features.loc[features.weather_candidate_60min,'GoA'].sum()),
        candidates_without_cor=int(features.weather_candidate_60min_without_cor.sum()),
        candidate_go_arounds_without_cor=int(features.loc[features.weather_candidate_60min_without_cor,'GoA'].sum()),
        input_unchanged=True,previous_pilot_unchanged=True,rows_deleted=0,availability_verified=False,
        decoder_sha256=digest(BASE/'scripts/decode_metar.py'))
    (BASE/'reports/09_decode_manifest.json').write_text(json.dumps(journal,indent=2)+'\n',encoding='utf8')
    build_notebook()
    print(json.dumps(journal,indent=2),flush=True)

def build_notebook():
    cells=[]
    def md(s):cells.append(nbf.v4.new_markdown_cell(s))
    def code(s):cells.append(nbf.v4.new_code_cell(s))
    md('''# 05. Покрытие и декодирование METAR расширенного пилота

## 1. Дизайн и происхождение данных

EDDF/EHAM/LEMD, полные январь и июль 2019 года по C40 UTC.
Выбор станций и месяцев зафиксирован до загрузки в [протоколе](04_expansion_protocol.ipynb).
Сохраняются обе метки и все исходные строки. Число пригодных сообщений
характеризует техническое качество сопоставления, а не точность модели.
[Архив IEM](https://mesonet.agron.iastate.edu/info/datasets/metar.html).
''')
    code(SETUP)
    code('''expanded=json.loads((R/'08_pilot_manifest.json').read_text(encoding='utf8'))
journal=json.loads((R/'09_decode_manifest.json').read_text(encoding='utf8'))
features=pd.read_parquet(DATA/'09_pilot_features.parquet')
reports=pd.read_parquet(DATA/'09_decoded_reports.parquet')
assert len(features)==expanded['rows'] and features.source_row.is_unique
assert int(features.GoA.sum())==expanded['go_arounds']
for source in expanded['downloads']:
    assert sha(DATA/'raw'/f"{source['airport']}_{source['month']}.csv")==source['sha256']
display(pd.DataFrame(expanded['downloads'])[['airport','month','raw_reports','identical_duplicates','ambiguous_times','correction_reports']])
display(pd.DataFrame([journal]).T.rename(columns={0:'Значение'}))
''')
    md('''## 2. Верификация сопоставления на момент C40

Повторно выполняется выбор последнего наблюдения строго до C40.
Разные тексты на одной отметке не заменяются более старым сообщением.
Основной предел давности — 60 минут, дополнительные — 30 и 90 минут.
Время наблюдения не является подтвержденным временем доступности METAR.
''')
    code('''matches=pd.read_parquet(DATA/'08_pilot_matches.parquet')
for source in expanded['downloads']:
    airport,month=source['airport'],source['month']
    raw=pd.read_csv(DATA/'raw'/f'{airport}_{month}.csv',dtype=str,keep_default_na=False)
    assert raw.station.eq(airport).all()
    raw['metar_utc']=pd.to_datetime(raw.valid,utc=True).astype('datetime64[ns, UTC]')
    raw=raw.drop_duplicates(['metar_utc','metar'])
    raw['ambiguous']=raw.duplicated('metar_utc',keep=False)
    right=raw.drop_duplicates('metar_utc').sort_values('metar_utc')
    part=matches.loc[matches.airport.eq(airport)&matches.month.eq(month)].sort_values('C40_CROSS_TIME')
    repeated=pd.merge_asof(part[['source_row','C40_CROSS_TIME']],right[['metar_utc','metar','ambiguous']],
        left_on='C40_CROSS_TIME',right_on='metar_utc',direction='backward',allow_exact_matches=False,tolerance=pd.Timedelta(minutes=90))
    repeated.loc[repeated.ambiguous.eq(True),'metar']=None
    pd.testing.assert_frame_equal(repeated[['source_row','metar_utc','metar']].reset_index(drop=True),
        part[['source_row','metar_utc','metar']].reset_index(drop=True))
coverage=pd.read_csv(R/'08_coverage.csv')
display(coverage.loc[coverage.max_age_minutes.eq(60)])
fig,axes=plt.subplots(1,2,figsize=(10,4))
for ax,label in zip(axes,[False,True]):
    table=coverage.loc[coverage.GoA.eq(label)&coverage.max_age_minutes.eq(60)].pivot(index='airport',columns='month',values='coverage_percent')
    im=ax.imshow(table.values,vmin=0,vmax=100,cmap='YlGn',aspect='auto')
    for i in range(len(table)):
        for j in range(len(table.columns)):
            ax.text(j,i,f'{table.iloc[i,j]:.2f}%',ha='center',va='center',color='white' if table.iloc[i,j]>70 else 'black')
    ax.set(xticks=range(len(table.columns)),xticklabels=table.columns,yticks=range(len(table)),yticklabels=table.index,title='GoA' if label else 'Обычные посадки')
fig.suptitle('Доля записей с однозначным предшествующим METAR, давность ≤ 60 мин')
save('14_expansion_coverage')
''')
    md('''## 3. Верификация декодирования и форматов сообщения

Каждый сырой текст повторно декодируется, а результат сравнивается с сохраненной
таблицей по всем полям. Прогнозные группы отделяются от наблюдения; единицы и
коды пропусков сохраняются. Неизвестные токены и расхождения дают замечание,
а не автоматическое исправление. Применяется тот же декодер, что в первом пилоте.
''')
    code('''from decode_metar import decode
display(pd.DataFrame([{'Сохраненный SHA256 декодера':journal['decoder_sha256'],
    'Текущий SHA256 декодера':sha(BASE/'scripts/decode_metar.py')}]))
decoded=[]
for source in expanded['downloads']:
    raw=pd.read_csv(DATA/'raw'/f"{source['airport']}_{source['month']}.csv",dtype=str,keep_default_na=False)
    for row in raw.itertuples(index=False):
        value=decode(row.metar,row.station,pd.Timestamp(row.valid,tz='UTC').to_pydatetime())
        value['archive_month']=source['month']
        decoded.append(value)
repeated=pd.DataFrame(decoded)
repeated['metar_utc']=pd.to_datetime(repeated.metar_utc,utc=True).astype('datetime64[ns, UTC]')
pd.testing.assert_frame_equal(repeated,reports,check_dtype=False)
formats=reports.groupby('station').agg(reports=('metar_id','size'),valid_full_text=('decoded_candidate','sum'),
    trend=('has_trend','sum'),cor=('correction_reported','sum'),cavok=('cavok','sum'))
display(formats)
display(reports.groupby(['station','wind_unit','qnh_unit'],dropna=False).size().rename('reports').reset_index())
residual=pd.read_csv(R/'09_decode_residuals.csv',keep_default_na=False)
display(residual.groupby(['station','decode_issues']).agg(reports=('metar_id','size'),
    matched_rows=('matched_rows','sum'),GoA=('matched_go_arounds','sum')).reset_index())
''')
    md('''## 4. Сообщения с замечаниями и заполненность основных полей

Ниже приводятся все сообщения с замечаниями, их исходные тексты и число
связанных записей. Это диагностический реестр; наличие замечания не доказывает
ошибку источника. Пропуск порыва или потолка облачности не интерпретируется как ноль.
''')
    code('''readable_table(residual[['station','metar_utc','metar','decode_issues','matched_rows','matched_go_arounds']])
fields=['wind_speed_kt','wind_gust_kt','visibility_m','cloud_ceiling_ft','temperature_c','dewpoint_c','qnh_hpa']
present=features.groupby('airport')[fields].agg(lambda s:100*s.notna().mean())
display(present.round(2))
fig,ax=plt.subplots(figsize=(10,4))
present.T.plot.bar(ax=ax)
ax.set(ylabel='Доля записей с числовым значением, %',xlabel='Поле',ylim=(0,105),title='Заполненность погодных полей в расширенном пилоте')
ax.legend(title='Аэропорт')
ax.tick_params(axis='x',rotation=30)
ax.set_xticklabels(['Скорость ветра, уз','Порыв, уз','Видимость, м',
    'Потолок облачности, фут','Температура, °C','Точка росы, °C','QNH, гПа'],rotation=30,ha='right')
save('15_expansion_field_presence')
''')
    md('''## 5. Диагностические сценарии качества

Кандидат требует однозначного предшествующего METAR в пределах 60 минут
и полного декодирования без замечаний. Дополнительно рассматривается отсутствие
COR в выбранном сообщении. Эти флаги не являются принятым правилом исключения.
Сравнение долей между этапами описательное: состав станций и движения различается.
''')
    code('''summary=pd.read_csv(R/'09_quality_summary.csv')
display(summary)
fig,axes=plt.subplots(1,2,figsize=(11,4))
for ax,label in zip(axes,[False,True]):
    grouped=summary.loc[summary.GoA.eq(label)].groupby('airport')[['rows','candidates','candidates_without_cor']].sum()
    for i,field in enumerate(['candidates','candidates_without_cor']):
        bars=ax.bar(np.arange(len(grouped))+(i-.5)*.35,100*grouped[field]/grouped.rows,width=.35,
            color=[CORAL,GREEN][i],label=['Полный текст без замечаний','Полный текст без замечаний и COR'][i])
        ax.bar_label(bars,fmt='%.3f',fontsize=8,padding=3)
    ax.set(xticks=np.arange(len(grouped)),xticklabels=grouped.index,ylim=(99.5,100.08),ylabel='Доля записей, % (увеличенная шкала)',title='GoA' if label else 'Обычные посадки')
axes[0].legend(fontsize=8,loc='lower left')
save('16_expansion_quality_scenarios')
''')
    md('''## 6. Ограничения и дальнейшая проверка

Историческая доступность сообщения и целевого аэропорта к C40 не подтверждена.
Три новые станции и два месяца не обеспечивают репрезентативность всей базы.
Следующие проверки: классификация остаточных случаев, отдельная верификация
основных полей и чувствительность к задержке публикации по методу ноутбука 03.
Обучение модели и загрузка для всех аэропортов пока не выполнялись.
''')
    code('''import unittest, test_decode_metar
outcome=unittest.TextTestRunner(verbosity=1,stream=sys.stdout).run(unittest.defaultTestLoader.loadTestsFromModule(test_decode_metar))
assert outcome.wasSuccessful()
assert sha(PROJECT/'europe/data/analytical/06_05_cases_excluded.parquet')==expanded['input_sha256']
assert len(features)==94384 and int(features.GoA.sum())==201
''')
    notebook=make(cells)
    path=NOTEBOOKS/'05_expansion_coverage_and_decoding.ipynb'
    manager=KernelManager(kernel_name='python3')
    manager.kernel_spec.argv=[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}']
    client=NotebookClient(notebook,km=manager,timeout=300,resources={'metadata':{'path':str(NOTEBOOKS)}},allow_errors=False)
    client.on_cell_start=lambda cell,cell_index,**kwargs: print(f'cell {cell_index+1}/{len(cells)}',flush=True)
    try:client.execute()
    finally:nbf.write(notebook,path)
    nbf.validate(notebook)
    outputs=[o for c in notebook.cells if c.cell_type=='code' for o in c.outputs]
    assert not any(o.output_type=='error' for o in outputs)
    verification=dict(notebook=path.name,executed=True,embedded_figures=sum('image/png' in o.get('data',{}) for o in outputs))
    assert verification['embedded_figures']==3
    (BASE/'reports/09_notebook_verification.json').write_text(json.dumps(verification,indent=2)+'\n',encoding='utf8')

if __name__=='__main__':
    if sys.argv[1:]==['--notebook-only']:
        build_notebook()
    else:
        assert not sys.argv[1:], 'Неизвестные аргументы'
        main()
