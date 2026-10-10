"""Фиксация ограниченного расширения пилота до загрузки новых METAR."""
from datetime import datetime, timezone
import json
import sys
import requests
import pandas as pd
import pyarrow.parquet as pq
import nbformat as nbf
from nbclient import NotebookClient
from jupyter_client.manager import KernelManager
from build_weather_notebooks import BASE, NOTEBOOKS, SETUP, make
from audit_pilot_input import INPUT, digest

def main():
    audit=json.loads((BASE/'reports/01_input_audit.json').read_text(encoding='utf8'))
    assert digest(INPUT)==audit['input_sha256']
    airports=pd.read_csv(BASE/'reports/01_airport_coverage.csv')
    ranked=airports.sort_values(['c40_early','airport'],ascending=[False,True])
    selected=ranked.loc[~ranked.airport.isin(audit['selected_airports'])].head(3).airport.tolist()
    assert selected==['EDDF','EHAM','LEMD']
    networks={'EDDF':'DE__ASOS','EHAM':'NL__ASOS','LEMD':'ES__ASOS'}
    metadata=[]
    rawdir=BASE/'data/station_metadata'
    rawdir.mkdir(parents=True,exist_ok=True)
    for station in selected:
        network=networks[station]
        url=f'https://mesonet.agron.iastate.edu/geojson/network/{network}.geojson'
        path=rawdir/f'{network}.geojson'
        sidecar=path.with_suffix('.json')
        if not path.exists():
            response=requests.get(url,timeout=60)
            response.raise_for_status()
            payload=response.json()
            assert payload['type']=='FeatureCollection'
            path.write_bytes(response.content)
            sidecar.write_text(json.dumps(dict(url=url,sha256=digest(path),
                downloaded_utc=datetime.now(timezone.utc).isoformat()),indent=2)+'\n',encoding='utf8')
        provenance=json.loads(sidecar.read_text(encoding='utf8'))
        assert provenance['sha256']==digest(path) and provenance['url']==url
        payload=json.loads(path.read_text(encoding='utf8'))
        matches=[f for f in payload['features'] if f['id']==station]
        assert len(matches)==1
        item=matches[0]
        metadata.append(dict(station=station,network=network,properties=item['properties'],
            coordinates=item['geometry']['coordinates'],**provenance))
        print(station,item['properties'],flush=True)
    parts=[]
    for batch in pq.ParquetFile(INPUT).iter_batches(columns=['airport','C40_CROSS_TIME','GoA'],batch_size=200000):
        frame=batch.to_pandas()
        frame['month']=frame.C40_CROSS_TIME.dt.strftime('%Y-%m')
        parts.append(frame.groupby(['airport','month']).agg(rows=('GoA','size'),go_arounds=('GoA','sum')))
    counts=pd.concat(parts).groupby(level=[0,1]).sum().reset_index()
    groups=counts.loc[counts.airport.isin(selected)&counts.month.isin(['2019-01','2019-07'])]
    assert len(groups)==6 and groups.rows.gt(0).all()
    assert digest(INPUT)==audit['input_sha256']
    counts.to_csv(BASE/'reports/07_c40_month_counts.csv',index=False)
    protocol=dict(created_utc=datetime.now(timezone.utc).isoformat(),
        status='protocol_frozen_before_new_metar_download',input_sha256=audit['input_sha256'],
        previous_airports=audit['selected_airports'],selected_airports=selected,
        selection='Top three previously untested airports by early C40 record count; ICAO alphabetical tie-break; GoA not used',
        months=['2019-01','2019-07'],groups=groups.to_dict('records'),
        station_metadata=metadata,metar_download_executed=False,training_executed=False,
        input_unchanged=True)
    (BASE/'reports/07_expansion_protocol.json').write_text(json.dumps(protocol,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    build_notebook()

def build_notebook():
    cells=[]
    def md(s): cells.append(nbf.v4.new_markdown_cell(s))
    def code(s): cells.append(nbf.v4.new_code_cell(s))
    md('''# 04. Протокол расширения пилота METAR

## 1. Цель и область исследования

Цель следующего этапа — проверить переносимость процедуры сопоставления и
декодирования на дополнительные станции. Протокол фиксируется до загрузки
новых сообщений METAR и оценки их покрытия. Это техническое исследование;
выборка не является вероятностной и не обеспечивает репрезентативность Европы.
Обучение модели и исключение записей не входят в этот этап.
''')
    code(SETUP)
    code('''protocol=json.loads((R/'07_expansion_protocol.json').read_text(encoding='utf8'))
airports=pd.read_csv(R/'01_airport_coverage.csv')
counts=pd.read_csv(R/'07_c40_month_counts.csv')
assert protocol['input_sha256']==manifest['input_sha256']
selected=protocol['selected_airports']
ranked=airports.sort_values(['c40_early','airport'],ascending=[False,True])
assert ranked.loc[~ranked.airport.isin(protocol['previous_airports'])].head(3).airport.tolist()==selected
display(ranked[['airport','rows','c40_early']].assign(
    stage=ranked.airport.map(lambda a:'Расширение' if a in selected else ('Первый пилот' if a in protocol['previous_airports'] else 'Вне пилота'))))
''')
    md('''## 2. Критерий выбора станций

Включаются три ранее не исследованных аэропорта с наибольшим числом записей
с C40 не позже первой попытки; при равенстве применяется алфавитный порядок
ICAO. Частота GoA и погодные характеристики в ранжировании не участвуют.
Германия, Нидерланды и Испания расширяют географию станций. При этом страна
не является доказательством отличий в единицах или структуре METAR.
Отбор по объему смещает исследование в сторону крупных аэропортов.
''')
    code('''from matplotlib.patches import Patch
top=ranked.head(8).sort_values('c40_early')
fig,ax=plt.subplots(figsize=(9,5))
ax.barh(top.airport,top.c40_early/1000,color=[CORAL if a in selected else GREEN for a in top.airport])
ax.set(xlabel='Число записей с C40 не позже первой попытки, тыс.',title='Объем данных и станции расширения пилота')
ax.legend(handles=[Patch(facecolor=CORAL,label='Станции расширения'),Patch(facecolor=GREEN,label='Другие станции')],loc='lower right')
save('11_expansion_station_selection')
''')
    md('''## 3. Проверка метаданных станций

Проверяется наличие единственного идентификатора станции в соответствующей
сети IEM. Сохраняются байты GeoJSON, URL, SHA256 и время загрузки. Координаты
и название приводятся как метаданные источника, без утверждения о независимой
геодезической верификации. Метаданные описывают текущее состояние каталога;
дата начала архива не подтверждает непрерывное покрытие METAR в 2019 году.
Соответствие кода внутри каждого сообщения будет проверено после загрузки.
''')
    code('''metadata=[]
for source in protocol['station_metadata']:
    path=DATA/'station_metadata'/f"{source['network']}.geojson"
    assert sha(path)==source['sha256']
    item=[f for f in json.loads(path.read_text(encoding='utf8'))['features'] if f['id']==source['station']]
    assert len(item)==1
    p=item[0]['properties']
    metadata.append(dict(station=source['station'],network=source['network'],name=p.get('sname'),
        longitude=source['coordinates'][0],latitude=source['coordinates'][1],
        archive_begin=p.get('archive_begin'),source_url=source['url'],sha256=source['sha256']))
readable_table(pd.DataFrame(metadata))
''')
    md('''## 4. Периоды наблюдения и покрытие аналитического входа

Сохраняются январь и июль 2019 года, как в первом пилоте. Периоды определяются
по времени C40 UTC, включая все дни и обе метки. Сохранение месяцев уменьшает
различия дизайна между этапами, но не устраняет различия состава движения.
Два месяца не позволяют оценить годовую сезонность или климатическую репрезентативность.
Число GoA описывается после выбора групп и не используется для их отбора.
''')
    code('''groups=counts.loc[counts.airport.isin(selected)&counts.month.isin(protocol['months'])].copy()
assert len(groups)==6
display(groups)
year=counts.loc[counts.airport.isin(selected)&counts.month.str.startswith('2019-')]
fig,ax=plt.subplots(figsize=(10,4))
for airport in selected:
    part=year.loc[year.airport.eq(airport)].sort_values('month')
    ax.plot(part.month,part.rows/1000,marker='o',label=airport)
ax.set(xlabel='Месяц C40, UTC',ylabel='Число записей, тыс.',title='Помесячное покрытие аналитического входа за 2019 год')
ax.tick_params(axis='x',rotation=45)
ax.legend()
save('12_expansion_monthly_input')
''')
    code('''fig,axes=plt.subplots(1,2,figsize=(10,4))
for ax,label in zip(axes,[False,True]):
    for i,month in enumerate(protocol['months']):
        part=groups.loc[groups.month.eq(month)].set_index('airport').reindex(selected)
        values=part.go_arounds if label else part.rows-part.go_arounds
        ax.bar(np.arange(len(selected))+(i-.5)*.35,values,width=.35,label=month,color=[CORAL,GREEN][i])
    ax.set(xticks=np.arange(len(selected)),xticklabels=selected,ylabel='Число записей',title='GoA' if label else 'Обычные посадки')
    ax.legend(title='Месяц C40, UTC')
save('13_expansion_group_sizes')
display(pd.DataFrame([dict(airport=a,rows=int(g.rows.sum()),go_arounds=int(g.go_arounds.sum())) for a,g in groups.groupby('airport')]))
''')
    md('''## 5. Правила загрузки, сопоставления и проверки

Планируется шесть запросов: три станции × два полных месяца. Каждый запрос
включает предшествующие сутки, routine и special, исходный текст METAR и UTC.
Сохраняются исходные байты, URL, SHA256 и время загрузки; дата наблюдения
не подменяется датой фактического получения сообщения.

Сопоставление: последнее наблюдение строго до C40, основной предел 60 минут,
чувствительность 30/90 минут. Разные тексты на одной отметке неоднозначны;
COR помечается отдельно. Повторный выбор при предполагаемой задержке
0/5/10/15 минут воспроизводит метод ноутбука 03. Качество не используется для
поиска более старого сообщения. Отсутствие погоды сохраняет исходную запись.

После загрузки проверяются единицы ветра и давления, коды видимости и облаков,
AUTO, COR, прогнозные группы, неизвестные токены и заполненность. Все
остаточные случаи учитываются; сомнительные значения автоматически не исправляются.
Проверка полного текста и основных полей остается раздельной.
''')
    code('''display(pd.DataFrame([
    {'Проверка':'Отбор без частоты GoA','Статус':'Выполнено'},
    {'Проверка':'Метаданные трех станций и SHA256','Статус':'Выполнено'},
    {'Проверка':'Шесть групп по C40 UTC','Статус':'Выполнено'},
    {'Проверка':'Загрузка новых METAR','Статус':'Запланировано'},
    {'Проверка':'Покрытие, декодирование и остаточные случаи','Статус':'Запланировано'},
    {'Проверка':'Фактическая историческая доступность','Статус':'Не подтверждена'}]))
assert sha(PROJECT/'europe/data/analytical/06_05_cases_excluded.parquet')==protocol['input_sha256']
assert protocol['metar_download_executed'] is False and protocol['training_executed'] is False
''')
    notebook=make(cells)
    path=NOTEBOOKS/'04_expansion_protocol.ipynb'
    manager=KernelManager(kernel_name='python3')
    manager.kernel_spec.argv=[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}']
    client=NotebookClient(notebook,km=manager,timeout=300,resources={'metadata':{'path':str(NOTEBOOKS)}},allow_errors=False)
    client.on_cell_start=lambda cell,cell_index,**kwargs:print(f'cell {cell_index+1}/{len(cells)}',flush=True)
    try:
        client.execute()
    finally:
        nbf.write(notebook,path)
    nbf.validate(notebook)
    output=[o for c in notebook.cells if c.cell_type=='code' for o in c.outputs]
    assert not any(o.output_type=='error' for o in output)
    result=dict(notebook=path.name,executed=True,embedded_figures=sum('image/png' in o.get('data',{}) for o in output))
    assert result['embedded_figures']==3
    (BASE/'reports/07_notebook_verification.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
    print(result,flush=True)

if __name__=='__main__':
    main()
