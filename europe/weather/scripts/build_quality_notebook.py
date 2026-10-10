"""Выполненный аудит основных полей и сценариев задержки публикации."""
import json
import sys
import nbformat as nbf
from nbclient import NotebookClient
from jupyter_client.manager import KernelManager
from build_weather_notebooks import BASE, NOTEBOOKS, SETUP, make

cells = []
def md(text): cells.append(nbf.v4.new_markdown_cell(text))
def code(text): cells.append(nbf.v4.new_code_cell(text))

md('''# 03. Качество основных полей и доступность к C40

Проверяем отдельно замечания к полному сообщению и каноническую часть до QNH.
Исходные тексты, строки и прежние флаги качества сохраняются. Проверенный
префикс не означает исправленное или полностью валидное сообщение.

Задержки 0, 5, 10 и 15 минут — **предположения для анализа чувствительности**.
В сохраненном ответе IEM имеются station, valid и metar, но нет времени
фактического получения сообщения. Архив не подтверждает историческую доступность.
[Описание IEM](https://mesonet.agron.iastate.edu/info/datasets/metar.html) ·
[API IEM](https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py?help=) ·
[Предыдущий ноутбук](02_metar_decoding.ipynb)
''')
code(SETUP)
code('''from decode_metar import decode, split_sections, PRESSURE, TEMPERATURE
reports = pd.read_parquet(DATA / '04_decoded_reports.parquet')
events = pd.read_parquet(DATA / '03_pilot_matches.parquet')
protected = {p:sha(p) for p in [DATA/'04_decoded_reports.parquet', DATA/'04_pilot_features.parquet', DATA/'03_pilot_matches.parquet']}
fields = [f for f in reports.columns if f.startswith(('wind_','visibility_','cloud_',
    'present_weather_','qnh_','vertical_visibility_')) or f in ['temperature_token',
    'temperature_c','dewpoint_c','cavok','gust_group_reported',
    'cumulonimbus_reported','towering_cumulus_reported']]
def same(a,b):
    if a is None or (isinstance(a,float) and np.isnan(a)):
        return b is None or (isinstance(b,float) and np.isnan(b))
    return a == b
def audit_core(raw,station,stamp,original):
    observed = split_sections(raw)[0]
    tokens = observed.split()
    pressure = [i for i,t in enumerate(tokens) if PRESSURE.fullmatch(t)]
    if len(pressure) != 1:
        return False, '', 'qnh_missing_or_multiple'
    end = pressure[0]
    temperatures = [t for t in tokens[:end] if TEMPERATURE.fullmatch(t)]
    if len(temperatures) != 1:
        return False, '', 'temperature_missing_or_multiple'
    prefix = ' '.join(tokens[:end+1])
    checked = decode(prefix,station,stamp.to_pydatetime())
    changed = [f for f in fields if not same(original[f],checked[f])]
    ok = checked['decoded_candidate'] and not changed
    return ok, prefix, ('value_disagreement:'+','.join(changed) if changed else checked['decode_issues'])
audits = [audit_core(row.metar,row.station,row.metar_utc,row._asdict())
          for row in reports.itertuples(index=False)]
reports[['core_validated','core_prefix','core_issues']] = pd.DataFrame(audits,index=reports.index)
assert len(reports)==9223 and reports.metar_id.is_unique
residual = reports.loc[~reports.decoded_candidate,
    ['metar_id','station','metar_utc','metar','core_prefix','core_validated','core_issues']]
readable_table(residual)
residual.to_csv(R/'06_core_residual_audit.csv',index=False)
display(reports.groupby(['decoded_candidate','core_validated']).size().rename('reports').reset_index())
''')
md('''## Повторный выбор сообщения при предполагаемой задержке

Для каждого C40 выбираем последнее сообщение, для которого время наблюдения
плюс задержка строго меньше C40. Давность считаем от наблюдения, максимум
30, 60 или 90 минут. Сообщения с замечаниями и COR не пропускаем при поиске:
сначала выбираем последнее, затем проверяем качество. Таким образом качество
не вызывает скрытый переход к более старому сообщению. Неоднозначные тексты
на одной временной отметке не дают кандидата. COR остается отдельным сценарием.
''')
code('''rows = []
zero_parts = []
for latency in [0,5,10,15]:
    for (airport,month),part in events.groupby(['airport','month']):
        right = reports.loc[reports.station.eq(airport) & reports.archive_month.eq(month)].copy()
        right['ambiguous'] = right.groupby('metar_utc').metar.transform('nunique').gt(1)
        right = right.drop_duplicates('metar_utc').sort_values('metar_utc')
        right['assumed_available_utc'] = right.metar_utc+pd.Timedelta(minutes=latency)
        joined = pd.merge_asof(part[['source_row','C40_CROSS_TIME','GoA']].sort_values('C40_CROSS_TIME'),
            right[['assumed_available_utc','metar_utc','metar_id','decoded_candidate','core_validated','correction_reported','ambiguous']],
            left_on='C40_CROSS_TIME',right_on='assumed_available_utc',direction='backward',allow_exact_matches=False)
        joined['age'] = (joined.C40_CROSS_TIME-joined.metar_utc).dt.total_seconds()/60
        assert joined.loc[joined.metar_utc.notna(),'age'].gt(latency).all()
        if latency==0:
            zero_parts.append(joined[['source_row','metar_utc']])
        for limit in [30,60,90]:
            timely = joined.age.le(limit) & ~joined.ambiguous.eq(True)
            for quality in ['decoded_candidate','core_validated']:
                for exclude_cor in [False,True]:
                    candidate = timely & joined[quality].eq(True)
                    if exclude_cor:
                        candidate &= ~joined.correction_reported.eq(True)
                    for label in [False,True]:
                        group = joined.GoA.eq(label)
                        rows.append(dict(airport=airport,month=month,latency_minutes=latency,
                            maximum_age_minutes=limit,quality=quality,exclude_cor=exclude_cor,GoA=label,
                            total=int(group.sum()),candidates=int((group & candidate).sum())))
scenarios = pd.DataFrame(rows)
scenarios['coverage_percent'] = 100*scenarios.candidates/scenarios.total
scenarios.to_csv(R/'06_availability_scenarios.csv',index=False)
baseline = pd.concat(zero_parts).set_index('source_row').sort_index()
pd.testing.assert_series_equal(baseline.metar_utc,events.set_index('source_row').sort_index().metar_utc)
summary = scenarios.groupby(['latency_minutes','maximum_age_minutes','quality','exclude_cor','GoA'])[['total','candidates']].sum().reset_index()
summary['coverage_percent'] = 100*summary.candidates/summary.total
display(summary.loc[summary.maximum_age_minutes.eq(60)])
''')
code('''fig,axes=plt.subplots(1,2,figsize=(12,4))
for ax,label in zip(axes,[False,True]):
    for quality,color in [('decoded_candidate',CORAL),('core_validated',GREEN)]:
        subset=summary.loc[summary.maximum_age_minutes.eq(60)&summary.GoA.eq(label)&summary.quality.eq(quality)]
        for exclude,style in [(False,'-'),(True,'--')]:
            series=subset.loc[subset.exclude_cor.eq(exclude)].sort_values('latency_minutes')
            ax.plot(series.latency_minutes,series.coverage_percent,style,marker='o',color=color,
                label=('Полный текст' if quality=='decoded_candidate' else 'Основные поля')+(' без COR' if exclude else ''))
    ax.set(title='GoA' if label else 'Обычные посадки',xlabel='Предполагаемая задержка, мин',ylabel='Кандидаты, %',xticks=[0,5,10,15])
axes[0].legend(fontsize=9)
save('08_latency_quality_coverage')
''')
code('''fig,axes=plt.subplots(1,2,figsize=(12,4))
for ax,label in zip(axes,[False,True]):
    for airport in sorted(events.airport.unique()):
        subset=scenarios.loc[scenarios.maximum_age_minutes.eq(60)&scenarios.GoA.eq(label)&
            scenarios.quality.eq('core_validated')&scenarios.exclude_cor.eq(True)&scenarios.airport.eq(airport)]
        series=subset.groupby('latency_minutes')[['total','candidates']].sum()
        ax.plot(series.index,100*series.candidates/series.total,marker='o',label=airport)
    ax.set(title='GoA' if label else 'Обычные посадки',xlabel='Предполагаемая задержка, мин',ylabel='Основные поля без COR, %',xticks=[0,5,10,15])
axes[0].legend()
save('09_latency_airport_coverage')
''')
code('''fig,axes=plt.subplots(1,2,figsize=(12,4))
for ax,label in zip(axes,[False,True]):
    for limit in [30,60,90]:
        subset=summary.loc[summary.maximum_age_minutes.eq(limit)&summary.GoA.eq(label)&
            summary.quality.eq('core_validated')&summary.exclude_cor.eq(True)].sort_values('latency_minutes')
        ax.plot(subset.latency_minutes,subset.coverage_percent,marker='o',label=f'Давность ≤ {limit} мин')
    ax.set(title='GoA' if label else 'Обычные посадки',xlabel='Предполагаемая задержка, мин',ylabel='Основные поля без COR, %',xticks=[0,5,10,15])
axes[0].legend()
save('10_latency_age_limits')
''')
md('''## Границы результата и следующий этап

Проверка префикса относится только к наблюдаемым основным полям до QNH.
RE, WS, состояние ВПП и неразмеченный хвост требуют самостоятельного разбора;
никакие недостающие разделители не вставлялись. Значения пропусков остаются
пропусками. Даже положительный флаг не означает наличие каждого признака.
Покрытие без COR может увеличиваться с задержкой: повторный выбор меняет
сообщение, иногда на более старое без COR. Это не улучшение фактической погоды.
Шкалы покрытия на первых двух графиках увеличены для видимости малых различий.

Перед расширением следует проверить новые станции и форматы, затем повторить
полные месяцы с сохранением сырого ответа, SHA256 и времени загрузки. Критерий
выбора станций должен опираться на объем и разнообразие источника, а не GoA.
Историческое время публикации и доступность аэропорта к C40 остаются открытыми
вопросами. Массовая загрузка и обучение здесь не выполняются.
''')
code('''for path,digest in protected.items():
    assert sha(path)==digest
assert len(events)==manifest['rows'] and int(events.GoA.sum())==manifest['go_arounds']
assert not audit_core('EGCC 221050Z VRB03KT 9999 BKN015 02/00 Q1000 Q1001','EGCC',pd.Timestamp('2019-01-22T10:50Z'),{})[0]
assert not audit_core('EGCC 221050Z VRB03KT 9999 BKN015','EGCC',pd.Timestamp('2019-01-22T10:50Z'),{})[0]
assert not audit_core('EGCC 221050Z VRB03KT 9999 BKN015 Q1000','EGCC',pd.Timestamp('2019-01-22T10:50Z'),{})[0]
display(Markdown('**Проверено:** исходные файлы неизменны; нулевая задержка воспроизводит прежний выбор METAR; неоднозначный QNH и отсутствующие основные группы не принимаются.'))
''')

def main():
    notebook=make(cells)
    path=NOTEBOOKS/'03_quality_and_availability.ipynb'
    manager=KernelManager(kernel_name='python3')
    manager.kernel_spec.argv=[sys.executable,'-m','ipykernel_launcher','-f','{connection_file}']
    client=NotebookClient(notebook,km=manager,timeout=300,resources={'metadata':{'path':str(NOTEBOOKS)}},allow_errors=False)
    client.on_cell_start=lambda cell,cell_index,**kwargs: print(f'cell {cell_index+1}/{len(cells)}',flush=True)
    try:
        client.execute()
    finally:
        nbf.write(notebook,path)
    nbf.validate(notebook)
    outputs=[o for c in notebook.cells if c.cell_type=='code' for o in c.outputs]
    assert not any(o.output_type=='error' for o in outputs)
    result=dict(notebook=path.name,executed=True,embedded_figures=sum('image/png' in o.get('data',{}) for o in outputs))
    assert result['embedded_figures']==3
    (BASE/'reports/06_notebook_verification.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
    print(result)

if __name__=='__main__':
    main()
