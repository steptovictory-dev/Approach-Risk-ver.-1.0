"""Прозрачные рабочие ноутбуки исследования с расчетами в ячейках."""
from pathlib import Path
import sys
import nbformat as nbf

STUDY = Path(__file__).resolve().parents[1]
SETUP = '''# Подключаем библиотеки и находим корень проекта
from pathlib import Path
import hashlib
import json
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython import get_ipython
from IPython.display import display, Markdown
root = next(p for p in [Path.cwd(), *Path.cwd().parents]
            if (p/'research/c40_geometry').is_dir())
study = root/'research/c40_geometry'
# Подключаем проверенные геодезические функции без запуска скриптов
sys.path.insert(0, str(root/'europe/scripts'))
from audit_c40_radial_cases import ellipsoid_distance, validate_distance
from investigate_c40_geometry import distance
if get_ipython() is not None:
    get_ipython().run_line_magic('matplotlib', 'inline')
plt.rcParams.update({'figure.dpi':120, 'font.size':11,
                     'axes.spines.top':False, 'axes.spines.right':False})
# Таблицы округляем только при отображении, расчетные значения сохраняем
pd.set_option('display.max_columns', 14)
'''


def build(name, title, question, inputs, sections, conclusion):
    if '--only' in sys.argv and name != sys.argv[sys.argv.index('--only')+1]:
        return
    path = STUDY/'notebooks'/name
    previous = nbf.read(path, as_version=4)
    # Сохраняем исходные методические пояснения и ссылки без потери контекста
    originals = previous.metadata.get('original_methodology')
    if originals is None:
        originals = [c.source for c in previous.cells if c.cell_type == 'markdown']
    cells = [nbf.v4.new_markdown_cell(f'# {title}\n\n**Исследовательский вопрос:** {question}\n\n'
        f'**Входные материалы:** {inputs}\n\nРасчеты ниже выполняются в ячейках. '
        'Все исходные строки сохраняются. Округление таблиц используется только для чтения результатов.'),
        nbf.v4.new_code_cell(SETUP)]
    for heading, explanation, code in sections:
        cells += [nbf.v4.new_markdown_cell(f'## {heading}\n\n{explanation}'), nbf.v4.new_code_cell(code)]
    cells += [nbf.v4.new_markdown_cell('## Что установлено и что остается открытым\n\n'+conclusion),
              nbf.v4.new_markdown_cell('## Методические подробности и происхождение источников\n\n'
                                      +'\n\n---\n\n'.join(originals))]
    nb = nbf.v4.new_notebook(cells=cells)
    nb.metadata.update(kernelspec=dict(display_name='Python 3', language='python', name='python3'),
                       curated_research=True, original_methodology=originals)
    if '--execute' in sys.argv:
        from nbclient import NotebookClient
        NotebookClient(nb, timeout=300, resources={'metadata':{'path':str(STUDY.parents[1])}}).execute()
    nbf.validate(nb)
    nbf.write(nb, path)
    print(f'Сохранен: {name}', flush=True)


def main():
    build('01_historical_arp.ipynb', '01 · Исторические ARP: независимая сверка',
          'Совпадают ли восстановленные центры с опубликованными координатами выбранных исторических документов?',
          'реестры источников, локальные документы AIP и восстановленные центры из этапа 07 Европы', [
        ('1. Проверяем происхождение документов',
         'Контрольная сумма подтверждает, что локальный документ совпадает с ранее сохраненным. Дата AIRAC, дата информации схемы и дата публикации не взаимозаменяемы.',
         '''# Проверяем целостность всех сохраненных источников
sources = json.loads((study/'literature/source_manifest.json').read_text(encoding='utf8'))
ad_sources = json.loads((study/'literature/historical_ad_manifest.json').read_text(encoding='utf8'))
checks = []
for item in sources + ad_sources:
    if item['status'] == 'saved':
        path = root/item['file']
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        assert actual == item['sha256'], str(path)
        checks.append({'Документ':path.name, 'SHA256 совпадает':True})
display(pd.DataFrame(checks))
# Показываем координаты вместе с датами и обозначением источника
pilot = pd.read_csv(study/'reports/historical_arp_sources.csv')
display(pilot[['airport','airac','designation','chart_information_date','publication_date',
               'published_lat','published_lon']])'''),
        ('2. Пересчитываем расхождение координат',
         'Разности считаем в угловых секундах. Для практически совпадающих точек используем локальную линейную оценку расстояния; это не оценка точности слежения за самолетом.',
         '''# Присоединяем восстановленный центр по коду аэропорта
centers = pd.read_csv(root/'europe/reports/airports/c40_geometry/07_07_best_formula.csv')
comparison = pilot.merge(centers[['airport','fit_lat','fit_lon']], on='airport', validate='many_to_one')
# Переводим разности градусов в угловые секунды
comparison['delta_lat'] = (comparison.published_lat-comparison.fit_lat)*3600
comparison['delta_lon'] = (comparison.published_lon-comparison.fit_lon)*3600
# Оцениваем локальное расстояние с учетом широты
comparison['recomputed_m'] = 6371000*np.pi/(180*3600)*np.hypot(
    comparison.delta_lat, comparison.delta_lon*np.cos(np.radians(comparison.published_lat)))
assert np.allclose(comparison.recomputed_m, comparison.approximate_difference_m, atol=1e-7, rtol=0)
display(comparison[['airport','airac','source_id','delta_lat','delta_lon','recomputed_m']])'''),
        ('3. Показываем временное покрытие',
         'Каждая отметка – отдельный проверенный срез. Линия времени не доказывает отсутствие изменений между документами.',
         '''# Показываем доступные даты вступления в силу по каждому аэропорту
fig, ax = plt.subplots(figsize=(10,3.5))
for i, (airport, group) in enumerate(pilot.groupby('airport')):
    dates = pd.to_datetime(group.airac).drop_duplicates().sort_values()
    ax.scatter(dates, [i]*len(dates), s=80, label=airport)
ax.set(yticks=range(pilot.airport.nunique()), yticklabels=sorted(pilot.airport.unique()),
       xlabel='Дата AIRAC', title='Проверенные исторические срезы')
ax.grid(axis='x', alpha=.2)
plt.tight_layout()
plt.show()
# Открытые аэропорты остаются явно отмеченными в общем реестре
registry = pd.read_csv(study/'reports/arp_reference_registry.csv')
display(registry)
display(registry.historical_status.value_counts().rename('Число аэропортов'))''')],
        'Координаты выбранных документов сверены с центрами. Срезы не покрывают непрерывно весь период; ранняя часть 2019 года EGCC остается без этой независимой проверки. Совпадение ARP не объясняет механизм записи C40.')

    build('02_egcc_geometry.ipynb', '02 · EGCC: радиус C40, сфера и WGS84',
          'Как меняется радиальное отклонение при выборе метрики и насколько велик возможный вклад координатной сетки?',
          'аналитический Parquet Европы и проверенные исторические координаты EGCC', [
        ('1. Проверяем вход и читаем EGCC',
         'Контрольная сумма связывает расчет с конкретной версией данных. Фильтр выбирает аэропорт, не удаляя строки исходного файла.',
         '''import pyarrow.parquet as pq
# Сверяем версию исходного аналитического файла потоковым чтением
source = root/'europe/data/analytical/06_05_cases_excluded.parquet'
audit = json.loads((study/'reports/egcc_geometry_audit.json').read_text(encoding='utf8'))
sha = hashlib.sha256()
with source.open('rb') as stream:
    for chunk in iter(lambda:stream.read(8*1024*1024), b''):
        sha.update(chunk)
assert sha.hexdigest() == audit['input_sha256']
display(validate_distance())
# Читаем только поля, необходимые для геометрического расчета
fields = ['flight_id','GoA','C40_CROSS_LAT','C40_CROSS_LON','C40_BEARING']
frame = pq.read_table(source, columns=fields, filters=[('airport','=','EGCC')]).to_pandas()
display(frame.head())
display(frame.GoA.value_counts().rename('Число записей'))'''),
        ('2. Считаем расстояния и отклонения',
         '40 морских миль соответствуют 74,08 км. Разность расстояния и этого радиуса – диагностическая величина. Используемая авторами модель расстояния пока не установлена.',
         '''# Проверяем координаты исторического ARP перед расчетом
arp_lat, arp_lon = audit['arp_lat'], audit['arp_lon']
official = pd.read_csv(study/'reports/historical_arp_sources.csv')
official = official[(official.airport == 'EGCC') & (official.designation == 'explicit_ARP_AD2_2')]
assert np.allclose(official.published_lat, arp_lat, atol=1e-12, rtol=0)
assert np.allclose(official.published_lon, arp_lon, atol=1e-12, rtol=0)
lat, lon = frame.C40_CROSS_LAT.to_numpy(), frame.C40_CROSS_LON.to_numpy()
# Сравниваем одну и ту же точку на сфере и эллипсоиде
frame['sphere_km'] = distance(arp_lat, arp_lon, lat, lon)
frame['wgs84_km'] = ellipsoid_distance(arp_lat, arp_lon, lat, lon)
frame['deviation_m'] = (frame.wgs84_km-74.08)*1000
frame['sphere_deviation_m'] = (frame.sphere_km-74.08)*1000
frame['model_difference_m'] = (frame.wgs84_km-frame.sphere_km)*1000
display(frame.groupby('GoA')[['deviation_m','sphere_deviation_m','model_difference_m']].agg(['median','min','max']).round(2))'''),
        ('3. Оцениваем вклад координатной сетки',
         'Проверяем два условных сценария: ±0,5 и ±1 угловая секунда по каждой координате. Алгоритм округления неизвестен. Расстояния до углов локальной ячейки дают оценку смещения для этих сценариев.',
         '''# Рассчитываем расстояние до четырех углов для каждого сценария
for name, width in [('nearest',.5), ('one_second',1.)]:
    corners = [1000*ellipsoid_distance(lat,lon,lat+dy*width/3600,lon+dx*width/3600)
               for dy in [-1,1] for dx in [-1,1]]
    frame[f'grid_{name}_bound_m'] = np.max(corners, axis=0)
display(frame[['grid_nearest_bound_m','grid_one_second_bound_m']].describe().round(2))
# Показываем превышения отдельно для обеих метрик без исключения записей
diagnostics = frame.groupby('GoA').agg(rows=('flight_id','size'))
for name in ['deviation_m','sphere_deviation_m']:
    diagnostics[name+'_gt_bound'] = frame.groupby('GoA')[name].apply(
        lambda s:int((s.abs() > frame.loc[s.index,'grid_one_second_bound_m']).sum()))
display(diagnostics)'''),
        ('4. Сопоставляем распределения и азимут',
         'GoA показан как описательный разрез. На рисунке распределения виден только диапазон от −1200 до 1700 м; экстремумы выше приведены в таблице. Различие метрик зависит от направления к точке.',
         '''fig, axes = plt.subplots(1,2,figsize=(13,4.5))
# Используем одинаковые границы столбцов для обеих групп и метрик
for label, color in [(False,'#2471A3'),(True,'#D68910')]:
    group = frame[frame.GoA.eq(label)]
    for metric, style in [('deviation_m','-'),('sphere_deviation_m','--')]:
        axes[0].hist(group[metric], bins=np.arange(-1200,1701,25), density=True,
                     histtype='step', color=color, linestyle=style, label=f'GoA={label} · {metric}')
axes[0].set(xlabel='Радиальное отклонение, м', ylabel='Плотность', title='WGS84 и сфера')
axes[0].legend(fontsize=8)
# Для читаемости графика берем регулярную выборку только при отображении
sample = frame.iloc[::max(1,len(frame)//6000)]
axes[1].scatter(sample.C40_BEARING,sample.model_difference_m,s=4,alpha=.35)
axes[1].set(xlabel='Азимут аэропорт → C40, градусы',ylabel='WGS84 минус сфера, м',title='Зависимость от азимута')
plt.tight_layout()
plt.show()
# Сверяем пересчитанные числа с сохраненным аудитом
assert len(frame) == audit['rows'] and int(frame.GoA.sum()) == audit['goa']
assert np.isclose(frame.grid_one_second_bound_m.max(),audit['max_one_second_grid_bound_m'])''')],
        'Метрика влияет на величину отклонения и состав диагностических случаев. Превышение выбранной границы сетки не устанавливает ошибку записи: определение пересечения и исходные траектории еще не проверены. Азимут от ARP не является магнитным курсом захода.')

    build('03_egcc_cases.ipynb', '03 · EGCC: шесть диагностических записей',
          'Какие факты подтверждены для каждой записи и что может показать условный временной сценарий?',
          'локальный разбор случаев, соседние записи тех же бортов и исходный Parquet', [
        ('1. Читаем случаи и перепроверяем исходные поля',
         'Порог 1 км выбран для диагностики по WGS84. Сверка с исходником не устанавливает физическую правильность позиции, но исключает изменение этих полей при подготовке.',
         '''import pyarrow.parquet as pq
cases = pd.read_csv(study/'reports/03_egcc_case_review.csv')
display(cases[['flight_id','icao24','typecode','GoA','sphere_m','wgs84_m','minutes_to_landing']].round(2))
# Подтягиваем только выбранные поля исходного EGCC
key = ['icao24','airport','landing_time']
fields = key+['flight_id','C40_CROSS_TIME','C40_CROSS_LAT','C40_CROSS_LON','C40_BEARING','n_attempts']
raw = pq.read_table(root/'europe/data/source/osn23_landings_merged.parquet.gzip',
                    columns=fields,filters=[('airport','=','EGCC')]).to_pandas(ignore_metadata=True)
# Приводим только представление времени для сопоставления ключей
for col in ['landing_time','C40_CROSS_TIME']:
    raw[col] = pd.to_datetime(raw[col],utc=True)
    cases[col] = pd.to_datetime(cases[col],utc=True)
merged = cases.merge(raw,on=key,suffixes=('_review','_source'),validate='one_to_one')
for field in [f for f in fields if f not in key]:
    assert merged[field+'_review'].eq(merged[field+'_source']).all(), field
display(Markdown(f'**Сверено с исходником: {len(merged)} записей**'))'''),
        ('2. Проверяем чувствительность к метрике',
         'Пересчитываем радиальные отклонения непосредственно по координатам. Случаи нельзя объявить ошибками или удалить только по диагностическому порогу.',
         '''# Используем подтвержденные координаты EGCC
lat, lon = cases.C40_CROSS_LAT.to_numpy(), cases.C40_CROSS_LON.to_numpy()
sphere = 1000*(distance(53.35388888888889,-2.275,lat,lon)-74.08)
wgs84 = 1000*(ellipsoid_distance(53.35388888888889,-2.275,lat,lon)-74.08)
assert np.allclose(sphere,cases.sphere_m) and np.allclose(wgs84,cases.wgs84_m)
# Сравниваем отклонения на одной шкале
fig, ax = plt.subplots(figsize=(10,4.5))
y = np.arange(len(cases))
ax.barh(y-.17,sphere,height=.3,label='Сфера',color='#7C3AED')
ax.barh(y+.17,wgs84,height=.3,label='WGS84',color='#059669')
for value in [-1000,1000]:
    ax.axvline(value,color='#888',linestyle='--')
ax.set(yticks=y,yticklabels=cases.flight_id,xlabel='Расстояние минус 74,08 км, м',title='Порог зависит от метрики')
ax.legend()
plt.tight_layout()
plt.show()'''),
        ('3. Считаем условный временной сдвиг',
         'Формула Δt = |отклонение| / предполагаемая радиальная скорость проверяет совместимость сценария. Скорости 50–150 м/с не измерены; частота исходных CPR неизвестна.',
         '''# Создаем сценарии явно, не выдавая скорости за наблюдения
rows = []
for case in cases.itertuples():
    for model, residual in [('sphere',case.sphere_m),('wgs84',case.wgs84_m)]:
        for speed in [50.,100.,150.]:
            rows.append({'flight_id':case.flight_id,'model':model,'radial_speed_m_s':speed,
                         'hypothetical_seconds':abs(residual)/speed})
scenarios = pd.DataFrame(rows)
display(scenarios.pivot(index=['flight_id','model'],columns='radial_speed_m_s',values='hypothetical_seconds').round(2))
# Сверяем сценарии с прежним отчетом по всем ключам
saved = pd.read_csv(study/'reports/03_egcc_sampling_scenarios.csv')
check = scenarios.merge(saved,on=['flight_id','model','radial_speed_m_s'],suffixes=('_new','_saved'),validate='one_to_one')
assert np.allclose(check.hypothetical_seconds_new,check.hypothetical_seconds_saved)
# Контекст соседних записей не считается независимой контрольной группой
display(cases[['flight_id','point_rows','same_board_other_rows_30d','sector_other_rows','n_attempts']])''')],
        'Поля всех шести случаев сверены с исходником. Условный временной сценарий не доказывает частоту наблюдений или причину отклонения. Все строки сохраняются; для механизма нужна последовательность наблюдений.')

    build('05_public_trajectory.ipynb', '05 · Публичная траектория: EPWA и ENT73JH',
          'Согласуются ли публичные наблюдения с координатами и временем C40 выбранного рейса?',
          'извлеченные state vectors OpenSky и метаданные ENT73JH_21100', [
        ('1. Читаем последовательность и проверяем свежесть',
         'Работаем с ранее извлеченными наблюдениями. Контрольные суммы и URL архивов показаны отдельно. Центр EPWA восстановлен из метаданных; исторический ARP пока не подтвержден.',
         '''# Читаем оба часа и ограничиваем последовательность временем посадки плюс две минуты
case = pd.read_csv(study/'reports/05_public_monday_candidates.csv').set_index('flight_id').loc['ENT73JH_21100']
track = pd.concat([pd.read_csv(study/f'data/public_pilot/ENT73JH_21100_state_vectors{s}.csv')
                   for s in ['', '_23']],ignore_index=True).sort_values('time')
t = pd.Timestamp(case.C40_CROSS_TIME)
landing = pd.Timestamp(case.landing_time)
track['utc'] = pd.to_datetime(track.time,unit='s',utc=True)
track = track[track.utc <= landing+pd.Timedelta(minutes=2)].copy()
track['offset_s'] = (track.utc-t).dt.total_seconds()
track['position_age_s'] = track.time-track.lastposupdate
track['contact_age_s'] = track.time-track.lastcontact
display(track[['utc','lat','lon','callsign','position_age_s','contact_age_s']].head())
display(track[['position_age_s','contact_age_s']].describe().round(3))
# Показываем происхождение архивов без повторной загрузки
for suffix in ['', '_23']:
    manifest = json.loads((study/f'reports/05_public_pilot_download{suffix}.json').read_text(encoding='utf8'))
    display(manifest)'''),
        ('2. Сопоставляем позиции и ищем пересечение',
         'Расстояние до метаданных и расстояние до центра отвечают на разные вопросы. Момент пересечения оцениваем линейно между соседними наблюдениями, а не считаем измеренным.',
         '''# Считаем две метрики радиуса и отдельное расстояние до точки метаданных
lat, lon = track.lat.to_numpy(),track.lon.to_numpy()
track['sphere_km'] = distance(case.fit_lat,case.fit_lon,lat,lon)
track['wgs84_km'] = ellipsoid_distance(case.fit_lat,case.fit_lon,lat,lon)
track['metadata_distance_m'] = 1000*ellipsoid_distance(case.C40_CROSS_LAT,case.C40_CROSS_LON,lat,lon)
local = track[track.offset_s.abs() <= 300].copy()
nearest = local.iloc[local.offset_s.abs().argmin()]
display(pd.DataFrame([{'Сдвиг времени, с':nearest.offset_s,'Расхождение позиций, м':nearest.metadata_distance_m,
                       'Возраст позиции, с':nearest.position_age_s}]).round(3))
# Находим переход снаружи внутрь отдельно для каждой метрики
crossings = []
for model in ['sphere_km','wgs84_km']:
    for i in range(1,len(track)):
        a,b = track.iloc[i-1],track.iloc[i]
        if a[model] >= 74.08 and b[model] < 74.08:
            weight = (a[model]-74.08)/(a[model]-b[model])
            epoch = a.time+weight*(b.time-a.time)
            crossings.append({'Модель':model,'Интервал, с':b.time-a.time,
                              'Оценка UTC':pd.to_datetime(epoch,unit='s',utc=True),
                              'После времени метаданных, с':epoch-t.timestamp()})
display(pd.DataFrame(crossings))
saved = json.loads((study/'reports/05_public_pilot_audit.json').read_text(encoding='utf8'))
assert np.isclose(nearest.metadata_distance_m,saved['closest_time_point_distance_m'])'''),
        ('3. Показываем временное и пространственное расхождение',
         'Пунктир отмечает границу 40 NM; вертикальная линия – время C40 метаданных. Даже точное совпадение времени не гарантирует совпадение позиций.',
         '''fig, axes = plt.subplots(1,2,figsize=(13,4.5))
# Сравниваем радиусы около времени метаданных
for model in ['sphere_km','wgs84_km']:
    axes[0].plot(local.offset_s,local[model],'.-',label=model)
axes[0].axhline(74.08,color='#888',linestyle='--',label='40 NM')
axes[0].axvline(0,color='#888',linestyle=':')
axes[0].set(xlabel='Относительно времени C40, с',ylabel='Радиус, км',title='Оценка пересечения границы')
axes[0].legend(fontsize=9)
# Отдельно показываем расстояние между источниками
axes[1].plot(local.offset_s,local.metadata_distance_m/1000,color='#A93226')
axes[1].axvline(0,color='#888',linestyle=':')
axes[1].set(xlabel='Относительно времени C40, с',ylabel='Расхождение позиций, км',title='Сопоставление с точкой метаданных')
plt.tight_layout()
plt.show()''')],
        'Публичная последовательность дает оценку пересечения и свежести позиции. Расхождение около 43 км остается нерешенным: определение радиальной границы само по себе его не объясняет. OpenSky ADS-B не предполагается тождественным исходным CPR EUROCONTROL.')

    build('06_epwa_controls.ipynb', '06 · Контрольные рейсы и временное выравнивание',
          'Насколько меняется сопоставление, если восстановить позицию точно ко времени C40?',
          'метаданные контрольных случаев и локальные последовательности EPWA/EGCC', [
        ('1. Проверяем состав контрольного пилота',
         'Название этапа историческое: в таблице есть рейсы EPWA и EGCC. Сопоставление по ICAO24, времени и callsign не подтверждает тождество физических измерений разных источников.',
         '''controls = pd.read_csv(study/'reports/06_epwa_control_comparison.csv')
observed = pd.read_csv(study/'data/public_pilot/EPWA_control_geometry.csv')
display(controls[['flight_id','airport','status','expected_callsign','observed_callsign','callsign_matches','observed_rows']])
display(controls.groupby(['airport','status']).size().rename('Число случаев'))'''),
        ('2. Пересчитываем позицию ко времени C40',
         'Используем ближайшие доступные наблюдения до и после времени метаданных. При точном наблюдении интервал равен нулю. Если пары нет, результат остается отсутствующим.',
         '''# Восстанавливаем позицию только по концам временного интервала
rows = []
for case in controls.itertuples():
    track = observed[observed.case_flight_id.eq(case.flight_id)].sort_values('time')
    if track.empty:
        continue
    epoch = pd.Timestamp(case.c40_utc).timestamp()
    before, after = track[track.time <= epoch].tail(1), track[track.time >= epoch].head(1)
    if before.empty or after.empty:
        continue
    a,b = before.iloc[0],after.iloc[0]
    gap = b.time-a.time
    weight = 0 if gap == 0 else (epoch-a.time)/gap
    lat = a.lat+weight*(b.lat-a.lat)
    lon = a.lon+weight*(b.lon-a.lon)
    difference = float(1000*ellipsoid_distance(case.metadata_lat,case.metadata_lon,np.array([lat]),np.array([lon]))[0])
    rows.append({'flight_id':case.flight_id,'gap_s':gap,'recomputed_m':difference})
comparison = controls.merge(pd.DataFrame(rows),on='flight_id',how='left',validate='one_to_one')
valid = comparison.recomputed_m.notna()
assert np.allclose(comparison.loc[valid,'recomputed_m'],comparison.loc[valid,'interpolated_position_difference_m'])
display(comparison[['flight_id','airport','nearest_time_offset_s','gap_s','position_difference_m','recomputed_m']].round(2))'''),
        ('3. Визуально сравниваем до и после выравнивания',
         'Логарифмическая шкала нужна из-за большого разброса расстояний. Выравнивание времени не является исправлением метаданных; положительное или отрицательное изменение показано только описательно.',
         '''# Ставим результаты методов рядом для каждого найденного случая
plot = comparison[comparison.recomputed_m.notna()].set_index('flight_id')
fig, ax = plt.subplots(figsize=(12,5))
x = np.arange(len(plot))
ax.scatter(x,plot.position_difference_m,label='Ближайшее время',color='#85929E',s=55)
ax.scatter(x,plot.recomputed_m,label='К времени C40',color='#2471A3',s=55)
for i,row in enumerate(plot.itertuples()):
    ax.plot([i,i],[row.position_difference_m,row.recomputed_m],color='#bbb')
ax.set(xticks=x,xticklabels=plot.index,yscale='log',ylabel='Расхождение позиций, м',title='Влияние временного выравнивания')
ax.tick_params(axis='x',rotation=45)
ax.legend()
plt.tight_layout()
plt.show()
# Не смешиваем итог по аэропортам и сохраняем число найденных случаев
display(plot.groupby('airport')[['position_difference_m','recomputed_m']].agg(['count','median','max']).round(2))''')],
        'Временное выравнивание проверено прямым пересчетом, но не объясняет все расхождения. Контрольные рейсы одного пилота не позволяют сделать вывод обо всех датах и аэропортах. Следующий этап 07 скрывает известные точки, чтобы оценить точность восстановления на проверочных ответах.')


if __name__ == '__main__':
    main()
