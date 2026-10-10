"""Собрать и выполнить погодные ноутбуки с кодом, таблицами и графиками."""
from pathlib import Path
import json
import sys
import nbformat as nbf
from nbclient import NotebookClient
from jupyter_client.manager import KernelManager

BASE = Path(__file__).resolve().parents[1]
NOTEBOOKS = BASE / 'notebooks'
NOTEBOOKS.mkdir(parents=True, exist_ok=True)

SETUP = '''from pathlib import Path
import hashlib, json, sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import display, Markdown, HTML
from html import escape
from cycler import cycler

BASE = Path.cwd().resolve()
while not (BASE / 'reports/03_pilot_manifest.json').exists():
    if BASE == BASE.parent:
        raise FileNotFoundError('Не найдена папка europe/weather с выполненным пилотом')
    BASE = BASE.parent
PROJECT = BASE.parents[1]
R, F, DATA = BASE / 'reports', BASE / 'figures', BASE / 'data'
F.mkdir(exist_ok=True)
sys.path.insert(0, str(BASE / 'scripts'))
manifest = json.loads((R / '03_pilot_manifest.json').read_text(encoding='utf8'))
def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()
assert sha(PROJECT / 'europe/data/analytical/06_05_cases_excluded.parquet') == manifest['input_sha256']
for source in manifest['downloads']:
    path = DATA / 'raw' / f"{source['airport']}_{source['month']}.csv"
    assert sha(path) == source['sha256']

GREEN, CORAL, LIGHT_GREEN, LIGHT_CORAL = '#00897B', '#D65A4A', '#80CBC4', '#E69A90'
plt.rcParams.update({'figure.dpi':120, 'font.size':11, 'axes.spines.top':False,
    'axes.spines.right':False, 'axes.grid':True, 'grid.alpha':.16,
    'axes.prop_cycle':cycler(color=[CORAL, GREEN, LIGHT_CORAL, LIGHT_GREEN])})
pd.set_option('display.max_columns', 20)
pd.set_option('display.max_rows', 30)
pd.set_option('display.max_colwidth', 100)
def save(name):
    plt.tight_layout()
    for ext in ['png', 'svg']:
        plt.savefig(F / f'{name}.{ext}', bbox_inches='tight')
    plt.show()
def readable_table(frame):
    html = frame.to_html(index=False, escape=True)
    display(HTML('<div style="max-width:100%;overflow:auto">' + html + '</div>'))
'''


def notebook_one():
    cells = []
    def md(value): cells.append(nbf.v4.new_markdown_cell(value))
    def code(value): cells.append(nbf.v4.new_code_cell(value))
    md('''# 01. Покрытие METAR на момент C40

**Задача:** проверить, есть ли предшествующая погода для обычных посадок и GoA.
Пилот охватывает EGCC, EGLL и LFPG за январь и июль 2019 года по времени C40 UTC.
Полные месяцы включают все дни; аэропорты не ранжировались по частоте GoA.

Здесь выполнены код сопоставления, таблицы покрытия и графики. Исходная база
не изменяется. Это техническая проверка шести групп, а не оценка всей базы
или доказательство доступности погоды в реальном прогнозе.

[Следующий ноутбук: погодные признаки](02_metar_decoding.ipynb) ·
[Протокол](../reports/02_pilot_protocol.md) ·
[IEM — описание архива](https://mesonet.agron.iastate.edu/info/datasets/metar.html)

## 1. Проверка происхождения

SHA256 фиксирует конкретную версию данных. Проверяем аналитический вход и
все шесть сохраненных ответов IEM; повторной загрузки при исполнении нет.''')
    code(SETUP)
    code('''audit = json.loads((R / '01_input_audit.json').read_text(encoding='utf8'))
matches = pd.read_parquet(DATA / '03_pilot_matches.parquet')
assert len(matches) == manifest['rows'] and matches.source_row.is_unique
assert int(matches.GoA.sum()) == manifest['go_arounds']
summary = pd.DataFrame([
    {'Показатель':'Исходные записи', 'Значение':audit['rows']},
    {'Показатель':'Исходные GoA', 'Значение':audit['go_arounds']},
    {'Показатель':'Записи пилота', 'Значение':len(matches)},
    {'Показатель':'GoA пилота', 'Значение':int(matches.GoA.sum())},
    {'Показатель':'Архивные METAR, включая предшествующие сутки',
     'Значение':sum(x['raw_reports'] for x in manifest['downloads'])},
])
display(summary)
source_table = pd.DataFrame(manifest['downloads'])[
    ['airport','month','raw_reports','ambiguous_times','correction_reports','downloaded_utc']]
display(source_table)
''')
    md('''## 2. Критерии выбора аэропортов и периодов наблюдения

EGCC включен как исходная станция технического пилота. EGLL и LFPG — два других аэропорта с наибольшим
по объему записей с C40 не позже первой попытки. В этой аналитической выборке
время C40 заполнено и удовлетворяет условию для всех записей.
Январь и июль 2019 года выбраны как два календарных месяца разных сезонов.
Такой выбор не обеспечивает репрезентативность погодных условий за год и не позволяет оценить сезонный эффект.

Выбор станций основан на объеме исходных записей и не использует частоту GoA.
Пилот предназначен для проверки сопоставления и декодирования METAR; его результаты не обобщаются на все аэропорты.''')
    code('''airports = pd.read_csv(R / '01_airport_coverage.csv').sort_values('c40_early', ascending=False)
top = pd.concat([airports.head(12),airports[airports.airport.isin(audit['selected_airports'])]])
top = top.drop_duplicates('airport').sort_values('c40_early')
fig, ax = plt.subplots(figsize=(9, 5.5))
colors = [CORAL if a in audit['selected_airports'] else GREEN for a in top.airport]
ax.barh(top.airport, top.c40_early / 1000, color=colors)
ax.set_xlabel('Число записей с C40 не позже первой попытки, тыс.')
ax.set_title('Распределение записей по аэропортам')
from matplotlib.patches import Patch
ax.legend(handles=[Patch(facecolor=CORAL, label='Аэропорты пилота'),
                   Patch(facecolor=GREEN, label='Другие аэропорты')],
          loc='lower right', frameon=True)
save('01_airport_selection')
display(airports[airports.airport.isin(audit['selected_airports'])][
    ['airport','rows','go_arounds','c40_early','c40_early_percent']])
group = matches.groupby(['airport','month','GoA']).size().unstack('GoA', fill_value=0)
display(group.rename(columns={False:'Обычная посадка',True:'GoA'}))
''')
    md('''## 3. Сопоставление METAR с записями на момент C40

Для каждого аэропорта выбираем последнее наблюдение **строго раньше C40**.
Основной предел — 60 минут; 30 и 90 минут — диагностические сценарии.
При различных текстах одного времени выставляем неоднозначность, не
заменяем ее более старой погодой. Одинаковые строки сворачиваем только в
рабочей таблице сопоставления; исходный ответ остается неизменным.

`source_row` сохраняет физический номер строки входного Parquet и кратность
наблюдений. Уникальность `flight_id` не предполагается.''')
    code('''recomputed = []
for source in manifest['downloads']:
    airport, month = source['airport'], source['month']
    raw = pd.read_csv(DATA / 'raw' / f'{airport}_{month}.csv', dtype=str, keep_default_na=False)
    raw['metar_utc'] = pd.to_datetime(raw.valid, utc=True).astype('datetime64[ns, UTC]')
    assert raw.station.eq(airport).all()
    raw = raw.drop_duplicates(['metar_utc','metar']).copy()
    raw['ambiguous'] = raw.duplicated('metar_utc', keep=False)
    raw = raw.drop_duplicates('metar_utc').sort_values('metar_utc')
    part = matches.loc[matches.airport.eq(airport) & matches.month.eq(month),
        ['source_row','airport','C40_CROSS_TIME','GoA']].sort_values('C40_CROSS_TIME')
    joined = pd.merge_asof(part, raw[['metar_utc','metar','ambiguous']],
        left_on='C40_CROSS_TIME', right_on='metar_utc', direction='backward',
        allow_exact_matches=False, tolerance=pd.Timedelta(minutes=90))
    joined['age_minutes'] = (joined.C40_CROSS_TIME-joined.metar_utc).dt.total_seconds()/60
    joined.loc[joined.ambiguous.eq(True),'metar'] = None
    recomputed.append(joined)
check = pd.concat(recomputed).sort_values('source_row').reset_index(drop=True)
expected = matches.sort_values('source_row').reset_index(drop=True)
fields = ['source_row','airport','C40_CROSS_TIME','GoA','metar_utc','metar','age_minutes']
pd.testing.assert_frame_equal(check[fields], expected[fields], check_dtype=False)
valid = check.metar_utc.notna()
assert check.loc[valid,'age_minutes'].gt(0).all()
assert check.loc[valid,'age_minutes'].le(90).all()
display(Markdown(f'Повторное сопоставление совпало для **{len(check):,} записей**. '
                 'Будущие наблюдения и равные C40 времена не использованы.'))
''')
    md('''## 4. Покрытие METAR для обычных посадок и GoA

Покрытие означает наличие однозначного архивного сообщения в заданном
временном окне. Оно не означает заполненность каждого погодного поля.
В каждой строке матрицы собственный знаменатель; числа GoA невелики.''')
    code('''coverage_rows = []
for (airport, month, label), part in matches.groupby(['airport','month','GoA']):
    for age in [30,60,90]:
        covered = part.metar_utc.notna() & part.age_minutes.le(age) & ~part.ambiguous.eq(True)
        coverage_rows.append(dict(airport=airport,month=month,GoA=bool(label),
            max_age_minutes=age, rows=len(part), covered=int(covered.sum()),
            coverage_percent=100*covered.mean(), not_covered=int((~covered).sum())))
coverage = pd.DataFrame(coverage_rows)
persisted = pd.read_csv(R / '03_coverage.csv')
assert coverage.covered.tolist() == persisted.covered.tolist()
table = coverage.query('max_age_minutes == 60').copy()
table['Метка'] = table.GoA.map({False:'Обычная посадка',True:'GoA'})
display(table[['airport','month','Метка','rows','covered','coverage_percent']])
coverage['Группа'] = (coverage.airport + ' / ' + coverage.month + ' / ' +
    coverage.GoA.map({False:'посадка',True:'GoA'}))
matrix = coverage.pivot(index='Группа', columns='max_age_minutes', values='coverage_percent')
fig, ax = plt.subplots(figsize=(9,6.5))
im = ax.imshow(matrix, cmap='Greens', vmin=99.5, vmax=100, aspect='auto')
ax.set_yticks(range(len(matrix)), matrix.index)
ax.set_xticks(range(3), ['≤30 мин','≤60 мин','≤90 мин'])
ax.grid(False)
for i in range(len(matrix)):
    for j in range(3):
        ax.text(j,i,f'{matrix.iloc[i,j]:.2f}%',ha='center',va='center',color='white',fontsize=10)
ax.set_title('Доля записей с предшествующим METAR')
fig.colorbar(im, ax=ax, label='Покрытие, %', shrink=.8)
save('02_weather_coverage')
''')
    md('''## 5. Давность наблюдений и чувствительность к порогу

Почти полное покрытие удобнее оценивать по числу непокрытых записей,
а не только по процентам. Порог 30 минут оставляет 53 обычные посадки без
подходящего сообщения; при 60 минутах покрыты все записи пилота.''')
    code('''age_table = matches.groupby(['airport','month']).age_minutes.agg(
    rows='size', median='median', p95=lambda x:x.quantile(.95), maximum='max')
display(age_table.round(2))
fig, axes = plt.subplots(1,2,figsize=(12,4.5))
for airport, color in zip(['EGCC','EGLL','LFPG'],[CORAL,GREEN,'#546E7A']):
    x = np.sort(matches.loc[matches.airport.eq(airport),'age_minutes'].dropna().to_numpy())
    axes[0].plot(x,np.arange(1,len(x)+1)/len(x)*100,label=airport,color=color)
axes[0].set(xlabel='Давность METAR, мин',ylabel='Накопленная доля записей, %',
    title='Распределение давности наблюдений')
axes[0].axvline(30,color='#777777',ls='--',lw=1)
axes[0].legend()
miss = coverage.query('max_age_minutes == 30').groupby(['airport','month']).not_covered.sum().unstack()
miss.plot.bar(ax=axes[1],color=[CORAL,GREEN],rot=0)
axes[1].set(xlabel='Аэропорт',ylabel='Число записей',title='Не покрыты при пределе 30 минут')
axes[1].legend(title='Месяц')
save('03_age_and_threshold')
''')
    md('''## 6. Сообщения COR и сопоставление на границе суток UTC

COR сообщает об исправлении, но архив не устанавливает, когда именно эта
редакция стала доступна. Строгое сопоставление по времени наблюдения этого
ограничения не снимает. Переход суток обрабатывается полным timestamp UTC.''')
    code('''cor = matches.groupby(['airport','month','GoA']).correction.sum().rename('Связи с COR')
display(cor.reset_index())
midnight = matches.C40_CROSS_TIME.dt.date.ne(matches.metar_utc.dt.date)
sample = matches.loc[midnight].sort_values('C40_CROSS_TIME').head(1)
readable_table(sample[['airport','C40_CROSS_TIME','metar_utc','age_minutes','metar']])
display(Markdown('**Проверено:** кратность сохранена, raw-файлы и вход не изменены. '
    '**Ограничение:** время наблюдения не равно времени доступности. '
    'Никакие записи не исключены; обучение не выполнялось.'))
''')
    return make(cells)


def notebook_two():
    cells = []
    def md(value): cells.append(nbf.v4.new_markdown_cell(value))
    def code(value): cells.append(nbf.v4.new_code_cell(value))
    md('''# 02. Декодирование METAR и качество погодных признаков

**Задача:** получить погодные поля из наблюдаемой части сообщения, сохранив
единицы, специальные коды и пропуски. Декодируем все 9 223 сообщения,
включая предшествующие сутки, и сохраняем 76 741 запись пилота.

Код, таблицы и графики выполнены. Наблюдение, тренд и примечания разделены.
Сведения о будущих попытках, итоговой ВПП и суточных экстремумах не становятся
погодными предикторами.

[Покрытие](01_metar_coverage.ipynb) ·
[Словарь полей](../reports/04_field_dictionary.md) ·
[Met Office — декодирование](https://docs.mavis.metoffice.gov.uk/guidance/metar-decode/) ·
[CAA CAP 746](https://www.caa.co.uk/publication/download/12602) ·
[python-metar](https://github.com/python-metar/python-metar)

## 1. Исходные данные и конфигурация декодирования METAR

Проверяем исходные SHA256. Версия `metar==2.0.1` зафиксирована.
Числовые поля еще не образуют утвержденный набор признаков модели.''')
    code(SETUP)
    code('''from importlib.metadata import version
from decode_metar import decode, split_sections, PARSER_VERSION
decode_manifest = json.loads((R/'04_decode_manifest.json').read_text(encoding='utf8'))
assert version('metar') == PARSER_VERSION == decode_manifest['parser_version']
assert sha(BASE/'scripts/decode_metar.py') == decode_manifest['decoder_sha256']
decoded = pd.read_parquet(DATA/'04_decoded_reports.parquet')
features = pd.read_parquet(DATA/'04_pilot_features.parquet')
assert len(decoded) == decode_manifest['raw_reports']
assert len(features) == manifest['rows'] and features.source_row.is_unique
assert int(features.GoA.sum()) == manifest['go_arounds']
display(pd.DataFrame([decode_manifest['environment']]))
display(pd.DataFrame([{
    'Сообщения':len(decoded), 'Без замечаний декодера':int(decoded.decoded_candidate.sum()),
    'С замечаниями':int((~decoded.decoded_candidate).sum()), 'Записи пилота':len(features),
    'GoA':int(features.GoA.sum()), 'Историческая доступность подтверждена':False}]))
''')
    md('''## 2. Разделение наблюдаемых условий и прогностических групп

`TEMPO`, `BECMG`, `NOSIG` и временные индикаторы тренда отделяются до разбора.
RMK также хранится отдельно. В коде ниже видно, какая часть передается в
декодер. Полный текст источника сохраняется.

При неизвестных группах после QNH мы не восстанавливаем отсутствующее слово
TEMPO и не включаем эти группы в наблюдаемые явления и облака.
Полная реализация — [decode_metar.py](../scripts/decode_metar.py).''')
    code('''example = features.loc[features.trend_text.str.contains('BECMG',na=False)].iloc[0]
observation, trend, remarks = split_sections(example.metar)
readable_table(pd.DataFrame([{'Часть':'Наблюдение','Текст':observation},
    {'Часть':'Прогноз','Текст':trend},{'Часть':'Примечания','Текст':remarks}]))
parsed_example = decode(example.metar, example.airport, example.metar_utc.to_pydatetime())
display(pd.DataFrame([parsed_example])[[
    'wind_token','wind_speed_kt','wind_gust_kt','visibility_token','visibility_relation',
    'cloud_tokens','present_weather_tokens','qnh_hpa','parser_status']])
''')
    md('''## 3. Верификация декодирования METAR и целостности сопоставления

Для всех исходных сообщений повторяем декодирование и сравниваем поля с
сохраненной таблицей. После этого проверяем присоединение `many_to_one`:
одно сообщение может использоваться несколькими записями, но ни одна
строка исходного пилота не должна размножиться или исчезнуть.''')
    code('''reparsed = []
for row in decoded.itertuples(index=False):
    reparsed.append(decode(row.metar,row.station,row.metar_utc.to_pydatetime()))
reparsed = pd.DataFrame(reparsed).sort_values('metar_id').reset_index(drop=True)
saved = decoded.sort_values('metar_id').reset_index(drop=True)
pd.testing.assert_frame_equal(reparsed, saved[reparsed.columns], check_dtype=False)
links = pd.read_parquet(DATA/'03_pilot_matches.parquet')
right = decoded.rename(columns={'station':'airport'})
keys = ['airport','metar_utc','metar']
rejoined = links[['source_row','GoA']+keys].merge(right,
    on=keys,how='left',validate='many_to_one')
assert len(rejoined) == len(links) and rejoined.source_row.is_unique
assert int(rejoined.GoA.sum()) == int(links.GoA.sum())
cols = ['metar_id','wind_speed_kt','visibility_m','qnh_hpa','parser_status']
pd.testing.assert_frame_equal(rejoined.sort_values('source_row')[cols].reset_index(drop=True),
    features.sort_values('source_row')[cols].reset_index(drop=True),check_dtype=False)
display(Markdown(f'Повторно проверены **{len(decoded):,} сообщений** и '
    f'**{len(features):,} связей**. Расхождений нет.'))
''')
    md('''## 4. Частота кодов и заполненность

В первой таблице знаменатель — число архивных сообщений; в графике
заполненности — число записей пилота. Это разные величины: сообщения
используются многократно.

Отсутствующий порыв не заменяется нулем. Отсутствие числового потолка может
означать CAVOK, отсутствие BKN/OVC, неизвестную высоту или неизвестное
количество облаков. Поэтому доля числовых значений не равна доле исправных
сообщений.''')
    code('''report_counts = decoded.groupby(['station','archive_month']).agg(
    METAR=('metar_id','size'), без_замечаний=('decoded_candidate','sum'),
    CAVOK=('cavok','sum'), VRB=('wind_variable','sum'), порывы=('gust_group_reported','sum'),
    тренд=('has_trend','sum'), COR=('correction_reported','sum'))
display(report_counts)
labels = {'wind_speed_kt':'Скорость ветра','wind_gust_kt':'Переданный порыв',
    'visibility_m':'Видимость / граница','cloud_ceiling_ft':'Числовая база BKN/OVC',
    'temperature_c':'Температура','dewpoint_c':'Точка росы','qnh_hpa':'QNH'}
presence = features.groupby(['airport','month'])[list(labels)].agg(lambda x:100*x.notna().mean())
presence.columns = [labels[x] for x in presence.columns]
display(presence.round(2))
core_missing = decoded.loc[decoded.temperature_token.isna() | decoded.qnh_token.isna(),
    ['station','metar_utc','metar','temperature_c','dewpoint_c','qnh_hpa','parser_status']]
readable_table(core_missing)
fig, ax = plt.subplots(figsize=(10.5,5.8))
matrix = presence.T
im = ax.imshow(matrix,cmap='Greens',vmin=0,vmax=100,aspect='auto')
ax.set_yticks(range(len(matrix)),matrix.index)
ax.set_xticks(range(len(matrix.columns)),[f'{a}\\n{m}' for a,m in matrix.columns])
ax.grid(False)
for i in range(len(matrix)):
    for j in range(len(matrix.columns)):
        v = matrix.iloc[i,j]
        ax.text(j,i,f'{v:.1f}%',ha='center',va='center',color='white' if v>55 else '#263238',fontsize=10)
ax.set_title('Доля записей с числовым значением поля; пропуски не заполнялись')
fig.colorbar(im,ax=ax,label='Числовое поле присутствует, %',shrink=.8)
save('04_field_presence')
''')
    md('''## 5. Наблюдаемые распределения

Показываем все записи пилота; это описание данных, без причинного вывода
о GoA. Верхняя граница измеренной видимости не восстанавливается:
9999 и CAVOK представлены отдельными кодовыми категориями.
Диапазоны — диагностика, не правила очистки.''')
    code('''display(pd.read_csv(R/'04_numeric_ranges.csv'))
fig, axes = plt.subplots(2,2,figsize=(12,8))
axes[0,0].hist(features.wind_speed_kt.dropna(),bins=np.arange(-.5,25.5,1),color=GREEN,edgecolor='white')
axes[0,0].set(xlabel='Скорость ветра, kt',ylabel='Записи',title='Ветер: наблюдаемая средняя скорость')
v = pd.Series(np.select([features.cavok.eq(True),features.visibility_token.eq('9999'),
    features.visibility_relation.eq('lt'),features.visibility_relation.eq('eq')],
    ['CAVOK: ≥10 км','9999: ≥10 км','0000: <50 м','Числовой код'],default='Пропуск / другое')).value_counts()
axes[0,1].barh(v.index[::-1],v.values[::-1],color=[GREEN,CORAL,LIGHT_GREEN,LIGHT_CORAL][:len(v)])
axes[0,1].set(xlabel='Записи',title='Видимость: исходные способы кодирования')
for month,color in [('2019-01',CORAL),('2019-07',GREEN)]:
    axes[1,0].hist(features.loc[features.month.eq(month),'temperature_c'].dropna(),
        bins=np.arange(-6,44,2),histtype='step',lw=2,color=color,label=month)
axes[1,0].set(xlabel='Температура, °C',ylabel='Записи',title='Два сезона пилота')
axes[1,0].legend()
axes[1,1].hist(features.qnh_hpa.dropna(),bins=np.arange(982,1048,2),color=GREEN,edgecolor='white')
axes[1,1].set(xlabel='QNH, hPa',ylabel='Записи',title='Давление из Q/A-группы; без QFE и SLP')
save('05_weather_distributions')
''')
    md('''## 6. Интерпретация облачных групп и кодов отсутствующих значений

`//////CB` и `//////TCU` передают тип облака при неизвестных количестве и
высоте (CAP 746, 4.149–4.150). В `SCT020///` известны количество и база,
а тип неизвестен (4.147). Эти коды сохранены.

`BKN000` означает базу ниже 100 ft (4.109): число 100 используется только
вместе с отношением `lt`. Для остальных баз сохраняется переданная ступень
кода, а не более точное измерение. Все высоты здесь относительно аэродрома.
VV хранится отдельно от базы BKN/OVC; CAVOK не превращается в бесконечный
потолок.''')
    code('''clouds = features.groupby(['airport','cloud_state']).size().unstack(fill_value=0)
display(clouds)
unknown = features.groupby('airport').agg(
    неизвестен_тип=('cloud_type_unavailable','sum'),
    неизвестно_количество=('cloud_amount_unavailable','sum'),
    передана_VV=('vertical_visibility_reported','sum'))
display(unknown)
fig, ax = plt.subplots(figsize=(9.5,4.5))
clouds.div(clouds.sum(axis=1),axis=0).mul(100).plot.barh(stacked=True,ax=ax,
    color=[GREEN,CORAL,LIGHT_GREEN,LIGHT_CORAL,'#90A4AE','#546E7A'],width=.6)
ax.set(xlabel='Доля записей, %',ylabel='Аэропорт',title='Способы кодирования облачности')
ax.legend(title='Код / группа',bbox_to_anchor=(1.02,1),loc='upper left')
save('06_cloud_coding')
''')
    md('''## 7. Примеры преобразования сообщений METAR в погодные признаки

Для каждой категории выбран первый по C40 пример, без отбора по исходу.
Возраст показывает сопоставление во времени, а не подтвержденную задержку
передачи. Исходная строка и прогноз показаны рядом с декодированными полями.''')
    code('''examples = pd.read_csv(R/'04_examples.csv',keep_default_na=False)
names = {'COR':'Исправление COR','CAVOK':'CAVOK','VRB':'Переменный ветер VRB',
    'gust':'Переданный порыв','direction_variation':'Диапазон направления',
    'trend_TEMPO':'Тренд TEMPO','trend_BECMG':'Тренд BECMG',
    'midnight_transition':'Переход суток UTC','reduced_visibility':'Сниженная видимость',
    'vertical_visibility':'Вертикальная видимость','recent_weather':'Недавнее явление RE',
    'automatic_cloud_unknowns':'Автоматическая группа с неизвестными количеством / высотой',
    'cloud_below_100ft':'База облака ниже 100 ft'}
for row in examples.itertuples(index=False):
    display(Markdown(f'### {names[row.category]} — {row.airport}'))
    display(HTML('<pre style="white-space:pre-wrap;overflow-wrap:anywhere;padding:10px">'+escape(row.metar)+'</pre>'))
    values = [('C40 UTC',row.c40_utc),('METAR UTC',row.metar_utc),('Давность, мин',round(row.age_minutes,2)),
        ('Наблюдаемая часть',row.observed_text),('Прогноз',row.trend_text or 'Не передан'),
        ('Ветер',row.wind_token),('Скорость / переданный порыв, kt',f'{row.wind_speed_kt} / {row.wind_gust_kt}'),
        ('Видимость: код / значение / отношение',f'{row.visibility_token} / {row.visibility_m} / {row.visibility_relation}'),
        ('Облака',row.cloud_tokens),('База BKN/OVC: значение / отношение',f'{row.cloud_ceiling_ft} / {row.cloud_ceiling_relation}'),
        ('Наблюдаемые явления',row.present_weather_tokens or 'Группа не передана'),
        ('Недавние явления',row.recent_weather_tokens or 'Группа не передана'),('QNH, hPa',row.qnh_hpa)]
    readable_table(pd.DataFrame(values,columns=['Поле','Значение']))
''')
    md('''## 8. Классификация сообщений с замечаниями и связанных записей

После поддержки документированных автоматических облачных кодов осталось
10 сообщений с замечаниями, затрагивающих 81 запись и 3 GoA.
Ни одна запись не удалена. Флаг `weather_candidate_60min` — осторожный
диагностический сценарий, не принятое исключение из основной базы.

Разные причины не смешиваем: неподдержанная группа состояния ВПП не
доказывает ошибочность основных погодных значений; пропущенный индикатор
тренда не восстанавливаем по догадке. Текст из источника остается исходным.''')
    code('''residuals = pd.read_csv(R/'04_decode_residuals.csv',keep_default_na=False)
reason_names = {'unsupported_runway_state_group':'Не поддержана группа состояния ВПП',
    'unlabelled_weather_after_qnh':'Погодные группы после QNH без индикатора тренда',
    'unclassified_five_slashes':'Не классифицирована группа /////',
    'temperature_and_qnh_absent':'Отсутствуют температура и QNH'}
residuals['Причина'] = residuals.audit_category.map(reason_names)
residual_summary = residuals.groupby('Причина').agg(
    Сообщения=('metar_id','size'), Записи=('matched_rows','sum'), GoA=('matched_go_arounds','sum'))
display(residual_summary)
readable_table(residuals[['station','metar_utc','Причина','matched_rows','matched_go_arounds']])
for row in residuals.itertuples(index=False):
    display(Markdown(f'**{row.station}, {row.metar_utc}** — {reason_names[row.audit_category]}; '
                     f'записи: {row.matched_rows}, GoA: {row.matched_go_arounds}.'))
    display(HTML('<pre style="white-space:pre-wrap;overflow-wrap:anywhere;padding:8px">'+escape(row.metar)+'</pre>'))
''')
    md('''## 9. Диагностические сценарии и проверки семантики

Сравниваем число записей с сообщением, число записей без замечаний декодера
и более осторожный вариант без COR. Это не новое формирование обучающей
выборки. Даже вариант без COR не доказывает историческую доступность.

Тесты проверяют отделение прогнозных групп, цензурированную видимость,
неизвестные облака, порывы, отрицательные температуры, давление и сохранение
неподдержанных групп как замечаний.''')
    code('''scenario_rows = []
for name,mask in [('Есть METAR до 60 минут',features.status.eq('matched')),
    ('Без замечаний декодера',features.weather_candidate_60min),
    ('Без замечаний и без COR',features.weather_candidate_60min_without_cor)]:
    scenario_rows.append({'Сценарий':name,'Записи':int(mask.sum()),
        'GoA':int(features.loc[mask,'GoA'].sum()),'Доля записей, %':100*mask.mean()})
scenarios = pd.DataFrame(scenario_rows)
display(scenarios.round(3))
fig, axes = plt.subplots(1,2,figsize=(12,4.5))
axes[0].barh(scenarios['Сценарий'][::-1],scenarios['Записи'][::-1],color=[CORAL,LIGHT_GREEN,GREEN])
axes[0].set(xlabel='Записи',title='Погодные диагностические сценарии')
for i,n in enumerate(scenarios['Записи'][::-1]):
    axes[0].text(n,i,f' {n:,}',va='center',fontsize=10)
axes[0].set_xlim(0,len(features)*1.16)
from matplotlib.ticker import FuncFormatter
axes[0].xaxis.set_major_formatter(FuncFormatter(lambda x,pos:f'{x/1000:.0f}'))
axes[0].set_xlabel('Записи, тыс.')
axes[1].barh(scenarios['Сценарий'][::-1],scenarios['GoA'][::-1],color=[CORAL,LIGHT_GREEN,GREEN])
axes[1].set(xlabel='GoA',title='Метка сохранена; модель не обучалась')
axes[1].set_yticklabels([])
for i,n in enumerate(scenarios['GoA'][::-1]):
    axes[1].text(n,i,f' {n}',va='center',fontsize=10)
axes[1].set_xlim(0,manifest['go_arounds']*1.16)
save('07_decode_scenarios')
''')
    code('''import unittest
import test_decode_metar
suite = unittest.defaultTestLoader.loadTestsFromModule(test_decode_metar)
outcome = unittest.TextTestRunner(verbosity=2,stream=sys.stdout).run(suite)
assert outcome.wasSuccessful()
display(Markdown('**Итог:** погодные поля и остаточные замечания сохранены; '
    'строки и исходные значения не удалялись. Историческая доступность '
    'сообщений и целевого аэропорта остается ограничением. Обучение и '
    'масштабирование на всю базу еще не выполнялись.'))
''')
    return make(cells)


def make(cells):
    return nbf.v4.new_notebook(cells=cells, metadata={
        'kernelspec':{'display_name':'Python (Approach Risk)', 'language':'python','name':'python3'},
        'language_info':{'name':'python'}})


def main():
    results = []
    for name, builder in [('01_metar_coverage.ipynb',notebook_one),
                          ('02_metar_decoding.ipynb',notebook_two)]:
        path = NOTEBOOKS / name
        notebook = builder()
        nbf.write(notebook,path)
        manager = KernelManager(kernel_name='python3')
        manager.kernel_spec.argv = [sys.executable,'-m','ipykernel_launcher','-f','{connection_file}']
        client = NotebookClient(notebook,km=manager,timeout=300,
            resources={'metadata':{'path':str(NOTEBOOKS)}},allow_errors=False)
        client.on_cell_start = lambda cell,cell_index,**kwargs: print(
            f'{name}: cell {cell_index+1}/{len(notebook.cells)}',flush=True)
        try:
            client.execute()
        finally:
            nbf.write(notebook,path)
        nbf.validate(notebook)
        code_cells = [c for c in notebook.cells if c.cell_type=='code']
        assert all(c.execution_count is not None for c in code_cells)
        assert not any(o.output_type=='error' for c in code_cells for o in c.outputs)
        images = sum('image/png' in o.get('data',{}) for c in code_cells for o in c.outputs)
        tables = sum('text/html' in o.get('data',{}) for c in code_cells for o in c.outputs)
        assert images >= 3 and tables >= 4
        results.append(dict(notebook=name,code_cells=len(code_cells),executed=True,
                            embedded_figures=images,html_outputs=tables))
    (BASE / 'reports/05_notebook_verification.json').write_text(
        json.dumps(results,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps(results,ensure_ascii=False,indent=2),flush=True)


if __name__ == '__main__':
    main()
