"""07_06: распределения C40 обеих меток, геометрия и диагностические записи."""
import json
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import nbformat
from audit_airports import ROOT, INPUT
from audit_feature_availability import digest

REPORT=ROOT/'europe/reports/airports/distributions'
FIGURES=ROOT/'europe/figures/07_airports'
NUMBERS=['C40_BEARING','C40_CROSS_LAT','C40_CROSS_LON']


def table_md(data):
    return '| '+' | '.join(map(str,data.columns))+' |\n| '+' | '.join(['---']*len(data.columns))+' |\n'+''.join('| '+' | '.join(map(str,row))+' |\n' for row in data.itertuples(index=False,name=None))


def save(fig,name):
    for ext in ['png','svg','pdf']:fig.savefig(FIGURES/f'{name}.{ext}',dpi=300,bbox_inches='tight')
    plt.close(fig)


def main():
    REPORT.mkdir(parents=True,exist_ok=True)
    expected=json.loads((ROOT/'europe/reports/airports/07_01_audit.json').read_text(encoding='utf8'))
    before=digest(INPUT);assert before==expected['input_sha256']
    columns=['airport','GoA','C40_CROSS_TIME',*NUMBERS]
    frame=pq.read_table(INPUT,columns=columns).to_pandas(ignore_metadata=True)
    assert len(frame)==expected['rows'] and int(frame.GoA.sum())==expected['go_arounds']
    frame['month_c40']=frame.C40_CROSS_TIME.dt.strftime('%Y-%m')
    invalid=~np.isfinite(frame[NUMBERS]).all(axis=1)|~frame.C40_BEARING.between(0,360,inclusive='left')|~frame.C40_CROSS_LAT.between(-90,90)|~frame.C40_CROSS_LON.between(-180,180)
    valid=frame[~invalid].copy()
    # Общий центр аэропорта вычисляется без использования метки; это не координата аэродрома.
    centers=valid.groupby('airport')[['C40_CROSS_LAT','C40_CROSS_LON']].median().rename(columns={'C40_CROSS_LAT':'center_lat','C40_CROSS_LON':'center_lon'})
    valid=valid.join(centers,on='airport',validate='many_to_one')
    valid['lat_offset_deg']=valid.C40_CROSS_LAT-valid.center_lat
    valid['lon_offset_deg']=(valid.C40_CROSS_LON-valid.center_lon+180)%360-180
    lat1=np.radians(valid.C40_CROSS_LAT);lat2=np.radians(valid.center_lat)
    a=np.sin((lat1-lat2)/2)**2+np.cos(lat1)*np.cos(lat2)*np.sin(np.radians(valid.lon_offset_deg)/2)**2
    valid['distance_to_sample_center_km']=2*6371.0088*np.arcsin(np.sqrt(a.clip(0,1)))
    valid['bearing_sin']=np.sin(np.radians(valid.C40_BEARING));valid['bearing_cos']=np.cos(np.radians(valid.C40_BEARING))
    fields=NUMBERS+['lat_offset_deg','lon_offset_deg','distance_to_sample_center_km']
    stats=[]
    for (airport,label),group in valid.groupby(['airport','GoA']):
        for field in fields:
            v=group[field];q=v.quantile([.01,.05,.25,.5,.75,.95,.99])
            stats.append(dict(airport=airport,GoA=bool(label),field=field,rows=len(v),min=v.min(),max=v.max(),mean=v.mean(),std=v.std(),**{f'q{int(p*100):02d}':value for p,value in q.items()}))
    stats=pd.DataFrame(stats)
    circular=valid.groupby(['airport','GoA']).agg(rows=('GoA','size'),mean_sin=('bearing_sin','mean'),mean_cos=('bearing_cos','mean')).reset_index()
    circular['mean_direction_deg']=np.degrees(np.arctan2(circular.mean_sin,circular.mean_cos))%360
    circular['resultant_length']=np.hypot(circular.mean_sin,circular.mean_cos)
    monthly=valid.groupby(['airport','GoA','month_c40']).agg(rows=('GoA','size'),mean_sin=('bearing_sin','mean'),mean_cos=('bearing_cos','mean'),median_lat=('C40_CROSS_LAT','median'),median_lon=('C40_CROSS_LON','median'),median_distance_km=('distance_to_sample_center_km','median')).reset_index()
    monthly['mean_direction_deg']=np.degrees(np.arctan2(monthly.mean_sin,monthly.mean_cos))%360
    monthly['resultant_length']=np.hypot(monthly.mean_sin,monthly.mean_cos)
    bins=[]
    for (airport,label),group in valid.groupby(['airport','GoA']):
        count,edges=np.histogram(group.C40_BEARING,bins=np.arange(0,361,15))
        for index,n in enumerate(count):bins.append(dict(airport=airport,GoA=bool(label),lower_deg=edges[index],upper_deg=edges[index+1],rows=int(n),within_label_percent=100*n/len(group)))
    bins=pd.DataFrame(bins)
    # Округление служит только диагностике профилей; исходные значения сохраняются.
    profiles=valid[['airport','GoA',*NUMBERS]].copy()
    profiles['bearing_bin']=np.floor(profiles.C40_BEARING).astype(int)
    profiles['lat_bin']=profiles.C40_CROSS_LAT.round(2)
    profiles['lon_bin']=profiles.C40_CROSS_LON.round(2)
    key=['airport','bearing_bin','lat_bin','lon_bin']
    counts=profiles.groupby(key).GoA.agg(rows='size',go_arounds='sum').reset_index()
    counts['ga_percent']=100*counts.go_arounds/counts.rows
    top=counts.sort_values('rows',ascending=False).groupby('airport',sort=False).head(3)
    # Проверка точных совпадений только внутри выбранных округленных групп.
    selected=profiles.merge(top[key],on=key,how='inner',validate='many_to_one')
    exact=selected.groupby(['airport',*NUMBERS]).GoA.agg(rows='size',go_arounds='sum').reset_index()
    exact=exact[exact.rows.gt(1)].sort_values('rows',ascending=False)
    far=valid.sort_values('distance_to_sample_center_km',ascending=False).groupby(['airport','GoA'],sort=False).head(3)
    candidate_indices=set(far.index)|set(frame[invalid].index)
    candidate=[];profile_records=[];position=0
    for batch in pq.ParquetFile(INPUT).iter_batches(columns=['flight_id','icao24','airport','GoA','landing_time','C40_CROSS_TIME',*NUMBERS],batch_size=100000):
        chunk=batch.to_pandas(ignore_metadata=True);chunk.index=np.arange(position,position+len(chunk));position+=len(chunk)
        part=chunk.loc[chunk.index.isin(candidate_indices)].copy()
        if len(part):
            part['review_reason']=np.where(part.index.isin(frame[invalid].index),'out_of_diagnostic_range','largest_distance_to_sample_center')
            part=part.join(valid[['distance_to_sample_center_km']],how='left');candidate.append(part)
        chunk['bearing_bin']=np.floor(chunk.C40_BEARING).astype('Int64');chunk['lat_bin']=chunk.C40_CROSS_LAT.round(2);chunk['lon_bin']=chunk.C40_CROSS_LON.round(2)
        hit=chunk.merge(top[key],on=key,how='inner')
        if len(hit):profile_records.append(hit)
    candidates=pd.concat(candidate,ignore_index=True)
    profile_records=pd.concat(profile_records,ignore_index=True)
    for data,name in [(stats,'summary'),(circular,'circular'),(monthly,'monthly'),(bins,'bearing_histogram'),(centers.reset_index(),'sample_centers'),(top,'frequent_profiles'),(exact,'exact_profiles_selected_groups'),(candidates,'review_records'),(profile_records,'profile_records')]:
        data.to_csv(REPORT/f'07_06_{name}.csv',index=False)
    plt.rcParams.update({'font.family':'DejaVu Sans','svg.fonttype':'none','pdf.fonttype':42})
    order=frame.groupby('airport').size().sort_values(ascending=False).index.tolist()
    fig,axes=plt.subplots(5,4,figsize=(16,15),sharex=True)
    for ax,airport in zip(axes.flat,order):
        for label,color in [(False,'#537c99'),(True,'#b45335')]:
            d=bins[bins.airport.eq(airport)&bins.GoA.eq(label)]
            ax.step(d.lower_deg+7.5,d.within_label_percent,where='mid',color=color,label=str(label))
        ax.set_title(airport);ax.set(xlim=(0,360),xticks=[0,90,180,270,360],ylabel='Доля метки, %');ax.grid(alpha=.15)
    for ax in list(axes.flat)[len(order):]:ax.set_visible(False)
    axes.flat[0].legend(title='GoA');fig.suptitle('Направление C40: сравнение внутри каждого аэропорта');fig.tight_layout(rect=(0,.035,1,.97));fig.text(.01,.01,'Интервалы 15°. Шкалы Y различаются. 0° и 360° – соседние направления; прямую среднюю углов не интерпретируем.',fontsize=9)
    save(fig,'07_06_bearing_by_airport')
    fig,axes=plt.subplots(1,2,figsize=(13,10),sharey=True)
    for ax,field,title in zip(axes,['lat_offset_deg','lon_offset_deg'],['Широта относительно медианы аэропорта','Долгота относительно медианы аэропорта']):
        for label,color,shift in [(False,'#537c99',-.15),(True,'#b45335',.15)]:
            d=stats[stats.field.eq(field)&stats.GoA.eq(label)].set_index('airport').loc[order]
            ax.errorbar(d.q50,np.arange(len(order))+shift,xerr=np.vstack([d.q50-d.q05,d.q95-d.q50]),fmt='o',color=color,label=str(label),capsize=2)
        ax.set_yticks(range(len(order)),order);ax.set(title=title,xlabel='Отклонение, градусы');ax.axvline(0,color='#8997a1',linestyle='--');ax.grid(alpha=.15)
    axes[0].invert_yaxis();axes[0].legend(title='GoA');fig.suptitle('C40: медиана и диапазон 5–95% внутри каждой метки');fig.tight_layout(rect=(0,.04,1,.96));fig.text(.01,.01,'Центр – общая медиана выборки аэропорта, не официальная опорная точка. Полосы – квантили, не доверительные интервалы.',fontsize=9)
    save(fig,'07_06_coordinates_by_airport')
    fig,axes=plt.subplots(5,4,figsize=(16,15),sharex=True)
    months=pd.date_range('2019-01-01','2023-05-01',freq='MS').strftime('%Y-%m')
    for ax,airport in zip(axes.flat,order):
        for label,color in [(False,'#537c99'),(True,'#b45335')]:
            d=monthly[monthly.airport.eq(airport)&monthly.GoA.eq(label)].set_index('month_c40').reindex(months)
            ax.plot(np.arange(len(months)),d.median_distance_km,color=color,label=str(label),linewidth=1)
        ax.set_title(airport);ax.set_ylabel('Медиана, км');ax.set_xticks([0,12,24,36,48],['2019','2020','2021','2022','2023']);ax.grid(alpha=.15)
    for ax in list(axes.flat)[len(order):]:ax.set_visible(False)
    axes.flat[0].legend(title='GoA');fig.suptitle('C40: месячная геометрия по обеим меткам');fig.tight_layout(rect=(0,.04,1,.96));fig.text(.01,.01,'Расстояние до медианного центра наблюдений, не радиус ASMA. Месяц задан C40 UTC; разрыв означает отсутствие группы.',fontsize=9)
    save(fig,'07_06_monthly_geometry')
    audit=dict(input_sha256=before,rows=len(frame),go_arounds=int(frame.GoA.sum()),airports=len(order),diagnostic_out_of_range=int(invalid.sum()),selected_exact_repeat_groups=len(exact),selected_exact_repeat_rows=int(exact.rows.sum()),review_records=len(candidates),profile_records=len(profile_records),rows_deleted=0,values_changed=0,feature_admission_finalized=False,rounded_profiles_not_duplicates=True)
    text=f'''# 07_06 – Распределения направления и координат C40

Проверены {len(frame):,} записей и обе метки по {len(order)} аэропортам. Диагностические нарушения диапазонов или нечисловые значения: **{int(invalid.sum())}**. Все исходные значения сохранены.

## Направление

![Направление по аэропортам](../../../figures/07_airports/07_06_bearing_by_airport.png)

Направление циклическое: 359° и 1° близки. Гистограммы нормированы отдельно внутри меток. В таблицах есть круговая средняя и длина результирующего вектора от 0 до 1; малое значение означает слабую концентрацию вокруг средней и может возникать при нескольких противоположных пиках. Среднее не заменяет просмотр полного распределения. Числовые квантили исходного угла сохранены только как описание значений на шкале [0,360), не как циклический интервал.

## Координаты

![Координаты внутри аэропортов](../../../figures/07_airports/07_06_coordinates_by_airport.png)

Глобальное распределение координат смешивает географию разных аэропортов. Поэтому сравнение выполнено внутри каждого аэропорта и относительно общей медианы его наблюдений, без использования метки при выборе центра. Этот центр вычислен для описательного аудита и не является подготовленным признаком модели: если такой признак потребуется, его параметры оцениваются на обучающей части.

## Временная динамика

![Месячная геометрия](../../../figures/07_airports/07_06_monthly_geometry.png)

Группировка по месяцу C40 UTC. Полные n каждой метки, координатные медианы и круговые показатели направления сохранены в `07_06_monthly.csv`. Месяцы малого положительного класса могут давать неустойчивые показатели. Плавность или скачок линии не подтверждают изменение авиационного риска.

## Повторяющиеся профили и конкретные записи

Округленная диагностическая группа: аэропорт, целый градус направления (округление вниз), широта и долгота до 0,01°. Наиболее частые три группы каждого аэропорта сохранены в `07_06_frequent_profiles.csv`; исходные записи – в `07_06_profile_records.csv`. Округление не меняет вход, близость профилей не доказывает дубликат: одна геометрия может повторяться у обычных заходов.

В выбранных группах точных совпадений трех числовых полей с повтором: {len(exact):,} групп, {int(exact.rows.sum()):,} записей. Это ограниченная проверка выбранных групп, не глобальный поиск содержательных дублей. Идентификаторы и времена сохранены для проверки разных бортов и периодов.

В `07_06_review_records.csv` сохранены {len(candidates)} конкретных записей: три наиболее удаленные от медианного центра внутри каждой группы аэропорт × метка и все нарушения диапазонов. Расстояние вычислено по сфере радиуса 6371,0088 км; это диагностический порядок просмотра, не нормативный порог и не расстояние до официального центра аэродрома.

## Решение и следующий разбор

Новых исключений и исправлений нет. Выборка сохраняется по решениям 07_04 и 07_05. Отличие распределений не устанавливает причинную связь и не подтверждает пригодность поля для прогноза. Следующий разбор – конкретные повторяющиеся и наиболее удаленные профили, с проверкой времени, бортов и исходных значений. Погодные признаки для всей выборки еще не присоединены. Окончательный допуск признаков и временное разбиение выполняются после аудита.
'''
    (REPORT/'07_06_review.md').write_text(text,encoding='utf8')
    code="""from pathlib import Path
import pandas as pd
root = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p/'europe/reports/airports/distributions/07_06_summary.csv').exists())
report = root/'europe/reports/airports/distributions'
airport = 'EGCC'
summary = pd.read_csv(report/'07_06_summary.csv')
summary.loc[summary.airport.eq(airport)]
"""
    records="""# Конкретные записи для проверки выбранного аэропорта.
records = pd.read_csv(report/'07_06_review_records.csv', dtype={'icao24': str})
records.loc[records.airport.eq(airport)]
"""
    nb=nbformat.v4.new_notebook(cells=[nbformat.v4.new_markdown_cell(text.replace('../../../figures/','../../figures/')),nbformat.v4.new_code_cell(code),nbformat.v4.new_code_cell(records),nbformat.v4.new_markdown_cell('Воспроизведение: `.\\.venv\\Scripts\\python.exe -u -X utf8 europe/scripts/audit_c40_distributions.py`. [Отчет и полные таблицы](../../reports/airports/distributions/07_06_review.md).')],metadata={'kernelspec':{'name':'python3','display_name':'Python 3','language':'python'}})
    nbformat.validate(nb);nbformat.write(nb,ROOT/'europe/notebooks/07_airports/07_06_c40_distributions.ipynb')
    assert digest(INPUT)==before;audit['input_unchanged']=True
    (REPORT/'07_06_audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    path=ROOT/'europe/notebooks/07_airports/README.md';note='\n\n- [07_06 – Распределения направления и координат C40](07_06_c40_distributions.ipynb): обе метки, аэропорты, месяцы и конкретные профили для проверки.\n';current=path.read_text(encoding='utf8')
    if note.strip() not in current:path.write_text(current+note,encoding='utf8')
    print(json.dumps(audit,ensure_ascii=False,indent=2),flush=True)
    print(top.sort_values('rows',ascending=False).head(10).to_string(index=False),flush=True)
    print(candidates.sort_values('distance_to_sample_center_km',ascending=False).head(10).to_string(index=False),flush=True)


if __name__=='__main__':main()
