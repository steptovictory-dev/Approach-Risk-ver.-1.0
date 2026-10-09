"""07_08: радиальные отклонения C40 и самые частые точные точки."""
import json
from datetime import datetime, timezone
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import nbformat
from audit_airports import ROOT, INPUT, SOURCE
from audit_feature_availability import digest
from investigate_c40_geometry import distance

REPORT=ROOT/'europe/reports/airports/c40_geometry'
FIGURES=ROOT/'europe/figures/07_airports'
RADIUS=74.08
KEY=['icao24','airport','landing_time']


def ellipsoid_distance(lat1,lon1,lat2,lon2,inv_f=298.257223563):
    """Обратная формула Vincenty; векторизована, с контролем сходимости."""
    a=6378137.;f=1/inv_f;b=a*(1-f)
    u1=np.arctan((1-f)*np.tan(np.radians(lat1)))
    u2=np.arctan((1-f)*np.tan(np.radians(lat2)))
    s1,c1=np.sin(u1),np.cos(u1);s2,c2=np.sin(u2),np.cos(u2)
    length=np.radians((np.asarray(lon2)-np.asarray(lon1)+180)%360-180)
    lam=length.copy()
    for _ in range(100):
        sin_lam,cos_lam=np.sin(lam),np.cos(lam)
        sin_sigma=np.sqrt((c2*sin_lam)**2+(c1*s2-s1*c2*cos_lam)**2)
        cos_sigma=s1*s2+c1*c2*cos_lam
        sigma=np.arctan2(sin_sigma,cos_sigma)
        sin_alpha=np.divide(c1*c2*sin_lam,sin_sigma,out=np.zeros_like(sin_sigma),where=sin_sigma>0)
        cos2_alpha=1-sin_alpha**2
        cos2_sigma=cos_sigma-np.divide(2*s1*s2,cos2_alpha,out=np.zeros_like(cos2_alpha),where=cos2_alpha>1e-15)
        coef=f/16*cos2_alpha*(4+f*(4-3*cos2_alpha))
        updated=length+(1-coef)*f*sin_alpha*(sigma+coef*sin_sigma*(cos2_sigma+coef*cos_sigma*(-1+2*cos2_sigma**2)))
        if np.max(np.abs(updated-lam))<1e-12:
            lam=updated;break
        lam=updated
    else:raise ValueError('Vincenty не сошелся; результат не сохраняется')
    # Финальная геометрия после обновления lambda.
    sl,cl=np.sin(lam),np.cos(lam)
    ss=np.sqrt((c2*sl)**2+(c1*s2-s1*c2*cl)**2);cs=s1*s2+c1*c2*cl;sig=np.arctan2(ss,cs)
    sa=np.divide(c1*c2*sl,ss,out=np.zeros_like(ss),where=ss>0);ca=1-sa**2
    c2s=cs-np.divide(2*s1*s2,ca,out=np.zeros_like(ca),where=ca>1e-15)
    u2coef=ca*(a*a-b*b)/(b*b)
    aa=1+u2coef/16384*(4096+u2coef*(-768+u2coef*(320-175*u2coef)))
    bb=u2coef/1024*(256+u2coef*(-128+u2coef*(74-47*u2coef)))
    delta=bb*ss*(c2s+bb/4*(cs*(-1+2*c2s**2)-bb/6*c2s*(-3+4*ss**2)*(-3+4*c2s**2)))
    return b*aa*(sig-delta)/1000


def validate_distance():
    # Официальный пример NGS (GRS80) и пример GeographicLib (WGS84).
    lat1=34+12.12345/3600;lon1=-(111+12.12345/3600)
    lat2=33+22/60+11.54321/3600;lon2=-(112+55/60+44.33333/3600)
    ngs=float(ellipsoid_distance(lat1,lon1,np.array([lat2]),np.array([lon2]),inv_f=298.25722210088)[0]*1000)
    jfk=float(ellipsoid_distance(40.6,-73.8,np.array([51.6]),np.array([-.5]))[0]*1000)
    assert abs(ngs-191872.1190)<.002
    assert abs(jfk-5551759.400318678)<.002
    assert ellipsoid_distance(51.,0.,np.array([51.]),np.array([0.]))[0]==0
    return dict(ngs_reference_m=191872.1190,ngs_computed_m=ngs,geographiclib_reference_m=5551759.400318678,geographiclib_computed_m=jfk,tolerance_m=.002)


def markdown(table):
    return '| '+' | '.join(map(str,table.columns))+' |\n| '+' | '.join(['---']*len(table.columns))+' |\n'+''.join('| '+' | '.join(map(str,row))+' |\n' for row in table.itertuples(index=False,name=None))


def main():
    checks=validate_distance();print(json.dumps(checks),flush=True)
    expected=json.loads((REPORT/'07_07_audit.json').read_text(encoding='utf8'))
    before=digest(INPUT);source_before=digest(SOURCE)
    overview=json.loads((ROOT/'europe/reports/airports/07_01_audit.json').read_text(encoding='utf8'))
    assert before==expected['input_sha256'] and source_before==overview['source_sha256']
    centers=pd.read_csv(REPORT/'07_07_best_formula.csv').set_index('airport')
    fields=['flight_id','icao24','airport','GoA','landing_time','C40_CROSS_TIME','C40_BEARING','C40_CROSS_LAT','C40_CROSS_LON','typecode','n_attempts','attempt_times']
    frame=pq.read_table(INPUT,columns=fields).to_pandas(ignore_metadata=True)
    frame=frame.join(centers[['fit_lat','fit_lon']],on='airport',validate='many_to_one')
    frame['radius_sphere_km']=distance(frame.fit_lat.to_numpy(),frame.fit_lon.to_numpy(),frame.C40_CROSS_LAT.to_numpy(),frame.C40_CROSS_LON.to_numpy())
    frame['radius_wgs84_km']=ellipsoid_distance(frame.fit_lat.to_numpy(),frame.fit_lon.to_numpy(),frame.C40_CROSS_LAT.to_numpy(),frame.C40_CROSS_LON.to_numpy())
    frame['deviation_sphere_km']=frame.radius_sphere_km-RADIUS
    frame['deviation_wgs84_km']=frame.radius_wgs84_km-RADIUS
    frame['abs_deviation_km']=frame.deviation_wgs84_km.abs()
    frame['sphere_to_wgs84_m']=1000*(frame.radius_wgs84_km-frame.radius_sphere_km)
    frame['month_c40']=frame.C40_CROSS_TIME.dt.strftime('%Y-%m')
    frame['tail_1km']=frame.abs_deviation_km.gt(1)
    frame['tail_3km']=frame.abs_deviation_km.gt(3)
    stats=[]
    for (airport,label),group in frame.groupby(['airport','GoA']):
        q=group.radius_wgs84_km.quantile([.01,.5,.99])
        stats.append(dict(airport=airport,GoA=bool(label),rows=len(group),min_radius_km=group.radius_wgs84_km.min(),q01_radius_km=q.loc[.01],median_radius_km=q.loc[.5],q99_radius_km=q.loc[.99],max_radius_km=group.radius_wgs84_km.max(),tail_1km=int(group.tail_1km.sum()),tail_3km=int(group.tail_3km.sum()),tail_1km_percent=100*group.tail_1km.mean()))
    stats=pd.DataFrame(stats)
    monthly=frame.groupby(['airport','GoA','month_c40']).agg(rows=('GoA','size'),tail_1km=('tail_1km','sum'),tail_3km=('tail_3km','sum')).reset_index()
    monthly['tail_1km_percent']=100*monthly.tail_1km/monthly.rows
    tails=frame[frame.tail_1km].copy()
    tails['first_attempt']=tails.attempt_times.map(lambda x:pd.to_datetime(x[0],utc=True))
    tails['minutes_c40_to_first_attempt']=(tails.first_attempt-tails.C40_CROSS_TIME).dt.total_seconds()/60
    tails['minutes_c40_to_landing']=(tails.landing_time-tails.C40_CROSS_TIME).dt.total_seconds()/60
    tails['attempt_times_json']=tails.attempt_times.map(lambda x:json.dumps([str(t) for t in x]))
    assert tails.minutes_c40_to_first_attempt.gt(0).all()
    assert not tails.duplicated(KEY).any()
    # Проверяем неизменность аномальных значений относительно исходного Parquet.
    source_hits=[]
    raw_fields=[*KEY,'GoA','C40_CROSS_TIME','C40_BEARING','C40_CROSS_LAT','C40_CROSS_LON']
    keys=tails[KEY]
    for batch in pq.ParquetFile(SOURCE).iter_batches(columns=raw_fields,batch_size=100000):
        chunk=batch.to_pandas(ignore_metadata=True)
        hit=chunk.merge(keys,on=KEY,how='inner',validate='one_to_one')
        if len(hit):source_hits.append(hit)
    source_hits=pd.concat(source_hits,ignore_index=True)
    matched=tails.merge(source_hits,on=KEY,suffixes=('','_source'),validate='one_to_one')
    assert len(matched)==len(tails)
    for field in raw_fields:
        if field not in KEY:assert matched[field].eq(matched[field+'_source']).all(),field
    tails['source_c40_exact_match']=True
    for data,name in [(stats,'radius_by_label'),(monthly,'radius_by_month'),(tails.drop(columns='attempt_times'),'radial_records')]:data.to_csv(REPORT/f'07_08_{name}.csv',index=False)
    # Самая частая точная пара координат каждого аэропорта, обе метки.
    profile_key=['airport','C40_CROSS_LAT','C40_CROSS_LON']
    profiles=frame.groupby(profile_key).GoA.agg(rows='size',go_arounds='sum').reset_index()
    dominant=profiles.sort_values('rows',ascending=False).groupby('airport',sort=False).head(1)
    selected=frame.merge(dominant[profile_key],on=profile_key,how='inner',validate='many_to_one')
    selected['date_c40']=selected.C40_CROSS_TIME.dt.strftime('%Y-%m-%d')
    identities=selected.groupby(profile_key).agg(rows=('GoA','size'),go_arounds=('GoA','sum'),aircraft=('icao24','nunique'),dates=('date_c40','nunique'),months=('month_c40','nunique'),first_c40=('C40_CROSS_TIME','min'),last_c40=('C40_CROSS_TIME','max'),bearing=('C40_BEARING','first')).reset_index()
    airport_counts=frame.groupby('airport').GoA.agg(all_rows='size',all_go_arounds='sum').reset_index()
    identities=identities.merge(airport_counts,on='airport',validate='one_to_one')
    identities['share_airport_percent']=100*identities.rows/identities.all_rows
    identities['ga_percent']=100*identities.go_arounds/identities.rows
    identities['airport_ga_percent']=100*identities.all_go_arounds/identities.all_rows
    profile_month=selected.groupby(['airport','GoA','month_c40']).size().reset_index(name='profile_rows')
    profile_month=profile_month.merge(monthly[['airport','GoA','month_c40','rows']],on=['airport','GoA','month_c40'],validate='one_to_one')
    profile_month['share_label_month_percent']=100*profile_month.profile_rows/profile_month.rows
    assert not selected.duplicated(KEY).any()
    selected.drop(columns='attempt_times').to_csv(REPORT/'07_08_dominant_point_records.csv',index=False)
    identities.to_csv(REPORT/'07_08_dominant_points.csv',index=False)
    profile_month.to_csv(REPORT/'07_08_dominant_point_months.csv',index=False)
    types=selected.groupby(['airport','GoA','typecode'],dropna=False).size().reset_index(name='rows');types.to_csv(REPORT/'07_08_dominant_point_types.csv',index=False)
    plt.rcParams.update({'font.family':'DejaVu Sans','svg.fonttype':'none','pdf.fonttype':42})
    order=airport_counts.sort_values('all_rows',ascending=False).airport.tolist()
    fig,axes=plt.subplots(1,2,figsize=(14,9),sharey=True)
    for ax,label in zip(axes,[False,True]):
        d=stats[stats.GoA.eq(label)].set_index('airport').loc[order]
        ax.errorbar(d.median_radius_km,np.arange(len(order)),xerr=np.vstack([d.median_radius_km-d.q01_radius_km,d.q99_radius_km-d.median_radius_km]),fmt='o',capsize=2,color='#537c99' if not label else '#b45335')
        ax.axvline(RADIUS,color='#788894',linestyle='--');ax.set_yticks(range(len(order)),order);ax.set(title=f'GoA={label}',xlabel='Расстояние WGS84, км');ax.grid(alpha=.15)
    axes[0].invert_yaxis();fig.suptitle('C40: медиана и диапазон 1–99% расстояния до восстановленного центра');fig.tight_layout(rect=(0,.045,1,.96));fig.text(.01,.01,'Центры из 07_07, не независимые официальные ARP. Пунктир – 40 NM = 74,08 км. Полосы – квантили, не доверительные интервалы.',fontsize=9)
    for ext in ['png','svg','pdf']:fig.savefig(FIGURES/f'07_08_radius_by_label.{ext}',dpi=300,bbox_inches='tight')
    plt.close(fig)
    aggregated=stats.groupby('airport').agg(rows=('rows','sum'),tail_1km=('tail_1km','sum'),tail_3km=('tail_3km','sum'),min_radius_km=('min_radius_km','min'),max_radius_km=('max_radius_km','max')).reset_index()
    display=aggregated.copy();display['tail_percent']=100*display.tail_1km/display.rows
    for field in ['min_radius_km','max_radius_km','tail_percent']:display[field]=display[field].map(lambda x:f'{x:.3f}')
    display=display[['airport','rows','tail_1km','tail_3km','tail_percent','min_radius_km','max_radius_km']].rename(columns={'airport':'Аэропорт','rows':'n','tail_1km':'Отклонение >1 км','tail_3km':'Отклонение >3 км','tail_percent':'>1 км, %','min_radius_km':'Мин., км','max_radius_km':'Макс., км'})
    frequent=identities.sort_values('rows',ascending=False)[['airport','rows','go_arounds','aircraft','dates','months','share_airport_percent']].copy();frequent.share_airport_percent=frequent.share_airport_percent.map(lambda x:f'{x:.3f}')
    extreme=tails.sort_values('abs_deviation_km',ascending=False).head(10)[['flight_id','icao24','airport','GoA','radius_sphere_km','radius_wgs84_km','minutes_c40_to_first_attempt']].copy()
    for field in ['radius_sphere_km','radius_wgs84_km','minutes_c40_to_first_attempt']:extreme[field]=extreme[field].map(lambda x:f'{x:.3f}')
    summary=dict(input_sha256=before,source_sha256=source_before,rows=len(frame),go_arounds=int(frame.GoA.sum()),tail_1km=len(tails),tail_3km=int(frame.tail_3km.sum()),tail_go_arounds=int(tails.GoA.sum()),tail_percent=100*len(tails)/len(frame),all_tail_values_match_source=True,all_tail_c40_before_first_attempt=True,dominant_point_rows=len(selected),dominant_point_duplicate_landing_keys=0,max_sphere_wgs84_difference_m=float(frame.sphere_to_wgs84_m.abs().max()),distance_validation=checks,rows_deleted=0,values_changed=0,diagnostic_thresholds_km=[1,3],thresholds_not_cleanup_rules=True,estimated_centers_not_official=True,generated_utc=datetime.now(timezone.utc).isoformat())
    text=r'''# 07_08 – Радиальные отклонения и частые точки C40

## Что проверяем

В 07_07 восстановлена постоянная точка, от которой вычислено направление C40. Теперь считаем расстояние от нее до каждой точки входа. Это не независимая валидация официального ARP: центр восстановлен из самих данных. Система координат источника отдельно не подтверждена; WGS84 используется как явная расчетная гипотеза.

$$\Delta d=d-74.08\ \mathrm{км},\qquad 40\ \mathrm{NM}=74.08\ \mathrm{км}.$$

Проверяем сферу из 07_07 и эллипсоид WGS84. Для эллипсоида использована обратная формула Vincenty: $a=6378137$ м, $f=1/298.257223563$, $b=a(1-f)$, приведенная широта $U=\arctan((1-f)\tan\varphi)$. Итерационно решается разность долгот на вспомогательной сфере; итоговое расстояние $s=bA(\sigma-\Delta\sigma)$ с поправками Vincenty. Реализация проверена по [официальному примеру NGS](https://www.ngs.noaa.gov/PC_PROD/Inv_Fwd/readme.htm) на GRS80 и [примеру GeographicLib](https://geographiclib.sourceforge.io/html/python/examples.html) на WGS84 с допуском 2 мм. Контроль сходимости выполнен для всех строк; установка дополнительных библиотек не требовалась.

Пороги $|\Delta d|>1$ км и $>3$ км – диагностические списки для просмотра, не авиационные нормы и не правила очистки. Второй список входит в первый. Ни один порог не оценивает точность C40 или границы ASMA.

## Сводка радиальных отклонений

'''+markdown(display)+'\n\n'
    text+=f'За границами диагностического интервала 74,08 ± 1 км: **{len(tails):,} записей ({summary["tail_percent"]:.4f}%)**, включая **{summary["tail_go_arounds"]} GoA=True**. Отклонение более 3 км: **{summary["tail_3km"]} записей**. Максимальная разница между сферой и WGS84 в этой выборке: **{summary["max_sphere_wgs84_difference_m"]:.1f} м**. Смена модели Земли не должна автоматически объяснять многокилометровый хвост.\n\n'
    text+='![Радиальная геометрия](../../../figures/07_airports/07_08_radius_by_label.png)\n\n## Конкретные крайние записи\n\n'+markdown(extreme)+'\n\n'
    text+='''Все записи первого диагностического списка сохранены в `07_08_radial_records.csv` с исходными идентификаторами, координатами, UTC-временами и списками попыток. Значения C40 и GoA каждой записи побитово не сравнивались, но их числовое/временное равенство исходному Parquet проверено без допусков по составному ключу. Во всех этих записях C40 раньше первой отмеченной попытки; исходные ключи не повторяются. Значит, хвост существовал уже в исходнике и не появился при текущей подготовке.

Механизм отклонений не установлен. Возможны отличия расчета пересечения, редкая периодичность/интерполяция CPR, границы сегмента или сопоставление источников, но это гипотезы без исходных CPR. По одному радиусу нельзя объявить полет или координату ошибочными.

## Самая частая точная точка каждого аэропорта

'''+markdown(frequent)+'\n\n'
    text+='''Точка выбрана по частоте во всей актуальной выборке аэропорта; метка не участвовала в выборе. Число разных бортов и дней показывает, сосредоточено ли повторение в одном борте/коротком периоде. Повторов составного ключа посадки в этих группах нет. Значения, округленные для оформления таблиц, не использовались для определения точных групп.

Полные записи: `07_08_dominant_point_records.csv`. Месячные количества и доля внутри каждого класса: `07_08_dominant_point_months.csv`. Типы отдельно по классам: `07_08_dominant_point_types.csv`. Месяцы без событий выбранной точки отсутствуют в таблице профиля; календарная таблица радиальных групп содержит знаменатели для их восстановления.

Повторы согласуются с установленной секундной сеткой и сосредоточением входов в отдельных местах. Связь с конкретными опубликованными маршрутами или процедурами не подтверждена. Высокая частота геометрии не подтверждает дубль физического полета.

## Статус и дальнейшее решение

Все значения и записи сохраняются, новых исключений нет. Представлены диагностические материалы; механизм радиального хвоста без CPR не подтвержден. Нельзя переносить диагностические флаги в модель как признаки причины GoA. Следующее решение касается достаточности проверки для продолжения исследования, а не обязательного удаления хвоста. Для модели можно запланировать сравнение вариантов геометрии и проверку чувствительности к диагностическим группам на общем тесте после окончательного допуска признаков.
'''
    (REPORT/'07_08_review.md').write_text(text,encoding='utf8')
    code="""from pathlib import Path
import pandas as pd
root = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p/'europe/reports/airports/c40_geometry/07_08_radial_records.csv').exists())
report = root/'europe/reports/airports/c40_geometry'
records = pd.read_csv(report/'07_08_radial_records.csv', dtype={'icao24': str})
records.sort_values('abs_deviation_km', ascending=False).head(20)
"""
    nb=nbformat.v4.new_notebook(cells=[nbformat.v4.new_markdown_cell(text.replace('../../../figures/','../../figures/')),nbformat.v4.new_code_cell(code),nbformat.v4.new_markdown_cell('Воспроизведение: `.\\.venv\\Scripts\\python.exe -u -X utf8 europe/scripts/audit_c40_radial_cases.py`. [Полные таблицы и отчет](../../reports/airports/c40_geometry/07_08_review.md).')],metadata={'kernelspec':{'name':'python3','display_name':'Python 3','language':'python'}})
    nbformat.validate(nb);nbformat.write(nb,ROOT/'europe/notebooks/07_airports/07_08_c40_radial_cases.ipynb')
    assert digest(INPUT)==before and digest(SOURCE)==source_before
    summary['input_unchanged']=True;summary['source_unchanged']=True
    (REPORT/'07_08_audit.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    path=ROOT/'europe/notebooks/07_airports/README.md';note='\n\n- [07_08 – Радиальные отклонения и частые точки C40](07_08_c40_radial_cases.ipynb): проверка сферы/эллипсоида, исходных записей и состава повторов.\n';current=path.read_text(encoding='utf8')
    if note.strip() not in current:path.write_text(current+note,encoding='utf8')
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)
    print(extreme.to_string(index=False),flush=True)
    print(frequent.to_string(index=False),flush=True)


if __name__=='__main__':main()
