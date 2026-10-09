"""07_07: квантование C40 и обратная проверка формулы направления."""
import json
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from types import SimpleNamespace
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import nbformat
from audit_airports import ROOT, INPUT
from audit_feature_availability import digest

REPORT=ROOT/'europe/reports/airports/c40_geometry'
FIGURES=ROOT/'europe/figures/07_airports'


def least_squares(function,initial,**options):
    """Локальный Гаусс–Ньютон с численным Якобианом и уменьшением шага."""
    x=np.asarray(initial,dtype=float).copy()
    success=False
    for _ in range(options.get('max_nfev',200)):
        residual=np.asarray(function(x));loss=float(residual@residual)
        columns=[]
        for index in range(len(x)):
            step=1e-5 if index<2 else 1e-4
            plus=x.copy();minus=x.copy();plus[index]+=step;minus[index]-=step
            columns.append((function(plus)-function(minus))/(2*step))
        jacobian=np.column_stack(columns)
        change=np.linalg.lstsq(jacobian,-residual,rcond=None)[0]
        if np.max(np.abs(change))<1e-12:success=True;break
        improved=False
        for scale in [1,.5,.25,.125,.0625,.03125,.015625,.0078125]:
            candidate=x+scale*change
            if not (-90<candidate[0]<90 and -180<candidate[1]<180):continue
            if len(candidate)>2 and candidate[2]<=0:continue
            values=function(candidate)
            if float(values@values)<loss:
                x=candidate;improved=True;break
        if not improved:success=True;break
    return SimpleNamespace(x=x,success=success)


def bearing(lat1,lon1,lat2,lon2):
    p1,p2=np.radians(lat1),np.radians(lat2);delta=np.radians(lon2-lon1)
    return np.degrees(np.arctan2(np.sin(delta)*np.cos(p2),np.cos(p1)*np.sin(p2)-np.sin(p1)*np.cos(p2)*np.cos(delta)))%360


def difference(a,b):return (a-b+180)%360-180


def distance(lat1,lon1,lat2,lon2):
    p1,p2=np.radians(lat1),np.radians(lat2)
    a=np.sin((p2-p1)/2)**2+np.cos(p1)*np.cos(p2)*np.sin(np.radians(lon2-lon1)/2)**2
    return 2*6371.0088*np.arcsin(np.sqrt(np.clip(a,0,1)))


def main():
    REPORT.mkdir(parents=True,exist_ok=True)
    expected=json.loads((ROOT/'europe/reports/airports/distributions/07_06_audit.json').read_text(encoding='utf8'))
    before=digest(INPUT);assert before==expected['input_sha256']
    fields=['airport','GoA','C40_BEARING','C40_CROSS_LAT','C40_CROSS_LON']
    frame=pq.read_table(INPUT,columns=fields).to_pandas(ignore_metadata=True)
    grid=[];pairs=[];results=[]
    for airport,group in frame.groupby('airport',sort=True):
        for field in ['C40_CROSS_LAT','C40_CROSS_LON']:
            for scale in [60,3600]:
                e=np.abs(group[field].to_numpy()*scale-np.rint(group[field].to_numpy()*scale))
                grid.append(dict(airport=airport,field=field,scale=scale,rows=len(group),on_grid=int((e<1e-7).sum()),max_residual_units=e.max(),p99_residual_units=np.quantile(e,.99)))
        unique=group.groupby(['C40_CROSS_LAT','C40_CROSS_LON']).C40_BEARING.agg(rows='size',distinct_bearings='nunique',bearing_min='min',bearing_max='max').reset_index()
        pairs.append(dict(airport=airport,rows=len(group),coordinate_pairs=len(unique),pairs_multiple_bearings=int(unique.distinct_bearings.gt(1).sum()),rows_multiple_bearings=int(unique.loc[unique.distinct_bearings.gt(1),'rows'].sum())))
        # Подгонка и проверка на разных уникальных координатах, без использования GoA.
        unique=unique.sort_values(['C40_CROSS_LAT','C40_CROSS_LON'])
        train=unique.iloc[::2];test=unique.iloc[1::2]
        stride=max(1,len(train)//5000);train=train.iloc[::stride]
        lat=train.C40_CROSS_LAT.to_numpy();lon=train.C40_CROSS_LON.to_numpy();target=train.bearing_min.to_numpy()
        initial=[group.C40_CROSS_LAT.median(),group.C40_CROSS_LON.median(),74.08]
        circle=least_squares(lambda x:distance(x[0],x[1],lat,lon)-x[2],initial,bounds=([-90,-180,0],[90,180,1000]),max_nfev=200,xtol=1e-12,ftol=1e-12,gtol=1e-12)
        center=circle.x[:2]
        for direction in ['airport_to_point','point_to_airport']:
            def predict(x,p_lat,p_lon):
                return bearing(x[0],x[1],p_lat,p_lon) if direction=='airport_to_point' else bearing(p_lat,p_lon,x[0],x[1])
            fit=least_squares(lambda x:difference(predict(x,lat,lon),target),center,bounds=([-90,-180],[90,180]),max_nfev=200,xtol=1e-12,ftol=1e-12,gtol=1e-12)
            residual=np.abs(difference(predict(fit.x,test.C40_CROSS_LAT.to_numpy(),test.C40_CROSS_LON.to_numpy()),test.bearing_min.to_numpy()))
            radii=distance(fit.x[0],fit.x[1],group.C40_CROSS_LAT.to_numpy(),group.C40_CROSS_LON.to_numpy())
            results.append(dict(airport=airport,formula='spherical_initial_bearing',direction=direction,fit_lat=fit.x[0],fit_lon=fit.x[1],fit_unique_points=len(train),test_unique_points=len(test),fit_success=bool(fit.success),median_abs_error_deg=np.median(residual),p99_abs_error_deg=np.quantile(residual,.99),max_abs_error_deg=residual.max(),radius_q01_km=np.quantile(radii,.01),radius_median_km=np.median(radii),radius_q99_km=np.quantile(radii,.99),circle_fitted_radius_km=circle.x[2]))
        print(airport,results[-2]['p99_abs_error_deg'],results[-1]['p99_abs_error_deg'],flush=True)
    grid=pd.DataFrame(grid);pairs=pd.DataFrame(pairs);results=pd.DataFrame(results)
    for data,name in [(grid,'coordinate_grid'),(pairs,'bearing_by_coordinate'),(results,'bearing_formula')]:data.to_csv(REPORT/f'07_07_{name}.csv',index=False)
    best=results.sort_values('p99_abs_error_deg').groupby('airport',sort=False).head(1).sort_values('airport')
    best.to_csv(REPORT/'07_07_best_formula.csv',index=False)
    seconds=grid[grid.scale.eq(3600)]
    summary=dict(input_sha256=before,rows=len(frame),coordinate_values=len(frame)*2,arcsecond_grid_values=int(seconds.on_grid.sum()),max_grid_residual_arcsec=float(seconds.max_residual_units.max()),pairs_multiple_bearings=int(pairs.pairs_multiple_bearings.sum()),direction_counts=best.direction.value_counts().to_dict(),largest_best_p99_error_deg=float(best.p99_abs_error_deg.max()),rows_deleted=0,values_changed=0,estimated_centers_not_official=True)
    example=frame[frame.airport.eq('EGLL')].groupby(['C40_CROSS_LAT','C40_CROSS_LON']).size().reset_index(name='rows')
    dominant=example.sort_values('rows',ascending=False).iloc[0]
    zoom=example[(example.C40_CROSS_LAT-dominant.C40_CROSS_LAT).abs().lt(10/3600)&(example.C40_CROSS_LON-dominant.C40_CROSS_LON).abs().lt(10/3600)]
    plt.rcParams.update({'font.family':'DejaVu Sans','svg.fonttype':'none','pdf.fonttype':42})
    fig,axes=plt.subplots(1,2,figsize=(13,5))
    center=best[best.airport.eq('EGLL')].iloc[0]
    points=example.iloc[::max(1,len(example)//12000)]
    x=(points.C40_CROSS_LON-center.fit_lon)*111.195*np.cos(np.radians(center.fit_lat));y=(points.C40_CROSS_LAT-center.fit_lat)*111.195
    axes[0].scatter(x,y,s=3,alpha=.3);angle=np.linspace(0,2*np.pi,361);axes[0].plot(74.08*np.cos(angle),74.08*np.sin(angle),linestyle='--',color='#b45335');axes[0].set(aspect='equal',xlabel='Восток относительно оцененного центра, км',ylabel='Север, км',title='EGLL: уникальные точки C40 и круг 40 NM')
    im=axes[1].scatter(3600*(zoom.C40_CROSS_LON-dominant.C40_CROSS_LON),3600*(zoom.C40_CROSS_LAT-dominant.C40_CROSS_LAT),c=np.log10(zoom.rows),s=35,cmap='viridis');fig.colorbar(im,ax=axes[1],label='log10(число записей)')
    axes[1].set(xlabel='Смещение долготы, угловые секунды',ylabel='Смещение широты, угловые секунды',title='Увеличение участка: координатная сетка');axes[1].grid(alpha=.2);fig.tight_layout()
    for ext in ['png','svg','pdf']:fig.savefig(FIGURES/f'07_07_coordinate_grid.{ext}',dpi=300,bbox_inches='tight')
    plt.close(fig)
    display=best[['airport','direction','p99_abs_error_deg','radius_median_km','radius_q01_km','radius_q99_km']].copy()
    for f in ['p99_abs_error_deg','radius_median_km','radius_q01_km','radius_q99_km']:display[f]=display[f].map(lambda x:f'{x:.10g}')
    table='| '+' | '.join(display.columns)+' |\n| '+' | '.join(['---']*len(display.columns))+' |\n'+''.join('| '+' | '.join(row)+' |\n' for row in display.itertuples(index=False,name=None))
    text=r'''# 07_07 – Почему повторяется геометрия C40

## Источник полей

[Figuet et al., 2023, разделы 2.2–2.3](https://journals.open.tudelft.nl/joas/article/download/7219/5898): C40 получен из EUROCONTROL Correlated Position Reports, а метки посадочных сегментов – из OpenSky. ASMA описана как область радиуса 40 NM вокруг опорной точки аэропорта. [Zenodo](https://zenodo.org/records/10210178) определяет координаты и bearing как положение и направление у входа. Это не обязательно сырые координаты ADS-B в сохраненном OpenSky-сегменте. Публичный ноутбук автора читает готовый Parquet; сохраненный материал не содержит алгоритма получения C40 или точности координат. Указанный ниже механизм квантования установлен численно, его конкретный этап в цепочке источника не подтвержден.

## Проверка координатной сетки

Для координаты в градусах $x$ проверяем расстояние до целого количества угловых секунд:

$$\epsilon(x)=\left|3600x-\operatorname{round}(3600x)\right|.$$

Диагностический числовой допуск: $\epsilon<10^{-7}$ угловой секунды. Это допуск проверки floating point, не точность измерения самолета. Сетка $1/3600$ градуса соответствует одной угловой секунде. Линейный шаг на сфере:

$$\Delta s_\varphi\approx R\frac{\pi}{180\cdot3600},\qquad
\Delta s_\lambda\approx R\cos\varphi\frac{\pi}{180\cdot3600}.$$

При $R=6371008.8$ м: около 30,9 м по широте; по долготе шаг зависит от широты. Это шаг представления, не оценка погрешности наблюдения. Если координаты квантуются, разные положения попадают в одну ячейку. Повторение точки у разных полетов возможно без дублирования полета.

## Проверка зависимости bearing от координат

Для начального сферического азимута от точки 1 к точке 2, в радианах:

$$y=\sin(\lambda_2-\lambda_1)\cos\varphi_2,$$
$$x=\cos\varphi_1\sin\varphi_2-\sin\varphi_1\cos\varphi_2\cos(\lambda_2-\lambda_1),$$
$$\theta=\operatorname{atan2}(y,x)\pmod{2\pi}.$$

Проверяем обе ориентации: аэропорт → точка и точка → аэропорт. Это направление между точками, не курс самолета. Угловая ошибка с учетом цикличности:

$$r_i=\big((\hat\theta_i-\theta_i+180)\bmod360\big)-180.$$

Постоянный центр каждого аэропорта оценивается минимизацией $\sum_i r_i^2$. Половина отсортированных уникальных координат служит подгонке, другая – проверке; для подгонки берется систематическая подвыборка примерно до 5000 точек. Одна координата не входит одновременно в подгонку и проверку. Метка GoA в подгонке не используется. Это числовая проверка функции, не оценка качества ML-модели. Оцененные центры не являются официальными историческими опорными точками аэропортов.

Радиус проверяем по сферической формуле:

$$a=\sin^2\frac{\varphi_2-\varphi_1}{2}+\cos\varphi_1\cos\varphi_2\sin^2\frac{\lambda_2-\lambda_1}{2},\qquad
d=2R\arcsin\sqrt a.$$

$40\ \mathrm{NM}=74.08$ км. Расстояние на сфере – диагностическое приближение; отличие от 74,08 км само по себе не является ошибкой. Независимые официальные координаты и эллипсоидальная геометрия здесь не проверены.

## Результаты

'''
    text+=f'На секундную сетку попадают **{summary["arcsecond_grid_values"]:,} из {summary["coordinate_values"]:,} значений координат**. Максимальный остаток: {summary["max_grid_residual_arcsec"]:.3g} угловой секунды. Число координатных пар, которым соответствуют разные точные значения bearing: **{summary["pairs_multiple_bearings"]:,}**.\n\n'
    text+=table+'\n\n![Сетка и геометрия EGLL](../../../figures/07_airports/07_07_coordinate_grid.png)\n\n'
    text+='''## Что это означает для повторов и модели

Сначала читаем числовые результаты: квантование координат, постоянство bearing для пары координат и ошибка восстановления азимута проверяют разные части гипотезы. Маленькая ошибка на других координатах дает свидетельство вычислительной зависимости bearing от координат и центра; она не подтверждает точность слежения, источник округления или правильность сопоставления записи с полетом.

Если связь подтверждается, три поля не являются тремя независимыми измерениями. Bearing может быть удобным преобразованием геометрии; при моделировании сравним варианты признаков после окончательного допуска, без удаления исходных значений. Повторная геометрия сама по себе не повод удалять запись. Для нейтрального сравнения используем одинаковые временные границы и тест.

Предыдущая проверка 07_06 уже показала: частые профили встречаются у разных бортов и дат, а составной ключ посадки в выбранных группах не повторяется. Квантование объясняет возможность точных совпадений; частота отдельных ячеек также может зависеть от маршрутов и структуры движения, что без траекторий не установлено.

## Границы вывода

Сырые CPR и алгоритм формирования C40 отсутствуют. Нельзя установить, округлялись ли координаты при хранении, передаче, подготовке EU-NM или при сборке этой таблицы. Траектории независимо не проверены. Решения о пригодности признаков и первый/последний вход C40 здесь не пересматриваются. Все записи и значения сохраняются.
'''
    (REPORT/'07_07_review.md').write_text(text,encoding='utf8')
    source_manifest=dict(article='https://journals.open.tudelft.nl/joas/article/download/7219/5898',dataset='https://zenodo.org/records/10210178',local_article_sha256=digest(ROOT/'docs/sources/files/Figuet_et_al_2023_go_arounds.pdf'),local_notebook_sha256=digest(ROOT/'docs/sources/files/c40_review/author_OSN_paper_code.ipynb'),algorithm_available=False)
    (REPORT/'07_07_sources.json').write_text(json.dumps(source_manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    code="""from pathlib import Path
import pandas as pd
root = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p/'europe/reports/airports/c40_geometry/07_07_best_formula.csv').exists())
report = root/'europe/reports/airports/c40_geometry'
pd.read_csv(report/'07_07_best_formula.csv')
"""
    nb=nbformat.v4.new_notebook(cells=[nbformat.v4.new_markdown_cell(text.replace('../../../figures/','../../figures/')),nbformat.v4.new_code_cell(code),nbformat.v4.new_markdown_cell('Воспроизведение: `.\\.venv\\Scripts\\python.exe -u -X utf8 europe/scripts/investigate_c40_geometry.py`. [Полный отчет](../../reports/airports/c40_geometry/07_07_review.md).')],metadata={'kernelspec':{'name':'python3','display_name':'Python 3','language':'python'}})
    nbformat.validate(nb);nbformat.write(nb,ROOT/'europe/notebooks/07_airports/07_07_c40_geometry.ipynb')
    assert digest(INPUT)==before;summary['input_unchanged']=True
    (REPORT/'07_07_audit.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    path=ROOT/'europe/notebooks/07_airports/README.md';note='\n\n- [07_07 – Координатная сетка и формула направления C40](07_07_c40_geometry.ipynb): математическая проверка происхождения повторяющихся профилей.\n';current=path.read_text(encoding='utf8')
    if note.strip() not in current:path.write_text(current+note,encoding='utf8')
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)
    print(best.to_string(index=False),flush=True)


if __name__=='__main__':main()
