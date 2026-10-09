"""Радиальная геометрия EGCC и границы вклада сетки координат."""
from pathlib import Path
import sys
import json
import hashlib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import nbformat
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[3]
STUDY = ROOT / 'research/c40_geometry'
sys.path.insert(0, str(ROOT / 'europe/scripts'))
# Импортируем проверенные чистые функции; европейские файлы не изменяются.
from audit_c40_radial_cases import ellipsoid_distance, validate_distance
from investigate_c40_geometry import distance


def main():
    checks = validate_distance()
    source = ROOT / 'europe/data/analytical/06_05_cases_excluded.parquet'
    expected = '1111c7480ff7615042a332c586c8f60459bf5d6000da11ea4d0ec2a773fc0719'
    sha = hashlib.sha256()
    with source.open('rb') as stream:
        for chunk in iter(lambda: stream.read(8*1024*1024), b''):
            sha.update(chunk)
    assert sha.hexdigest() == expected
    frame = pq.read_table(source, columns=['flight_id','icao24','landing_time','airport','GoA','C40_CROSS_LAT','C40_CROSS_LON','C40_CROSS_TIME','C40_BEARING'],filters=[('airport','=','EGCC')]).to_pandas()
    lat, lon = 53+21/60+14/3600, -(2+16/60+30/3600)
    reference = pd.read_csv(STUDY/'reports/historical_arp_sources.csv')
    official = reference[(reference.airport=='EGCC') & (reference.designation=='explicit_ARP_AD2_2')]
    assert len(official) >= 4
    assert np.allclose(official.published_lat,lat,rtol=0,atol=1e-12)
    assert np.allclose(official.published_lon,lon,rtol=0,atol=1e-12)
    y,x = frame.C40_CROSS_LAT.to_numpy(),frame.C40_CROSS_LON.to_numpy()
    frame['sphere_km'] = distance(lat,lon,y,x)
    frame['wgs84_km'] = ellipsoid_distance(lat,lon,y,x)
    frame['deviation_m'] = 1000*(frame.wgs84_km-74.08)
    frame['sphere_deviation_m'] = 1000*(frame.sphere_km-74.08)
    frame['model_difference_m'] = 1000*(frame.wgs84_km-frame.sphere_km)
    # Максимальное расстояние до углов ячейки: два сценария, алгоритм округления неизвестен.
    for scenario,halfwidth in [('nearest',.5),('one_second',1.)]:
        corner_distances = [1000*ellipsoid_distance(y,x,y+dy*halfwidth/3600,x+dx*halfwidth/3600)
                            for dy in [-1,1] for dx in [-1,1]]
        frame[f'grid_{scenario}_bound_m'] = np.max(corner_distances,axis=0)
    summary = frame.groupby('GoA').agg(rows=('GoA','size'),median_radius_km=('wgs84_km','median'),
              min_radius_km=('wgs84_km','min'),max_radius_km=('wgs84_km','max'),
              median_deviation_m=('deviation_m','median'),max_abs_model_difference_m=('model_difference_m',lambda s:s.abs().max()),
              max_nearest_grid_bound_m=('grid_nearest_bound_m','max'),max_one_second_grid_bound_m=('grid_one_second_bound_m','max'))
    summary['abs_deviation_gt_1km'] = frame.groupby('GoA').deviation_m.apply(lambda s:int((s.abs()>1000).sum()))
    summary['median_sphere_deviation_m'] = frame.groupby('GoA').sphere_deviation_m.median()
    summary['sphere_beyond_one_second_bound'] = frame.groupby('GoA').apply(lambda g:int((g.sphere_deviation_m.abs()>g.grid_one_second_bound_m).sum()),include_groups=False)
    summary['beyond_one_second_bound'] = frame.groupby('GoA').apply(lambda g:int((g.deviation_m.abs()>g.grid_one_second_bound_m).sum()),include_groups=False)
    reports = STUDY/'reports';reports.mkdir(exist_ok=True)
    summary.to_csv(reports/'egcc_geometry_by_label.csv')
    tails = frame[frame.deviation_m.abs()>1000].copy()
    tails.to_csv(reports/'egcc_radial_records.csv',index=False)
    previous = pd.read_csv(ROOT/'europe/reports/airports/c40_geometry/07_08_radial_records.csv')
    assert set(tails.flight_id)==set(previous.loc[previous.airport=='EGCC','flight_id'])
    figdir = STUDY/'figures';figdir.mkdir(exist_ok=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10})
    fig,axs=plt.subplots(1,3,figsize=(16,4.8),layout='constrained')
    for label,color in [(False,'#286b9e'),(True,'#d66737')]:
        group=frame[frame.GoA==label]
        axs[0].hist(group.deviation_m,bins=np.arange(-1200,1701,25),density=True,histtype='step',linewidth=1.8,color=color,label=f'GoA={label}, n={len(group):,}')
        axs[0].hist(group.sphere_deviation_m,bins=np.arange(-1200,1701,25),density=True,histtype='step',linewidth=1.3,linestyle='--',color=color)
    axs[0].axvline(0,color='#333333',linestyle='--',linewidth=1)
    axs[0].set(xlabel='Расстояние минус 74.08 км, м',ylabel='Плотность, 1/м',title='Сплошная: WGS84; пунктир: сфера')
    axs[0].legend(fontsize=8)
    sample=frame.iloc[::max(1,len(frame)//6000)]
    axs[1].scatter(sample.C40_BEARING,sample.model_difference_m,s=3,alpha=.4,color='#286b9e')
    axs[1].set(xlabel='Азимут аэропорт → C40, градусы',ylabel='WGS84 минус сфера, м',title='Разница геометрических моделей',xlim=(0,360))
    axs[2].hist(frame.grid_one_second_bound_m,bins=25,color='#438778',alpha=.85)
    axs[2].set(xlabel='Граница смещения точки, м',ylabel='Число записей',title='Ячейка ±1 угловая секунда')
    fig.suptitle('EGCC: официальный ARP и геометрия C40',fontsize=15)
    for extension in ['png','svg','pdf']:
        fig.savefig(figdir/f'02_egcc_geometry.{extension}',dpi=300)
    plt.close(fig)
    result=dict(airport='EGCC',rows=len(frame),goa=int(frame.GoA.sum()),input_sha256=expected,
                arp_lat=lat,arp_lon=lon,official_ad2_documents=len(official),
                first_c40=str(frame.C40_CROSS_TIME.min()),last_c40=str(frame.C40_CROSS_TIME.max()),
                validation=checks,max_nearest_grid_bound_m=float(frame.grid_nearest_bound_m.max()),
                max_one_second_grid_bound_m=float(frame.grid_one_second_bound_m.max()),
                grid_bound_assumption='Within +/-0.5 arcsec (nearest) or +/-1 arcsec; rounding mechanism not established')
    (reports/'egcc_geometry_audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    mathematical = r'''## Граница влияния сетки

Координаты лежат на сетке с шагом $h=1/3600$ градуса. Само наблюдение сетки не устанавливает алгоритм округления.

Для округления к ближайшему узлу предполагаем $|\delta\varphi|,|\delta\lambda|\le h/2$; для второго сценария допускаем до $h$ по каждой координате. Второй сценарий также требует предположения о максимальной ошибке: он не охватывает произвольную ошибку исходных координат.

В первом приближении смещение точки:
$$\epsilon\approx\sqrt{(M\delta\varphi)^2+(N\cos\varphi\,\delta\lambda)^2},$$
где приращения в радианах, $M=a(1-e^2)/(1-e^2\sin^2\varphi)^{3/2}$, $N=a/\sqrt{1-e^2\sin^2\varphi}$ – радиусы кривизны WGS84.

В расчетах используем геодезические расстояния до четырех углов ячейки; это оценка максимального смещения для указанных локальных ячеек. По неравенству треугольника при фиксированном ARP:
$$|d(A,P)-d(A,\widehat P)|\le d(P,\widehat P)\le\epsilon.$$
Если центр тоже имеет ошибку $\epsilon_A$, общая граница становится $\epsilon+\epsilon_A$. Здесь сравниваем с опубликованным ARP; точность его физического определения не установлена.

Отклонение от 74.08 км превышающее эту границу нельзя объяснить только выбранным сценарием квантования. Оно не становится автоматически ошибкой записи: определение пересечения, частота наблюдений и исходные траектории еще не проверены.
'''
    report = '# EGCC – независимый ARP и радиальная геометрия\n\n'
    report += 'AD 2.2 подтверждает ARP 532114N 0021630W: середина ВПП 05L/23R. Совпадение проверено в четырех исторических срезах. Это не непрерывная проверка всех AIRAC.\n\n'
    report += '```text\n'+summary.to_string()+'\n```\n\n'+mathematical
    report += '\nСравниваем обе модели до интерпретации квантования. Смещение WGS84 относительно сферы систематически зависит от азимута. Превышение границы сетки относительно 74.08 км в WGS84 само по себе не устанавливает ошибку пересечения, если исходная граница определялась на сфере.\n'
    report += '\n\nВсе строки сохранены. Сфера и WGS84 дают разные расстояния; используемая авторами модель расстояния пока не установлена. Азимут не является магнитным курсом захода.\n'
    (reports/'02_egcc_geometry.md').write_text(report,encoding='utf8')
    nb=nbformat.v4.new_notebook(cells=[nbformat.v4.new_markdown_cell('# EGCC – независимая геометрическая проверка\n\nARP подтвержден историческими AD 2.2. Данные читаются из Европы, результаты сохраняются в research. Все записи сохраняются.'),
        nbformat.v4.new_code_cell("from pathlib import Path\nimport pandas as pd\nfrom IPython.display import display, Image\nroot = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p/'research/c40_geometry/reports/egcc_geometry_audit.json').exists())\nstudy=root/'research/c40_geometry'\ndisplay(pd.read_csv(study/'reports/egcc_geometry_by_label.csv'))"),
        nbformat.v4.new_markdown_cell(mathematical),
        nbformat.v4.new_code_cell("display(Image(filename=str(study/'figures/02_egcc_geometry.png')))"),
        nbformat.v4.new_markdown_cell('Воспроизведение из корня: `.\\.venv\\Scripts\\python.exe -u -X utf8 research/c40_geometry/scripts/analyze_egcc_geometry.py`. Различия между GoA показаны описательно; причинный эффект и статистическая значимость не оценивались.')])
    nb.metadata['kernelspec']=dict(display_name='Python 3',language='python',name='python3')
    nbformat.write(nb,STUDY/'notebooks/02_egcc_geometry.ipynb')
    assert len(frame)==87435 and int(frame.GoA.sum())==306
    print(summary.to_string(),flush=True)
    print(json.dumps(result,ensure_ascii=False),flush=True)


if __name__=='__main__':
    main()
