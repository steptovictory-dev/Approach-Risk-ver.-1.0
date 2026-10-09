"""Конкретные радиальные случаи EGCC: источник, контекст и условная дискретизация."""
from pathlib import Path
import sys
import json
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import nbformat
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[3]
STUDY=ROOT/'research/c40_geometry'
sys.path.insert(0,str(ROOT/'europe/scripts'))
from audit_c40_radial_cases import ellipsoid_distance
from investigate_c40_geometry import distance


def main():
    fields=['flight_id','icao24','airport','landing_time','GoA','C40_CROSS_TIME','C40_CROSS_LAT','C40_CROSS_LON','C40_BEARING','n_attempts','attempt_times','typecode']
    analytical=ROOT/'europe/data/analytical/06_05_cases_excluded.parquet'
    original=ROOT/'europe/data/source/osn23_landings_merged.parquet.gzip'
    frame=pq.read_table(analytical,columns=fields,filters=[('airport','=','EGCC')]).to_pandas(ignore_metadata=True)
    original_fields=pq.read_schema(original).names
    raw=pq.read_table(original,columns=[c for c in fields if c in original_fields],filters=[('airport','=','EGCC')]).to_pandas(ignore_metadata=True)
    frame['sphere_m']=1000*(distance(53.35388888888889,-2.275,frame.C40_CROSS_LAT.to_numpy(),frame.C40_CROSS_LON.to_numpy())-74.08)
    frame['wgs84_m']=1000*(ellipsoid_distance(53.35388888888889,-2.275,frame.C40_CROSS_LAT.to_numpy(),frame.C40_CROSS_LON.to_numpy())-74.08)
    frame['first_attempt']=frame.attempt_times.map(lambda values:pd.to_datetime(values[0],utc=True))
    frame['minutes_to_first_attempt']=(frame.first_attempt-frame.C40_CROSS_TIME).dt.total_seconds()/60
    frame['minutes_to_landing']=(frame.landing_time-frame.C40_CROSS_TIME).dt.total_seconds()/60
    cases=frame[frame.wgs84_m.abs()>1000].copy()
    assert len(cases)==6
    key=['icao24','airport','landing_time']
    merged=cases.merge(raw,on=key,suffixes=('_analytical','_source'),validate='one_to_one')
    verified=['flight_id','C40_CROSS_TIME','C40_CROSS_LAT','C40_CROSS_LON','C40_BEARING','n_attempts']
    for field in verified:
        assert (merged[field+'_analytical']==merged[field+'_source']).all(),field
    assert cases.n_attempts.eq(1).all() and (cases.C40_CROSS_TIME<cases.first_attempt).all()
    contexts=[]
    neighbors=[]
    controls=frame[~frame.GoA]
    for record in cases.itertuples():
        point=(frame.C40_CROSS_LAT==record.C40_CROSS_LAT)&(frame.C40_CROSS_LON==record.C40_CROSS_LON)
        board=frame[frame.icao24==record.icao24]
        nearby=board[(board.landing_time-record.landing_time).abs()<=pd.Timedelta(days=30)]
        others=nearby[nearby.flight_id!=record.flight_id]
        neighbor=others.copy()
        neighbor['case_flight_id']=record.flight_id
        neighbor['days_from_case']=(neighbor.landing_time-record.landing_time).dt.total_seconds()/86400
        neighbors.append(neighbor)
        circular=(frame.C40_BEARING-record.C40_BEARING+180)%360-180
        sector=frame[(circular.abs()<=5)&(frame.flight_id!=record.flight_id)]
        contexts.append(dict(flight_id=record.flight_id,point_rows=int(point.sum()),same_board_total=len(board),
           same_board_other_rows_30d=len(others),same_board_other_max_abs_wgs84_m=float(others.wgs84_m.abs().max()) if len(others) else None,
           sector_other_rows=len(sector),sector_median_sphere_m=float(sector.sphere_m.median()),
           minutes_to_landing_percentile_false=100*float((controls.minutes_to_landing<=record.minutes_to_landing).mean()),
           landing_key_source_matches=1,source_fields_exact=True))
    context=pd.DataFrame(contexts)
    cases=cases.merge(context,on='flight_id',validate='one_to_one')
    cases['sphere_gt_1km']=cases.sphere_m.abs()>1000
    cases['wgs84_gt_1km']=cases.wgs84_m.abs()>1000
    # Условный радиальный ход. Эти скорости не измерены и не приравниваются к воздушной скорости.
    scenarios=[]
    for record in cases.itertuples():
        for model,residual in [('sphere',record.sphere_m),('wgs84',record.wgs84_m)]:
            for radial_speed in [50.,100.,150.]:
                scenarios.append(dict(flight_id=record.flight_id,model=model,radial_speed_m_s=radial_speed,
                      abs_residual_m=abs(residual),hypothetical_seconds=abs(residual)/radial_speed,
                      direction='outside' if residual>0 else 'inside'))
    scenarios=pd.DataFrame(scenarios)
    reports=STUDY/'reports'
    cases.to_csv(reports/'03_egcc_case_review.csv',index=False)
    pd.concat(neighbors,ignore_index=True).to_csv(reports/'03_egcc_same_board_neighbors.csv',index=False)
    scenarios.to_csv(reports/'03_egcc_sampling_scenarios.csv',index=False)
    summary=dict(rows=len(frame),cases=len(cases),GoA=int(cases.GoA.sum()),sphere_gt_1km=int(cases.sphere_gt_1km.sum()),
                 wgs84_gt_1km=int(cases.wgs84_gt_1km.sum()),all_source_fields_exact=True,
                 all_one_attempt=True,all_c40_before_attempt=True,unique_boards=int(cases.icao24.nunique()),
                 unique_dates=int(cases.landing_time.dt.date.nunique()),same_point_counts=cases.point_rows.tolist(),
                 source_rows_egcc=len(raw),formula='abs(radial_deviation_m)/assumed_radial_speed_m_s',
                 assumptions='Constant local radial speed, same distance model; no observed speeds or sampling intervals.')
    (reports/'03_egcc_case_audit.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    fig,axs=plt.subplots(1,2,figsize=(13,5),layout='constrained')
    positions=np.arange(len(cases))
    for field,label,color,shift in [('sphere_m','Сфера','#286b9e',-.15),('wgs84_m','WGS84','#d66737',.15)]:
        axs[0].barh(positions+shift,cases[field],height=.28,label=label,color=color)
    axs[0].set_yticks(positions,cases.flight_id)
    axs[0].axvline(0,color='#555',linewidth=1)
    for threshold in [-1000,1000]:axs[0].axvline(threshold,color='#555',linestyle='--',linewidth=1)
    axs[0].set(xlabel='Расстояние минус 74.08 км, м',title='Порог 1 км зависит от метрики')
    axs[0].legend()
    for index,record in enumerate(cases.itertuples()):
        speed=np.linspace(30,170,120)
        axs[1].plot(speed,abs(record.sphere_m)/speed,label=record.flight_id)
    axs[1].set(xlabel='Предполагаемая радиальная скорость, м/с',ylabel='Условный временной сдвиг, с',title='Сценарий: |отклонение на сфере| / скорость')
    axs[1].legend(fontsize=8)
    fig.suptitle('EGCC: шесть сохраненных записей, без установленной причины',fontsize=13)
    for extension in ['png','svg','pdf']:fig.savefig(STUDY/f'figures/03_egcc_cases.{extension}',dpi=300)
    plt.close(fig)
    explanation=r'''## Что проверено

Шесть разных бортов и дат; все записи имеют одну попытку посадки. C40 предшествует попытке. Ключи и выбранные геометрические поля точно совпадают с исходным Parquet. По WGS84 все шесть превышают 1 км; по сфере только три. Следовательно, состав группы по этому диагностическому порогу чувствителен к метрике. Это не правило удаления.

## Что дает временной сценарий

При постоянной локальной радиальной скорости $v_r$:
$$\Delta t\approx |d-r|/|v_r|.$$
Это расчет совместимости гипотезы с величиной отклонения. Скорость и частота CPR в этих записях неизвестны. Радиальная скорость является компонентой путевой скорости относительно ARP и не равна воздушной скорости самолета. Числа 50, 100 и 150 м/с заданы как сценарии, не как измерения или авиационная норма.

При входящем движении точка снаружи границы совместима с наблюдением до пересечения, внутри – после него. Но без траектории направление радиального движения не установлено; по одной точке выбирать механизм нельзя.

Близкие записи того же борта и азимутного сектора – описательный контекст. Это не независимые контрольные группы, а другие наблюдения той же базы. Процентили времени до посадки вычислены среди GoA=False и не оценивают статистическую значимость.

## Решение

Все записи сохранены. Для каждой отдельно фиксируем метрику, отклонение и диагностические признаки. Следующий источник для проверки механизма – последовательность CPR/траектория и алгоритм выбора точки пересечения. Уходов в этой группе нет; она не объясняет всплеск GoA февраля 2020 года.
'''
    (reports/'03_egcc_cases.md').write_text('# Разбор шести радиальных записей EGCC\n\n'+explanation+'\n\n```text\n'+cases[['flight_id','typecode','sphere_m','wgs84_m','minutes_to_landing','point_rows','same_board_other_rows_30d']].to_string(index=False)+'\n```\n',encoding='utf8')
    notebook=nbformat.v4.new_notebook(cells=[nbformat.v4.new_markdown_cell('# Конкретные радиальные случаи EGCC\n\nИсточник и локальный контекст каждой записи. Все строки сохраняются.'),
       nbformat.v4.new_code_cell("from pathlib import Path\nimport pandas as pd\nfrom IPython.display import display, Image\nroot=next(p for p in [Path.cwd(),*Path.cwd().parents] if (p/'research/c40_geometry/reports/03_egcc_case_review.csv').exists())\nstudy=root/'research/c40_geometry'\ncases=pd.read_csv(study/'reports/03_egcc_case_review.csv')\ndisplay(cases[['flight_id','typecode','GoA','sphere_m','wgs84_m','minutes_to_landing','point_rows','same_board_other_rows_30d']])"),
       nbformat.v4.new_markdown_cell(explanation),
       nbformat.v4.new_code_cell("display(Image(filename=str(study/'figures/03_egcc_cases.png')))\ndisplay(pd.read_csv(study/'reports/03_egcc_sampling_scenarios.csv'))"),
       nbformat.v4.new_markdown_cell('Воспроизведение: `.\\.venv\\Scripts\\python.exe -u -X utf8 research/c40_geometry/scripts/investigate_egcc_cases.py`. Исторические координаты и ссылки – в 01_historical_arp. Геодезические методы – Vincenty (1975), https://www.ngs.noaa.gov/PUBS_LIB/inverse.pdf ; Karney (2013), https://doi.org/10.1007/s00190-012-0578-z . Формула временного сценария – наша локальная кинематическая оценка, не результат этих статей.')])
    notebook.metadata['kernelspec']=dict(display_name='Python 3',language='python',name='python3')
    nbformat.write(notebook,STUDY/'notebooks/03_egcc_cases.ipynb')
    print(json.dumps(summary,ensure_ascii=False),flush=True)
    print(cases[['flight_id','typecode','sphere_m','wgs84_m','minutes_to_landing','point_rows','same_board_other_rows_30d']].to_string(index=False),flush=True)


if __name__=='__main__':main()
