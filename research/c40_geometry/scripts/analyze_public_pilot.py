"""Независимая публичная последовательность OpenSky вокруг C40 EPWA."""
from pathlib import Path
import sys
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import nbformat

ROOT=Path(__file__).resolve().parents[3]
STUDY=ROOT/'research/c40_geometry'
sys.path.insert(0,str(ROOT/'europe/scripts'))
from investigate_c40_geometry import distance
from audit_c40_radial_cases import ellipsoid_distance


def main():
    case=pd.read_csv(STUDY/'reports/05_public_monday_candidates.csv').set_index('flight_id').loc['ENT73JH_21100']
    frame=pd.read_csv(STUDY/'data/public_pilot/ENT73JH_21100_state_vectors.csv').sort_values('time')
    next_hour=STUDY/'data/public_pilot/ENT73JH_21100_state_vectors_23.csv'
    if next_hour.exists():
        frame=pd.concat([frame,pd.read_csv(next_hour)],ignore_index=True).sort_values('time')
    frame['utc']=pd.to_datetime(frame.time,unit='s',utc=True)
    t=pd.Timestamp(case.C40_CROSS_TIME)
    landing=pd.Timestamp(case.landing_time)
    frame=frame[frame.utc<=landing+pd.Timedelta(minutes=2)].copy()
    frame['seconds_from_c40']=(frame.utc-t).dt.total_seconds()
    frame['position_age_s']=frame.time-frame.lastposupdate
    frame['contact_age_s']=frame.time-frame.lastcontact
    # Центр восстановлен из метаданных; исторический ARP EPWA еще не проверен.
    frame['sphere_km']=distance(case.fit_lat,case.fit_lon,frame.lat.to_numpy(),frame.lon.to_numpy())
    frame['wgs84_km']=ellipsoid_distance(case.fit_lat,case.fit_lon,frame.lat.to_numpy(),frame.lon.to_numpy())
    frame['distance_to_metadata_point_m']=1000*ellipsoid_distance(case.C40_CROSS_LAT,case.C40_CROSS_LON,frame.lat.to_numpy(),frame.lon.to_numpy())
    local=frame[frame.seconds_from_c40.abs()<=300].copy()
    assert len(local)>0
    crossings=[]
    for model in ['sphere_km','wgs84_km']:
        for i in range(1,len(frame)):
            before,after=frame.iloc[i-1],frame.iloc[i]
            if before[model]>=74.08 and after[model]<74.08:
                weight=(before[model]-74.08)/(before[model]-after[model])
                epoch=float(before.time+weight*(after.time-before.time))
                crossings.append(dict(model=model,before_utc=str(before.utc),after_utc=str(after.utc),
                     before_radius_km=float(before[model]),after_radius_km=float(after[model]),
                     gap_s=int(after.time-before.time),linear_estimate_utc=str(pd.to_datetime(epoch,unit='s',utc=True)),
                     estimated_seconds_before_metadata=float(t.timestamp()-epoch),
                     max_position_age_s=float(max(before.position_age_s,after.position_age_s))))
    nearest=local.iloc[local.seconds_from_c40.abs().argmin()]
    closest=local.iloc[local.distance_to_metadata_point_m.argmin()]
    landing_point=frame.iloc[(frame.utc-landing).abs().argmin()]
    result=dict(flight_id='ENT73JH_21100',icao24='489221',airport='EPWA',rows=len(frame),local_rows=len(local),
        center_status='recovered_center_historical_ARP_pending',c40_utc=str(t),
        metadata_sphere_km=float(case.radius_sphere_km),metadata_wgs84_km=float(case.radius_wgs84_km),
        closest_time_seconds_offset=float(nearest.seconds_from_c40),closest_time_point_distance_m=float(nearest.distance_to_metadata_point_m),
        closest_position_seconds_offset=float(closest.seconds_from_c40),closest_position_distance_m=float(closest.distance_to_metadata_point_m),
        median_observed_interval_s=float(frame.time.diff().median()),crossings=crossings,
        next_hour_downloaded=next_hour.exists(),callsigns=sorted(frame.callsign.dropna().str.strip().unique().tolist()),
        observed_at_c40=dict(lat=float(nearest.lat),lon=float(nearest.lon),sphere_km=float(nearest.sphere_km),wgs84_km=float(nearest.wgs84_km),position_age_s=float(nearest.position_age_s)),
        metadata_coordinates=dict(lat=float(case.C40_CROSS_LAT),lon=float(case.C40_CROSS_LON)),
        near_landing=dict(utc=str(landing_point.utc),time_offset_s=float((landing_point.utc-landing).total_seconds()),radius_wgs84_km=float(landing_point.wgs84_km),baroaltitude_m=float(landing_point.baroaltitude),onground=bool(landing_point.onground)))
    frame.to_csv(STUDY/'data/public_pilot/ENT73JH_21100_geometry.csv',index=False)
    local.to_csv(STUDY/'reports/05_public_pilot_local_points.csv',index=False)
    pd.DataFrame(crossings).to_csv(STUDY/'reports/05_public_pilot_crossings.csv',index=False)
    (STUDY/'reports/05_public_pilot_audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    fig,axs=plt.subplots(1,3,figsize=(18,4.8),layout='constrained')
    for model,label in [('sphere_km','Сфера'),('wgs84_km','WGS84')]:axs[0].plot(local.seconds_from_c40,local[model],'.-',label=label)
    axs[0].axhline(74.08,color='#555',linestyle='--',label='40 NM')
    axs[0].scatter([0],[case.radius_wgs84_km],marker='x',s=90,color='#c34935',label='C40 метаданных, WGS84')
    axs[0].axvline(0,color='#aaa',linewidth=1)
    axs[0].set(xlabel='Время относительно C40 метаданных, с',ylabel='Расстояние от восстановленного центра, км',title='Публичные наблюдения около пересечения')
    axs[0].legend(fontsize=8)
    axs[1].plot(local.seconds_from_c40,local.position_age_s,'.-',label='Возраст позиции')
    axs[1].plot(local.seconds_from_c40,local.contact_age_s,'.-',label='Возраст контакта')
    axs[1].set(xlabel='Время относительно C40 метаданных, с',ylabel='Возраст, с',title='Свежесть state vectors')
    axs[1].legend()
    east=111.32*np.cos(np.radians(case.fit_lat))*(local.lon-case.fit_lon)
    north=111.2*(local.lat-case.fit_lat)
    axs[2].plot(east,north,'.-',label='OpenSky, ±5 минут')
    axs[2].scatter([111.32*np.cos(np.radians(case.fit_lat))*(case.C40_CROSS_LON-case.fit_lon)],
                   [111.2*(case.C40_CROSS_LAT-case.fit_lat)],marker='x',s=100,color='#c34935',label='Точка C40 метаданных')
    axs[2].scatter([0],[0],marker='^',color='#333',label='Восстановленный центр')
    angle=np.linspace(0,2*np.pi,300)
    axs[2].plot(74.08*np.sin(angle),74.08*np.cos(angle),'--',color='#999',label='Схематично 40 NM')
    axs[2].set(xlabel='Восток, км (локальная проекция)',ylabel='Север, км',title='Пространственное сопоставление',aspect='equal')
    axs[2].legend(fontsize=7)
    fig.suptitle('EPWA · ENT73JH · 21.10.2019 · публичный пилот')
    for extension in ['png','svg','pdf']:fig.savefig(STUDY/f'figures/05_public_pilot.{extension}',dpi=300)
    plt.close(fig)
    note='''# Публичная траектория EPWA – первый пилот

Из публичного часового архива OpenSky извлечен борт 489221. Сопоставление выполнено по ICAO24, времени и callsign. Использован восстановленный центр EPWA; исторический ARP еще не подтвержден независимо.

Пересечение 40 NM выделено по соседним наблюдениям снаружи и внутри границы. Линейная интерполяция расстояния во времени дает оценку, а не измеренный момент пересечения. Проверяем интервал и свежесть позиции. Это OpenSky ADS-B; совпадение с исходными CPR EUROCONTROL не предполагается. Результат этого рейса не переносится автоматически на EGCC.

Исходные наблюдения и геометрия сохранены локально в data/public_pilot, которая исключена из Git. Контрольная сумма и URL архива – в 05_public_pilot_download.json. Все записи основной базы сохраняются.
'''
    note+='\nВ момент метаданных существует наблюдение OpenSky с тем же ICAO24 и callsign, но расстояние между позициями около 43 км. Определение радиальной границы само по себе не объясняет пространственное несоответствие. Причина и источник расхождения пока не установлены. Проверку дополнить независимым историческим ARP и другими случаями.\n'
    (STUDY/'reports/05_public_pilot.md').write_text(note+'\n```json\n'+json.dumps(result,ensure_ascii=False,indent=2)+'\n```\n',encoding='utf8')
    nb=nbformat.v4.new_notebook(cells=[nbformat.v4.new_markdown_cell(note),nbformat.v4.new_code_cell("from pathlib import Path\nimport json\nimport pandas as pd\nfrom IPython.display import display, Image\nroot=next(p for p in [Path.cwd(),*Path.cwd().parents] if (p/'research/c40_geometry/reports/05_public_pilot_audit.json').exists())\nstudy=root/'research/c40_geometry'\nprint(json.dumps(json.loads((study/'reports/05_public_pilot_audit.json').read_text()),ensure_ascii=False,indent=2))\ndisplay(pd.read_csv(study/'reports/05_public_pilot_crossings.csv'))\ndisplay(Image(filename=str(study/'figures/05_public_pilot.png')))"),nbformat.v4.new_markdown_cell('Источник: https://opensky-network.org/datasets/ ; описание публичных state vectors: https://opensky-network.org/data/scientific . Геодезическая формула: Vincenty (1975), https://www.ngs.noaa.gov/PUBS_LIB/inverse.pdf . Воспроизведение: scripts/select_public_cases.py → probe_public_snapshots.py → fetch_public_pilot.py → analyze_public_pilot.py. Все результаты поиска относятся к публичным источникам; Trino не использовался.')])
    nb.metadata['kernelspec']=dict(display_name='Python 3',language='python',name='python3')
    nbformat.write(nb,STUDY/'notebooks/05_public_trajectory.ipynb')
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':main()
