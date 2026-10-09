"""Сравнение выбранного случая с соседними посадками EPWA в тех же открытых часах."""
from pathlib import Path
import sys
import json
import tarfile
import hashlib
import pandas as pd
import numpy as np
import pyarrow.parquet as pq
import nbformat
from notebook_output import write_overview
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[3]
STUDY=ROOT/'research/c40_geometry'
sys.path.insert(0,str(ROOT/'europe/scripts'))
from audit_c40_radial_cases import ellipsoid_distance
from investigate_c40_geometry import distance


def main():
    fields=['flight_id','icao24','airport','GoA','landing_time','C40_CROSS_TIME','C40_CROSS_LAT','C40_CROSS_LON','C40_BEARING','n_attempts','attempt_times']
    start=pd.Timestamp('2019-10-21 22:00:00',tz='UTC');end=start+pd.Timedelta(hours=1)
    frame=pq.read_table(ROOT/'europe/data/analytical/06_05_cases_excluded.parquet',columns=fields,filters=[('airport','in',['EPWA','EGCC','EDDF','EGLL'])]).to_pandas(ignore_metadata=True)
    selected=frame[(frame.C40_CROSS_TIME>=start)&(frame.C40_CROSS_TIME<end)].sort_values('C40_CROSS_TIME').copy()
    assert len(selected)>1
    ids=set(selected.icao24)
    observations=[]
    for hour in ['22','23']:
        path=STUDY/f'data/public_pilot/states_2019-10-21-{hour}.csv.tar'
        manifest=json.loads((STUDY/f'reports/05_public_pilot_download{ "" if hour=="22" else "_23"}.json').read_text())
        sha=hashlib.sha256()
        with path.open('rb') as stream:
            for block in iter(lambda:stream.read(8*1024*1024),b''):sha.update(block)
        assert sha.hexdigest()==manifest['sha256']
        with tarfile.open(path) as archive:
            for member in archive.getmembers():
                if member.isfile() and member.name.endswith('.csv.gz'):
                    for chunk in pd.read_csv(archive.extractfile(member),compression='gzip',chunksize=100000,dtype={'icao24':'string'},low_memory=False):
                        hit=chunk[chunk.icao24.isin(ids)].copy()
                        if len(hit):observations.append(hit)
    observed=pd.concat(observations,ignore_index=True).sort_values('time')
    observed.to_csv(STUDY/'data/public_pilot/EPWA_control_state_vectors.csv',index=False)
    centers=pd.read_csv(ROOT/'europe/reports/airports/c40_geometry/07_07_best_formula.csv').set_index('airport')
    results=[];snippets=[]
    for case in selected.itertuples():
        center=centers.loc[case.airport]
        t=pd.Timestamp(case.C40_CROSS_TIME).timestamp();landing=pd.Timestamp(case.landing_time).timestamp()
        track=observed[(observed.icao24==case.icao24)&(observed.time>=t-300)&(observed.time<=landing+120)].copy()
        result=dict(flight_id=case.flight_id,airport=case.airport,icao24=case.icao24,GoA=bool(case.GoA),n_attempts=case.n_attempts,c40_utc=str(case.C40_CROSS_TIME),landing_utc=str(case.landing_time),observed_rows=len(track),center_status='recovered_center',fit_lat=float(center.fit_lat),fit_lon=float(center.fit_lon))
        if track.empty:
            result['status']='aircraft_not_found';results.append(result);continue
        track['utc']=pd.to_datetime(track.time,unit='s',utc=True)
        track['sphere_km']=distance(center.fit_lat,center.fit_lon,track.lat.to_numpy(),track.lon.to_numpy())
        track['distance_to_metadata_m']=1000*ellipsoid_distance(case.C40_CROSS_LAT,case.C40_CROSS_LON,track.lat.to_numpy(),track.lon.to_numpy())
        track['case_flight_id']=case.flight_id
        snippets.append(track)
        nearest=track.iloc[(track.time-t).abs().argmin()]
        landing_point=track.iloc[(track.time-landing).abs().argmin()]
        before=track[track.time<=t].tail(1)
        after=track[track.time>=t].head(1)
        interpolated_difference=None;interpolation_gap=None
        if len(before) and len(after):
            p0,p1=before.iloc[0],after.iloc[0]
            interpolation_gap=float(p1.time-p0.time)
            weight=0 if interpolation_gap==0 else (t-p0.time)/interpolation_gap
            ilat=float(p0.lat+weight*(p1.lat-p0.lat))
            ilon=float(p0.lon+weight*(p1.lon-p0.lon))
            interpolated_difference=float(1000*ellipsoid_distance(case.C40_CROSS_LAT,case.C40_CROSS_LON,np.array([ilat]),np.array([ilon]))[0])
        meta_r=float(distance(center.fit_lat,center.fit_lon,case.C40_CROSS_LAT,case.C40_CROSS_LON))
        expected_callsign=case.flight_id.rsplit('_',1)[0]
        result.update(status='matched_icao24_time',expected_callsign=expected_callsign,
           observed_callsign=str(nearest.callsign).strip(),callsign_matches=str(nearest.callsign).strip()==expected_callsign,
           nearest_time_offset_s=float(nearest.time-t),position_difference_m=float(nearest.distance_to_metadata_m),
           interpolated_position_difference_m=interpolated_difference,interpolation_gap_s=interpolation_gap,
           metadata_sphere_km=meta_r,observed_sphere_km=float(nearest.sphere_km),metadata_lat=case.C40_CROSS_LAT,metadata_lon=case.C40_CROSS_LON,
           observed_lat=float(nearest.lat),observed_lon=float(nearest.lon),position_age_s=float(nearest.time-nearest.lastposupdate),
           near_landing_offset_s=float(landing_point.time-landing),near_landing_radius_km=float(landing_point.sphere_km),
           near_landing_baroaltitude_m=float(landing_point.baroaltitude),near_landing_onground=bool(landing_point.onground))
        results.append(result)
    result=pd.DataFrame(results)
    result.to_csv(STUDY/'reports/06_epwa_control_comparison.csv',index=False)
    pd.concat(snippets,ignore_index=True).to_csv(STUDY/'data/public_pilot/EPWA_control_geometry.csv',index=False)
    good=result[result.status=='matched_icao24_time']
    summary=dict(selected_cases=len(selected),found_cases=len(good),call_sign_matches=int(good.callsign_matches.sum()),
         interpolated_difference_gt_1km=int((good.interpolated_position_difference_m>1000).sum()),
         median_interpolated_difference_m=float(good.interpolated_position_difference_m.median()),
         difference_gt_1km=int((good.position_difference_m>1000).sum()),median_position_difference_m=float(good.position_difference_m.median()),
         min_position_difference_m=float(good.position_difference_m.min()),max_position_difference_m=float(good.position_difference_m.max()),
         selection='All retained EPWA, EGCC, EDDF, EGLL cases with C40 in 2019-10-21 22:00–23:00 UTC; same public hours, not random or representative.',
         airports=sorted(selected.airport.unique().tolist()))
    (STUDY/'reports/06_epwa_control_audit.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    fig,axs=plt.subplots(1,2,figsize=(14,5),layout='constrained')
    axs[0].barh(good.flight_id,good.interpolated_position_difference_m/1000,color=['#c55b35' if f=='ENT73JH_21100' else '#286b9e' for f in good.flight_id])
    axs[0].set(xlabel='Расстояние до интерполированной позиции C40, км',title='EPWA и EGCC, один час C40')
    axs[0].set_xscale('log')
    for record in good.itertuples():
        center=centers.loc[record.airport]
        mx=111.32*np.cos(np.radians(center.fit_lat))*(record.metadata_lon-center.fit_lon)
        my=111.2*(record.metadata_lat-center.fit_lat)
        ox=111.32*np.cos(np.radians(center.fit_lat))*(record.observed_lon-center.fit_lon)
        oy=111.2*(record.observed_lat-center.fit_lat)
        axs[1].plot([mx,ox],[my,oy],'-',color='#aaa',linewidth=.8)
        axs[1].scatter([mx],[my],marker='x',color='#c55b35')
        axs[1].scatter([ox],[oy],s=15,color='#286b9e')
    axs[1].scatter([],[],marker='x',color='#c55b35',label='Метаданные C40')
    axs[1].scatter([],[],color='#286b9e',label='Позиции OpenSky около C40')
    axs[1].set(xlabel='Восток, км',ylabel='Север, км',title='Локальная проекция относительно центра',aspect='equal')
    axs[1].legend(fontsize=8)
    fig.suptitle('21.10.2019 · контрольное сравнение открытых источников')
    for ext in ['png','svg','pdf']:fig.savefig(STUDY/f'figures/06_epwa_controls.{ext}',dpi=300)
    plt.close(fig)
    note='''# Контрольное сопоставление четырех аэропортов

В EPWA в выбранном часе есть только исходный случай. Для контроля выбраны все сохраненные посадки EPWA, EGCC, EDDF и EGLL с C40 в 22-м часе UTC 21.10.2019, без отбора по величине отклонения. Использованы уже скачанные публичные часы 22 и 23 UTC. Это локальная контрольная группа, а не случайная выборка аэропортов.

Сопоставление: ICAO24, время и callsign; дополнительно показано положение около времени посадки. onground и барометрическая высота не используются сами по себе как подтверждение касания. Барометрическая высота приведена в метрах как в источнике, не AGL. Исторический ARP EPWA еще не подтвержден.

Временная привязка использует ближайший state vector; его сдвиг от времени C40 и возраст позиции показаны отдельно. Дополнительно координаты линейно интерполируются между соседними наблюдениями к точному времени C40. Это оценка, а не наблюденная позиция; интервал интерполяции приведен отдельно. На графике расстояний логарифмическая шкала.

Изменение центра не меняет геодезическое расстояние между координатами двух источников. Поэтому независимая проверка ARP важна для радиусов, но не объясняет само межисточниковое позиционное расхождение. В выбранном часе из четырех предусмотренных аэропортов записи есть только в EPWA и EGCC; нулевое число в других аэропортах не считается проверкой отсутствия посадок в исходной базе.

Ни координаты метаданных, ни OpenSky не объявляются истинными без дальнейшей проверки. Все исходные значения и строки сохраняются. Источник: https://opensky-network.org/datasets/ ; SHA256 двух архивов – в отчетах 05_public_pilot_download*.json.
'''
    (STUDY/'reports/06_epwa_controls.md').write_text(note+'\n```json\n'+json.dumps(summary,ensure_ascii=False,indent=2)+'\n```\n',encoding='utf8')
    nb=nbformat.v4.new_notebook(cells=[nbformat.v4.new_markdown_cell(note),nbformat.v4.new_code_cell("from pathlib import Path\nimport pandas as pd\nfrom IPython.display import display, Image\nroot=next(p for p in [Path.cwd(),*Path.cwd().parents] if (p/'research/c40_geometry/reports/06_epwa_control_comparison.csv').exists())\nstudy=root/'research/c40_geometry'\ntable=pd.read_csv(study/'reports/06_epwa_control_comparison.csv')\ndisplay(table)\ndisplay(Image(filename=str(study/'figures/06_epwa_controls.png')))"),nbformat.v4.new_markdown_cell('Воспроизведение: `.\\.venv\\Scripts\\python.exe -u -X utf8 research/c40_geometry/scripts/compare_public_epwa_controls.py`. Крупное расхождение – диагностическое наблюдение, не правило удаления. Следующая проверка: другая дата, исходное объединение полей, возможность систематической ошибки координат.')])
    nb.metadata['kernelspec']=dict(display_name='Python 3',language='python',name='python3')
    write_overview(nb,STUDY/'notebooks/06_epwa_controls.ipynb')
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)
    print(good[['flight_id','airport','callsign_matches','nearest_time_offset_s','position_difference_m','interpolated_position_difference_m','interpolation_gap_s']].to_string(index=False),flush=True)


if __name__=='__main__':main()
