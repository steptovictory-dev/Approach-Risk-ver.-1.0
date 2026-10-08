"""Извлечение опубликованных траекторных точек и подготовка независимых SQL-проверок."""
from pathlib import Path
import hashlib
import json
import re
import zipfile
import pandas as pd
import matplotlib.pyplot as plt
from audit_feature_availability import ROOT, INPUT, REPORT, digest

ARCHIVE = ROOT / 'archive/2019_experiment/data/raw/validation_sample.zip'
DATA = ROOT / 'europe/data/review/04_06_trajectory_checks'
QUERIES = ROOT / 'europe/scripts/queries/04_06'


def main():
    before = digest(INPUT)
    pairs = pd.read_csv(REPORT / '04_05_pairs_with_goa.csv')
    files = {'EDDM':'EDDM_26R.html','LFPG':'LFPG_09L.html','EDDF':'EDDF_25R.html'}
    DATA.mkdir(parents=True,exist_ok=True)
    QUERIES.mkdir(parents=True,exist_ok=True)
    result = []
    with zipfile.ZipFile(ARCHIVE) as archive:
        for pair in pairs.itertuples(index=False):
            member = 'validation_sample/'+files[pair.airport]
            content = archive.read(member)
            text = content.decode('utf8')
            points = []
            for match in re.finditer(r'Plotly\.newPlot\(\s*"[^"]+"\s*,\s*',text):
                traces,_ = json.JSONDecoder().raw_decode(text[match.end():])
                for trace in traces:
                    for row in trace.get('customdata',[]):
                        if isinstance(row,list) and len(row)==6 and row[2]==pair.early_flight_id:
                            points.append(row)
            assert points, pair.early_flight_id
            frame = pd.DataFrame(points,columns=['lat','lon','source_flight_id','source_altitude','time','source_point_index'])
            frame.time = pd.to_datetime(frame.time,utc=True)
            frame = frame.drop_duplicates().sort_values('time')
            assert frame.time.max()==pd.Timestamp(pair.early_landing)
            frame.to_parquet(DATA/f'{pair.airport}_published_points.parquet',index=False)
            # Запрос ограничен адресом, интервалом и часовыми разделами согласно документации OpenSky
            start = pd.Timestamp(pair.C40_CROSS_TIME)-pd.Timedelta(minutes=10)
            end = pd.Timestamp(pair.late_landing)+pd.Timedelta(minutes=20)
            hours = range(int(start.floor('h').timestamp()),int(end.floor('h').timestamp())+1,3600)
            query = (
                '-- UTC; raw state vectors retained, stale records evaluated separately\n'
                'SELECT time, icao24, lat, lon, velocity, heading, vertrate, callsign,\n'
                '       onground, baroaltitude, geoaltitude, lastposupdate, lastcontact\n'
                'FROM minio.osky.state_vectors_data4\n'
                f"WHERE icao24 = '{pair.icao24}'\n"
                f"  AND hour IN ({', '.join(map(str,hours))})\n"
                f'  AND time BETWEEN {int(start.timestamp())} AND {int(end.timestamp())}\n'
                'ORDER BY time;\n')
            (QUERIES/f'{pair.airport}_{pair.icao24}.sql').write_text(query,encoding='utf8')
            result.append(dict(pair_id=pair.pair_id,airport=pair.airport,icao24=pair.icao24,
                early_flight_id=pair.early_flight_id,late_flight_id=pair.late_flight_id,
                published_points=len(frame),first_point=str(frame.time.min()),last_point=str(frame.time.max()),
                minimum_source_altitude=float(frame.source_altitude.min()),
                last_source_altitude=float(frame.source_altitude.iloc[-1]),
                altitude_unit='not established from HTML; retained as source values',
                archive_member=member,member_sha256=hashlib.sha256(content).hexdigest(),
                source_independent_of_labeling_pipeline=False,
                independent_state_vectors_downloaded=False,query_start=str(start),query_end=str(end),
                status='published_points_extracted; independent_check_pending'))
    assert digest(INPUT)==before
    summary=dict(input_sha256=before,input_unchanged=True,rows_changed=0,rows_deleted=0,
                 archive=str(ARCHIVE.relative_to(ROOT)),archive_sha256=digest(ARCHIVE),cases=result,
                 trino_documentation='https://openskynetwork.github.io/opensky-api/trino.html',
                 trajectories_independently_verified=0)
    (REPORT/'04_06_check_preparation.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    pd.DataFrame(result).to_csv(REPORT/'04_06_case_status.csv',index=False)
    print(json.dumps(summary,ensure_ascii=False,indent=2))
    return pd.DataFrame(result)


def plot_case(airport):
    frame=pd.read_parquet(DATA/f'{airport}_published_points.parquet')
    pair=pd.read_csv(REPORT/'04_05_pairs_with_goa.csv').set_index('airport').loc[airport]
    fig,ax=plt.subplots(figsize=(11,4))
    ax.plot(frame.time,frame.source_altitude,marker='.',markersize=3,label='Опубликованные точки раннего сегмента')
    for field,label,color in [('early_first_attempt','Первая попытка','#cc7941'),
            ('early_last_attempt','Последняя попытка ранней записи','#477ca8'),
            ('early_landing','landing_time ранней записи','#888888'),
            ('late_landing','landing_time поздней записи','#995299')]:
        ax.axvline(pd.Timestamp(pair[field]),color=color,linestyle='--',label=label)
    ax.set(title=f'{airport}: {pair.early_flight_id} – опубликованный сегмент',
           xlabel='Время UTC',ylabel='altitude: исходные значения, единица не подтверждена')
    ax.legend(fontsize=8)
    ax.grid(alpha=.2)
    fig.autofmt_xdate()
    fig.tight_layout()
    folder=ROOT/'europe/figures/04_analytical_sample'
    for extension in ['png','svg']:
        fig.savefig(folder/f'04_06_{airport}_published_segment.{extension}',dpi=160)
    return fig
