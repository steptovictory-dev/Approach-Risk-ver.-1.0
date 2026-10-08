"""Проверка одной пары по открытым измерениям, независимо от производной таблицы."""
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from audit_feature_availability import ROOT, INPUT, REPORT, digest

SOURCE=ROOT/'europe/data/source/public_pair_134/780b33_raw_state_vectors.parquet'


def main():
    before=digest(INPUT)
    manifest=json.loads((REPORT/'04_07_public_source_manifest.json').read_text(encoding='utf8'))
    assert digest(SOURCE)==manifest['filtered_sha256']
    pair=pd.read_csv(REPORT/'04_05_pair_review.csv').set_index('pair_id').loc[134]
    raw=pd.read_parquet(SOURCE).sort_values('time').copy()
    assert raw.icao24.eq(pair.icao24).all()
    raw['utc']=pd.to_datetime(raw.time,unit='s',utc=True)
    raw['contact_age_seconds']=raw.time-raw.lastcontact
    raw['position_age_seconds']=raw.time-raw.lastposupdate
    # Диагностический допуск свежести не заменяет авиационную норму или правило модели
    raw['fresh_contact_and_position']=(raw.contact_age_seconds.between(0,15)
        & raw.position_age_seconds.between(0,15) & raw.lat.notna() & raw.lon.notna())
    fresh=raw.loc[raw.fresh_contact_and_position]
    events=[]
    for name in ['early_first_attempt','early_landing','late_landing']:
        target=pd.Timestamp(pair[name])
        index=(fresh.utc-target).abs().idxmin()
        record=fresh.loc[index].to_dict()
        record.update(event=name,target_utc=str(target),time_difference_seconds=(record['utc']-target).total_seconds())
        events.append(record)
    event_table=pd.DataFrame(events)
    event_table.to_csv(REPORT/'04_07_event_measurements.csv',index=False)
    # Сохраняем все измерения; устаревшие строки не удаляем из источника
    raw.to_parquet(ROOT/'europe/data/review/04_07_public_pair_measurements.parquet',index=False)
    surface=fresh.loc[fresh.onground.eq(True)]
    early=event_table.loc[event_table.event=='early_landing'].iloc[0]
    assert early.utc==pd.Timestamp(pair.early_landing) and not early.onground
    assert early.geoaltitude>1000 and early.vertrate>0
    assert len(surface)>0
    assert digest(INPUT)==before
    summary=dict(pair_id=134,icao24=pair.icao24,airport=pair.airport,
        raw_rows=len(raw),fresh_contact_and_position_rows=len(fresh),
        maximum_raw_time_gap_seconds=float(raw.time.diff().max()),
        early_landing_time=pair.early_landing,late_landing_time=pair.late_landing,
        early_event_geoaltitude_metres=float(early.geoaltitude),
        early_event_vertical_rate_metres_per_second=float(early.vertrate),
        early_event_onground=False,first_fresh_surface_time=str(surface.utc.min()),
        barometric_altitude_above_20000m_rows=int((raw.baroaltitude>20000).sum()),
        barometric_outlier_rule='Diagnostic display count only; no raw value changed',
        conclusion='early landing_time occurs during observed airborne climb; late time close to surface transition',
        exact_touchdown_time_verified=False,operational_go_around_count_verified=False,
        C40_geometric_crossing_verified=False,
        rows_changed=0,rows_deleted=0,input_unchanged=True,input_sha256=before)
    (REPORT/'04_07_public_pair_audit.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps(summary,ensure_ascii=False,indent=2))
    return raw,event_table


def plot(raw):
    pair=pd.read_csv(REPORT/'04_05_pair_review.csv').set_index('pair_id').loc[134]
    section=raw.loc[raw.utc.between(pd.Timestamp('2019-03-04T04:30:00Z'),pd.Timestamp('2019-03-04T05:18:00Z'))].copy()
    # На графике показываем GNSS-высоту и вертикальную скорость, не высоту над ВПП
    fig,axes=plt.subplots(2,1,figsize=(12,7),sharex=True)
    axes[0].plot(section.utc,section.geoaltitude,label='Геометрическая высота, исходные данные',color='#477ca8')
    axes[1].plot(section.utc,section.vertrate,label='Вертикальная скорость',color='#477ca8')
    axes[1].axhline(0,color='gray',linewidth=.7)
    surface=section.loc[section.onground.eq(True)&section.fresh_contact_and_position]
    axes[0].scatter(surface.utc,np.zeros(len(surface)),color='#559944',marker='|',label='Свежий onground=True (условный уровень 0)')
    for ax in axes:
        for field,label,color in [('early_landing','Ранний landing_time','#cc7941'),('late_landing','Поздний landing_time','#995299')]:
            ax.axvline(pd.Timestamp(pair[field]),color=color,linestyle='--',label=label)
        ax.grid(alpha=.2)
    axes[0].set(ylabel='Геометрическая высота, м',title='LFPG, 04.03.2019: пара CES773 по публичным измерениям OpenSky')
    axes[1].set(ylabel='Вертикальная скорость, м/с',xlabel='Время UTC')
    axes[1].xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    axes[0].legend(fontsize=8)
    fig.autofmt_xdate()
    fig.tight_layout()
    folder=ROOT/'europe/figures/04_analytical_sample'
    for suffix in ['png','svg']:
        fig.savefig(folder/f'04_07_public_pair_134.{suffix}',dpi=160)
    return fig
