"""Таблица погоды у отмеченных попыток для дней всплесков 07_03."""
import html
import json
import re
from datetime import datetime, timezone
from io import StringIO
import pandas as pd
import pyarrow.parquet as pq
import requests
from audit_airports import ROOT, INPUT
from audit_feature_availability import digest


def span(values):
    values=pd.to_numeric(values,errors='coerce').dropna()
    if values.empty:return '—'
    lo,hi=values.min(),values.max()
    return f'{lo:g}' if lo==hi else f'{lo:g}–{hi:g}'


def main():
    egcc=ROOT/'europe/reports/airports/EGCC'
    report=ROOT/'europe/reports/airports/february_2020_weather'
    report.mkdir(parents=True,exist_ok=True)
    rawdir=ROOT/'docs/sources/files/07_03_weather';rawdir.mkdir(parents=True,exist_ok=True)
    maxima=pd.read_csv(egcc/'07_03_other_airport_maxima.csv').head(10)
    peak=pd.read_csv(egcc/'07_03_daily.csv').sort_values(['go_arounds','rows'],ascending=[False,False]).head(5)
    peak.insert(0,'airport','EGCC')
    chosen=pd.concat([maxima[['airport','date','rows','go_arounds','ga_percent']],peak[['airport','date','rows','go_arounds','ga_percent']]]).drop_duplicates(['airport','date']).sort_values(['date','airport'])
    before=digest(INPUT)
    fields=['flight_id','icao24','airport','landing_time','GoA','attempt_times']
    flights=pq.read_table(INPUT,columns=fields,filters=[('airport','in',chosen.airport.unique().tolist()),('GoA','=',True)]).to_pandas(ignore_metadata=True)
    flights['date']=flights.landing_time.dt.strftime('%Y-%m-%d')
    rows=[];matches=[];manifest=[]
    for row in chosen.itertuples():
        group=flights[flights.airport.eq(row.airport)&flights.date.eq(row.date)]
        assert len(group)==row.go_arounds
        events=[]
        for flight in group.itertuples():
            times=pd.to_datetime(flight.attempt_times,utc=True)
            # Последняя попытка относится к завершению сегмента; отмеченные уходы предшествуют ей.
            for i,event in enumerate(times[:-1]):
                events.append(dict(flight_id=flight.flight_id,icao24=flight.icao24,airport=row.airport,landing_date=row.date,attempt=i+1,event_utc=event))
        events=pd.DataFrame(events).sort_values('event_utc')
        start=min(pd.Timestamp(row.date,tz='UTC'),events.event_utc.min().floor('D'))
        end=max(pd.Timestamp(row.date,tz='UTC'),events.event_utc.max().floor('D'))+pd.Timedelta(days=1)
        params=dict(station=row.airport,data='all',year1=start.year,month1=start.month,day1=start.day,year2=end.year,month2=end.month,day2=end.day,tz='Etc/UTC',format='onlycomma',missing='M',trace='T',direct='no',report_type=[3,4])
        url=requests.Request('GET','https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py',params=params).prepare().url
        path=rawdir/f'{row.airport}_{start:%Y-%m-%d}_{end:%Y-%m-%d}.csv'
        if not path.exists():
            response=requests.get(url,timeout=45);response.raise_for_status()
            test=pd.read_csv(StringIO(response.text),comment='#',dtype=str,keep_default_na=False)
            assert {'station','valid','metar'}<=set(test.columns) and len(test)>0
            path.write_bytes(response.content)
        weather=pd.read_csv(path,comment='#',dtype=str,keep_default_na=False)
        weather['metar_utc']=pd.to_datetime(weather.valid,utc=True)
        assert weather.station.eq(row.airport).all()
        assert weather.metar_utc.ge(start).all() and weather.metar_utc.lt(end).all()
        for obs in weather.itertuples():
            token=re.search(r'\b'+row.airport+r'\s+(\d{6})Z\b',obs.metar)
            assert token and token[1]==obs.metar_utc.strftime('%d%H%M')
        wind=weather.metar.str.extract(r'\b(?P<direction>\d{3}|VRB)(?P<speed>\d{2,3})(?:G(?P<gust>\d{2,3}))?(?P<unit>KT|MPS|KMH)\b')
        factor=wind.unit.map({'KT':1.,'MPS':1.943844492,'KMH':.539956803})
        weather['wind_kt']=pd.to_numeric(wind.speed,errors='coerce')*factor
        weather['gust_kt']=pd.to_numeric(wind.gust,errors='coerce')*factor
        weather['wind_direction']=wind.direction
        # Видимость в европейском METAR; 9999 означает не менее 10 км, не ровно 9999 м.
        # Поиск после ветровой группы, включая VRB и порывы.
        vis=weather.metar.str.extract(r'(?:\d{3}|VRB)\d{2,3}(?:G\d{2,3})?(?:KT|MPS|KMH)\s+(?:\d{3}V\d{3}\s+)?(\d{4}|CAVOK)\b',expand=False)
        weather['visibility_token']=vis
        weather['phenomena']=weather.metar.map(lambda s:' '.join(t for t in s.split() if re.fullmatch(r'[+-]?(?:VC)?(?:MI|PR|BC|DR|BL|SH|TS|FZ)?(?:DZ|RA|SN|SG|IC|PL|GR|GS|UP|BR|FG|FU|VA|DU|SA|HZ|PY|PO|SQ|FC|SS|DS){1,3}',t)))
        joined=pd.merge_asof(events,weather.sort_values('metar_utc'),left_on='event_utc',right_on='metar_utc',direction='backward',tolerance=pd.Timedelta(minutes=60))
        joined['age_minutes']=(joined.event_utc-joined.metar_utc).dt.total_seconds()/60
        matches.append(joined)
        v=joined.visibility_token.dropna();numeric=pd.to_numeric(v.where(~v.isin(['9999','CAVOK'])),errors='coerce').dropna()/1000
        visibility=('≥10' if numeric.empty and len(v) else '—')
        if len(numeric):visibility=f'{numeric.min():g}–'+('≥10' if v.isin(['9999','CAVOK']).any() else f'{numeric.max():g}')
        phenomena=sorted({t for s in joined.phenomena.dropna() for t in s.split()})
        result=dict(date=row.date,airport=row.airport,rows=int(row.rows),go_arounds=int(row.go_arounds),ga_percent=round(row.ga_percent,3),
            attempts=len(joined),matched=int(joined.metar_utc.notna().sum()),wind_kt=span(joined.wind_kt.round(1)),gust_kt=span(joined.gust_kt.round(1)),
            visibility_km=visibility,phenomena=', '.join(phenomena) or 'не указаны',max_metar_age_min=round(joined.age_minutes.max(),1),
            directions=', '.join(sorted(joined.wind_direction.dropna().unique())),gust_reports=int(joined.gust_kt.notna().sum()))
        rows.append(result)
        manifest.append(dict(airport=row.airport,landing_date=row.date,url=url,file=str(path.relative_to(ROOT)),sha256=digest(path),reports=len(weather),station_timestamp_verified=True))
        print(json.dumps(result,ensure_ascii=False),flush=True)
    result=pd.DataFrame(rows);result.to_csv(report/'07_03_weather_comparison.csv',index=False)
    events=pd.concat(matches,ignore_index=True);events.to_csv(report/'07_03_weather_comparison_events.csv',index=False)
    assert events.age_minutes.dropna().between(0,60).all() and digest(INPUT)==before
    (report/'07_03_weather_comparison_manifest.json').write_text(json.dumps(dict(input_sha256=before,generated_utc=datetime.now(timezone.utc).isoformat(),selection='five EGCC days with most GoA and ten airport daily count maxima in February; descriptive post hoc selection',weather='latest METAR not after each attempt except final attempt, maximum age 60 minutes',units='wind normalized to kt; original METAR retained',sources=manifest,causality_established=False),ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    labels=['Дата UTC¹','Аэропорт','Все записи','GoA','Доля, %','Ветер, kt','Порывы², kt','Видимость, км','Явления³','METAR / попытки']
    matrix=[[r.date,r.airport,str(r.rows),str(r.go_arounds),f'{r.ga_percent:.3f}',r.wind_kt,r.gust_kt,r.visibility_km,r.phenomena,f'{r.matched}/{r.attempts}'] for r in result.itertuples()]
    note='¹ Группа задана днем посадки UTC; погода взята у каждой предшествующей попытки ухода, включая переход суток. Диапазоны отражают разные сообщения у попыток, а не суточные экстремумы. ² Показаны только явно переданные порывы; отсутствие G не означает нулевой порыв. ³ RA – дождь, SHRA – ливневый дождь, SN – снег, SHSN – снежный ливень, GS – снежная крупа или мелкий град; минус – слабая интенсивность. Видимость 9999/CAVOK показана как ≥10 км. METAR не старше 60 минут; повторное использование сообщения для нескольких попыток допустимо. Источник: вторичный архив Iowa Environmental Mesonet; оригиналы сохранены. Причина уходов не установлена.'
    md='# Погода у отмеченных уходов – февраль 2020\n\n'+'| '+' | '.join(labels)+' |\n| '+' | '.join(['---']*len(labels))+' |\n'
    md+=''.join('| '+' | '.join(row)+' |\n' for row in matrix)+'\n'+note+'\n'
    (report/'07_03_weather_table.md').write_text(md,encoding='utf8')
    body=''.join('<tr>'+''.join(f'<td>{html.escape(cell)}</td>' for cell in row)+'</tr>' for row in matrix)
    page='<!doctype html><html lang="ru"><meta charset="utf-8"><title>Погода у уходов · февраль 2020</title><style>body{margin:0;background:#eef2f6;color:#183044;font:15px Segoe UI,Arial,sans-serif}.wrap{max-width:1400px;margin:48px auto;padding:0 28px}h1{font-size:34px;margin:10px 0}p{line-height:1.65;color:#526778}.tag{color:#a65332;font-weight:700;letter-spacing:2px;font-size:12px}.panel{overflow:auto;background:white;border-radius:16px;box-shadow:0 8px 30px #1232;margin:26px 0}table{width:100%;border-collapse:collapse;white-space:nowrap}th{background:#18394d;color:white;text-align:left;padding:17px 14px;font-size:12px}td{padding:15px 14px;border-bottom:1px solid #e6edf2;font-variant-numeric:tabular-nums}tr:nth-child(even){background:#f7fafc}td:nth-child(2){font-weight:700}td:nth-child(4),td:nth-child(5){color:#ad4d2c;font-weight:700}.note{font-size:13px;max-width:1100px}@media print{body{background:white}.wrap{margin:0;padding:0}h1{font-size:23px}.panel{box-shadow:none}th,td{padding:8px;font-size:10px}}</style><div class="wrap"><div class="tag">APPROACH RISK · 07_03</div><h1>Погода у отмеченных уходов</h1><p>Февраль 2020 · дни всплесков EGCC и максимумы числа GoA в других аэропортах.<br>Ветер и видимость по METAR у попыток. Знаменатели включают обе метки.</p><div class="panel"><table><thead><tr>'+''.join('<th>'+html.escape(x)+'</th>' for x in labels)+'</tr></thead><tbody>'+body+'</tbody></table></div><p class="note">'+html.escape(note)+'</p></div></html>'
    (report/'07_03_weather_table.html').write_text(page,encoding='utf8')


if __name__=='__main__':main()
