"""Архив METAR EGCC за весь февраль 2020; контекст, не причинный вывод."""
import json
import re
from io import StringIO
from datetime import datetime, timezone
import pandas as pd
import requests
from audit_feature_availability import ROOT, digest


def main():
    report=ROOT/'europe/reports/airports/EGCC'
    folder=ROOT/'docs/sources/files/07_03_weather';folder.mkdir(parents=True,exist_ok=True)
    path=folder/'EGCC_2020-02.csv'
    params=dict(station='EGCC',data='all',year1=2020,month1=2,day1=1,year2=2020,month2=3,day2=1,
        tz='Etc/UTC',format='onlycomma',latlon='yes',elev='yes',missing='M',trace='T',direct='no',report_type=[3,4])
    url=requests.Request('GET','https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py',params=params).prepare().url
    if not path.exists():
        response=requests.get(url,timeout=60);response.raise_for_status()
        check=pd.read_csv(StringIO(response.text),comment='#',dtype=str,keep_default_na=False)
        assert {'station','valid','metar'} <= set(check.columns) and len(check)>0
        path.write_bytes(response.content)
    weather=pd.read_csv(path,comment='#',dtype=str,keep_default_na=False)
    weather['metar_utc']=pd.to_datetime(weather.valid,utc=True)
    assert weather.station.eq('EGCC').all()
    assert weather.metar_utc.ge(pd.Timestamp('2020-02-01',tz='UTC')).all()
    assert weather.metar_utc.lt(pd.Timestamp('2020-03-01',tz='UTC')).all()
    for row in weather.itertuples():
        match=re.search(r'\bEGCC\s+(\d{6})Z\b',row.metar)
        assert match and match[1]==row.metar_utc.strftime('%d%H%M'),row.metar
    # Ветер читается из исходного сообщения; KT, MPS и KMH не смешиваются.
    tokens=weather.metar.str.extract(r'\b(?P<direction>\d{3}|VRB)(?P<speed>\d{2,3})(?:G(?P<gust>\d{2,3}))?(?P<unit>KT|MPS|KMH)\b')
    weather['wind_kt']=pd.to_numeric(tokens.speed.where(tokens.unit.eq('KT')),errors='coerce')
    weather['gust_kt']=pd.to_numeric(tokens.gust.where(tokens.unit.eq('KT')),errors='coerce')
    records=pd.read_csv(report/'07_03_positive_records.csv',dtype={'icao24':str})
    events=[]
    for row in records.itertuples():
        times=json.loads(row.attempts_json)+[row.landing_time]
        for index,value in enumerate(times):
            events.append(dict(flight_id=row.flight_id,icao24=row.icao24,event='landing' if index==len(times)-1 else f'attempt_{index+1}',event_utc=pd.to_datetime(value,utc=True)))
    events=pd.DataFrame(events).sort_values('event_utc')
    joined=pd.merge_asof(events,weather.sort_values('metar_utc'),left_on='event_utc',right_on='metar_utc',direction='backward',tolerance=pd.Timedelta(minutes=60))
    joined['metar_age_minutes']=(joined.event_utc-joined.metar_utc).dt.total_seconds()/60
    joined.to_csv(report/'07_03_weather_at_events.csv',index=False)
    weather['date']=weather.metar_utc.dt.strftime('%Y-%m-%d')
    daily=weather.groupby('date').agg(reports=('metar','size'),max_wind_kt=('wind_kt','max'),max_gust_kt=('gust_kt','max'))
    daily.to_csv(report/'07_03_weather_daily.csv')
    manifest=dict(url=url,file=str(path.relative_to(ROOT)),sha256=digest(path),reports=len(weather),
        verified_utc=datetime.now(timezone.utc).isoformat(),archive='Iowa Environmental Mesonet; secondary archive of transmitted METAR',
        station_and_timestamp_verified=True,event_reports_missing_or_older_than_60min=int(joined.metar_utc.isna().sum()),
        events=len(joined),matching='latest report no later than event, maximum age 60 minutes',
        metoffice_context='https://weather.metoffice.gov.uk/warnings-and-advice/uk-storm-centre/uk-storm-season-2019-20',causality_established=False)
    (report/'07_03_weather_manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    text='# 07_03 – Погодный контекст EGCC\n\n'
    text+=f'Сохранен полный февраль: {len(weather)} исходных сообщений METAR/SPECI. Станция и день/время в тексте сверены с архивной отметкой UTC. Для {len(joined)} событий выбран последний предшествующий отчет не старше 60 минут; без подходящего отчета: {manifest["event_reports_missing_or_older_than_60min"]}.\n\n'
    audit=json.loads((report/'07_03_audit.json').read_text(encoding='utf8'))
    text+='| День посадки с пиком GoA | METAR | Макс. ветер, kt | Макс. порыв, kt |\n| --- | ---: | ---: | ---: |\n'
    for day in audit['top_five_days']:
        r=daily.loc[day];text+=f'| {day} | {int(r.reports)} | {r.max_wind_kt} | {r.max_gust_kt} |\n'
    text+='\nМаксимумы относятся ко всему дню, не обязательно к моменту ухода. Индивидуальные сообщения и возраст приведены в `07_03_weather_at_events.csv`. Полный месячный архив включает дни без пиков. KT извлечено непосредственно из METAR; другие единицы не преобразуются в эти столбцы.\n\n'
    text+='Официальная [хронология Met Office](https://weather.metoffice.gov.uk/warnings-and-advice/uk-storm-centre/uk-storm-season-2019-20) относит Ciara к 8–9 февраля, Dennis к 15–16 февраля. Совпадение по датам дает погодный контекст, но не доказывает причину конкретной разметки GoA. Архив IEM вторичный; исходные сообщения сохранены для проверки. Кроссветер, пределы самолета и причины решения экипажа здесь не установлены.\n'
    (report/'07_03_weather_review.md').write_text(text,encoding='utf8')
    audit['weather_status']='archive_verified_events_matched';(report/'07_03_audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    print(json.dumps(manifest,ensure_ascii=False,indent=2),flush=True)
    print(daily.loc[audit['top_five_days']].to_string(),flush=True)


if __name__=='__main__':main()
