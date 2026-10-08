"""Архивные METAR за дни трех случаев 06_05; исходные сообщения сохраняются."""
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from io import StringIO
import json
import re

import pandas as pd
import requests

from audit_feature_availability import ROOT, digest

REPORT = ROOT / "europe/reports/attempts"
RAW = ROOT / "docs/sources/files/06_05_weather"


def fetch(row):
    day=pd.Timestamp(row.landing_time).floor("D")
    end=day+pd.Timedelta(days=1)
    params=dict(station=row.airport,data="all",year1=day.year,month1=day.month,day1=day.day,year2=end.year,month2=end.month,day2=end.day,tz="Etc/UTC",format="onlycomma",latlon="yes",elev="yes",missing="M",trace="T",direct="no",report_type=[3,4])
    path=RAW/f"{row.airport}_{day:%Y-%m-%d}.csv"
    prepared=requests.Request("GET","https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py",params=params).prepare()
    if not path.exists():
        response=requests.get(prepared.url,timeout=45)
        response.raise_for_status()
        frame=pd.read_csv(StringIO(response.text),comment="#",dtype=str,keep_default_na=False)
        assert {"station","valid","metar"}<=set(frame.columns), response.text[:200]
        assert len(frame)>0, f"Нет сообщений: {row.airport} {day}"
        path.write_bytes(response.content)
    frame=pd.read_csv(path,comment="#",dtype=str,keep_default_na=False)
    frame["observation_utc"]=pd.to_datetime(frame.valid,utc=True)
    assert frame.station.eq(row.airport).all()
    assert frame.observation_utc.ge(day).all() and frame.observation_utc.lt(end).all()
    # Сверяем станцию и день/время непосредственно с исходным сообщением
    for obs in frame.itertuples(index=False):
        match=re.search(r"(?:^|\s)"+row.airport+r"\s+(\d{6})Z(?:\s|$)",obs.metar)
        assert match and match[1]==obs.observation_utc.strftime("%d%H%M"), obs.metar
    frame=frame.sort_values("observation_utc",kind="stable")
    start=pd.Timestamp(row.first_attempt);stop=pd.Timestamp(row.landing_time)
    window=frame[frame.observation_utc.between(start-pd.Timedelta(hours=1),stop+pd.Timedelta(hours=1))].copy()
    window.insert(0,"flight_id",row.flight_id)
    manifest=dict(flight_id=row.flight_id,airport=row.airport,date_utc=str(day.date()),url=prepared.url,file=str(path.relative_to(ROOT)),sha256=digest(path),reports=len(frame),window_reports=len(window),first_observation_utc=frame.observation_utc.min().isoformat(),last_observation_utc=frame.observation_utc.max().isoformat(),station_and_metar_timestamp_verified=True)
    return frame,window,manifest


def main():
    RAW.mkdir(parents=True,exist_ok=True)
    cases=pd.read_csv(REPORT/"06_05_case_summary.csv",dtype={"icao24":str})
    cases=cases[cases.flight_id.isin(["FIN1AB_178077","DLH3TV_118179","AFL2306_70497"])]
    assert len(cases)==3
    with ThreadPoolExecutor(max_workers=3) as pool:
        results=list(pool.map(fetch,cases.itertuples(index=False)))
    all_windows=[];events=[];manifest=[]
    record=pq_records()
    for row,(frame,window,entry) in zip(cases.itertuples(index=False),results):
        all_windows.append(window);manifest.append(entry)
        times=record[row.flight_id]+[pd.Timestamp(row.landing_time)]
        for index,event in enumerate(times):
            previous=frame[frame.observation_utc.le(event)]
            assert len(previous), f"Нет предшествующего METAR: {row.flight_id}"
            obs=previous.iloc[-1]
            age=(event-obs.observation_utc).total_seconds()/60
            assert age<=60, f"Устаревший METAR: {row.flight_id}, {age} минут"
            events.append(dict(flight_id=row.flight_id,airport=row.airport,event="landing" if index==len(times)-1 else f"attempt_{index+1}",event_utc=event,metar_utc=obs.observation_utc,age_minutes=age,metar=obs.metar))
        print(row.flight_id,entry["reports"],"reports",flush=True)
        for obs in window.itertuples(index=False):print(obs.observation_utc,obs.metar,flush=True)
    pd.concat(all_windows,ignore_index=True).to_csv(REPORT/"06_05_weather_windows.csv",index=False)
    pd.DataFrame(events).to_csv(REPORT/"06_05_weather_at_attempts.csv",index=False)
    journal=dict(checked_date="2026-10-09",downloaded_or_verified_utc=datetime.now(timezone.utc).isoformat(),archive="Iowa Environmental Mesonet, secondary archive of transmitted METAR",source_documentation="https://mesonet.agron.iastate.edu/request/download.phtml",timezone="UTC",full_days_requested=True,window_before_after_hours=1,selection="latest observation not after each event; no interpolation",cases=manifest,causality_established=False,weather_added_to_model=False)
    (REPORT/"06_05_weather_manifest.json").write_text(json.dumps(journal,ensure_ascii=False,indent=2)+"\n",encoding="utf8")


def pq_records():
    import pyarrow.parquet as pq
    frame=pq.read_table(ROOT/"europe/data/review/06_05_three_attempt_records.parquet").to_pandas(ignore_metadata=True)
    return {row.flight_id:[pd.Timestamp(value,tz="UTC") for value in row.attempt_times] for row in frame.itertuples(index=False)}


if __name__=="__main__":
    main()
