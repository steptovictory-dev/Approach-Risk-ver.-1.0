"""Выбор диагностических случаев на понедельниках для открытых выгрузок."""
from pathlib import Path
import pandas as pd

ROOT=Path(__file__).resolve().parents[3]
STUDY=ROOT/'research/c40_geometry'


def main():
    source=ROOT/'europe/reports/airports/c40_geometry/07_08_radial_records.csv'
    cases=pd.read_csv(source,parse_dates=['C40_CROSS_TIME','landing_time'])
    cases['weekday_utc']=cases.C40_CROSS_TIME.dt.day_name()
    candidates=cases[(cases.C40_CROSS_TIME.dt.dayofweek==0)&(cases.C40_CROSS_TIME.dt.year<=2022)].copy()
    candidates['snapshot_coverage_status']='date_matches_monday_file_not_yet_verified'
    candidates=candidates.sort_values('abs_deviation_km',ascending=False)
    candidates.to_csv(STUDY/'reports/05_public_monday_candidates.csv',index=False)
    print('Всего диагностических записей:',len(cases),'понедельничных кандидатов:',len(candidates),flush=True)
    print(candidates[['flight_id','icao24','airport','GoA','C40_CROSS_TIME','abs_deviation_km']].to_string(index=False),flush=True)


if __name__=='__main__':main()
