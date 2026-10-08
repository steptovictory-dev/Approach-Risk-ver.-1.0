"""Косвенные признаки смешения обозначений ВПП LSGG без очистки данных."""
import json
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from audit_feature_availability import ROOT, digest
from audit_attempt_intervals import INPUT

REPORT=ROOT/'europe/reports/attempts/LSGG'


def main():
    before=digest(INPUT)
    prior=json.loads((ROOT/'europe/reports/attempts/06_02_case_review.json').read_text(encoding='utf8'))
    assert before==prior['input_sha256']
    columns=['flight_id','icao24','callsign','typecode','airport','landing_time','attempt_times',
             'ILS','AP_C_RWY','GoA','n_attempts','C40_CROSS_TIME']
    pieces=[]
    for batch in pq.ParquetFile(INPUT).iter_batches(batch_size=100_000,columns=columns):
        table=pa.Table.from_batches([batch])
        frame=table.to_pandas(ignore_metadata=True)
        pieces.append(frame[frame.airport.eq('LSGG')].copy())
    flights=pd.concat(pieces,ignore_index=True)
    assert len(flights)==prior['airport_rows'] and int(flights.GoA.sum())==prior['airport_go_arounds']
    flights['month']=flights.landing_time.dt.strftime('%Y-%m')
    # Для долей обозначений каждая строка учитывается один раз на обозначение
    labels=flights[['flight_id','icao24','typecode','landing_time','month','GoA','AP_C_RWY','ILS']].copy()
    labels['runway_label']=labels.ILS.apply(lambda x:list(dict.fromkeys(x)))
    labels=labels.explode('runway_label')
    monthly=labels.groupby(['month','runway_label'],dropna=False).agg(observations=('GoA','size'),go_arounds=('GoA','sum')).reset_index()
    monthly['all_airport_rows']=monthly.month.map(flights.groupby('month').size())
    monthly['percent_of_airport_rows']=100*monthly.observations/monthly.all_airport_rows
    cross=labels.groupby(['runway_label','AP_C_RWY'],dropna=False).agg(observations=('GoA','size'),go_arounds=('GoA','sum')).reset_index()
    pairs=[]
    for row in flights[flights.n_attempts.gt(1)].itertuples(index=False):
        times=np.asarray(row.attempt_times,dtype='datetime64[ns]')
        landing=np.datetime64(row.landing_time.to_datetime64(),'ns')
        for index,gap in enumerate(np.diff(times)/np.timedelta64(1,'s')):
            start,end=row.ILS[index],row.ILS[index+1]
            pair='same_label' if start==end else '22R_22_pair' if {start,end}=={'22R','22'} else '04L_04_pair' if {start,end}=={'04L','04'} else 'other_change'
            pairs.append(dict(flight_id=row.flight_id,icao24=row.icao24,callsign=row.callsign,typecode=row.typecode,
                landing_time=row.landing_time,AP_C_RWY=row.AP_C_RWY,n_attempts=row.n_attempts,
                from_runway=start,to_runway=end,pair_group=pair,gap_seconds=float(gap),
                later_attempt_to_landing_seconds=float((landing-times[index+1])/np.timedelta64(1,'s')),
                final_attempt=index+2==len(times)))
    intervals=pd.DataFrame(pairs)
    stats=intervals.groupby(['pair_group','from_runway','to_runway']).agg(
        intervals=('gap_seconds','size'),min_seconds=('gap_seconds','min'),median_seconds=('gap_seconds','median'),
        max_seconds=('gap_seconds','max'),at_most_39_seconds=('gap_seconds',lambda x:int(x.le(39).sum())),
        later_attempt_equals_landing=('later_attempt_to_landing_seconds',lambda x:int(x.eq(0).sum()))).reset_index()
    # Один адрес может встречаться в нескольких группах и не доказывает идентичность борта
    address=labels.groupby('icao24').runway_label.agg(lambda x:sorted(set(x.dropna())))
    overlap=pd.DataFrame([dict(icao24=k,has_22='22' in v,has_22R='22R' in v,has_both_22_labels='22' in v and '22R' in v) for k,v in address.items()])
    types=labels.groupby(['typecode','runway_label'],dropna=False).agg(observations=('GoA','size'),go_arounds=('GoA','sum')).reset_index()
    target=flights[flights.flight_id.eq('TAP948_37820') & flights.icao24.eq('495295')].copy()
    assert len(target)==1
    REPORT.mkdir(parents=True,exist_ok=True)
    target.to_csv(REPORT/'06_03_target_context.csv',index=False)
    output={'monthly':monthly,'runway_cross':cross,'intervals':intervals,'interval_summary':stats,'address_overlap':overlap,'typecode_runways':types}
    for name,data in output.items(): data.to_csv(REPORT/f'06_03_{name}.csv',index=False)
    assert digest(INPUT)==before
    result=dict(input_sha256=before,input_unchanged=True,airport_rows=len(flights),airport_go_arounds=int(flights.GoA.sum()),
        intervals=len(intervals),pair_group_counts=intervals.pair_group.value_counts().to_dict(),
        addresses_with_both_22_labels=int(overlap.has_both_22_labels.sum()),
        addresses_checked=len(overlap),rows_changed=0,rows_deleted=0,
        historical_runway_status_verified=False,trajectory_validation_performed=False,
        AP_C_RWY_is_independent_evidence=False,threshold_39_seconds_is_diagnostic_only=True)
    (REPORT/'06_03_audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    return target,output,result
