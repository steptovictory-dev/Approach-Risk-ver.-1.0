"""Проверка последовательностей попыток на актуальной копии после этапа 05."""
import json
from collections import Counter
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from audit_feature_availability import ROOT, digest

INPUT=ROOT/'europe/data/analytical/05_aircraft_types_reviewed.parquet'
REPORT=ROOT/'europe/reports/attempts'


def main():
    # Сверяем вход с завершенным этапом восстановления типов
    before=digest(INPUT)
    prior=json.loads((ROOT/'europe/reports/aircraft_types/05_05_applied.json').read_text(encoding='utf8'))
    assert before==prior['output_sha256']
    old_intervals=pd.read_csv(ROOT/'europe/reports/preparation/09_high_attempt_intervals.csv')
    cases=old_intervals[old_intervals.gap_minutes.eq(0.65)].copy()
    assert len(cases)==2 and cases.airport.eq('EHAM').all()
    history_path=ROOT/'europe/data/review/09_high_attempt_records.parquet'
    history_hash=digest(history_path)
    archived=pq.read_table(history_path).to_pandas(ignore_metadata=True)
    archived=archived[archived.source_row_position.isin(cases.source_row_position)].copy()
    assert len(archived)==2
    casekeys=set(zip(archived.flight_id,archived.icao24,archived.airport,archived.landing_time))
    foundkeys=set()
    flags=Counter()
    intervals=[]
    distribution=Counter()
    rows=positives=0
    cols=['flight_id','icao24','airport','landing_time','attempt_times','ILS','n_attempts','GoA','C40_CROSS_TIME']
    for batch in pq.ParquetFile(INPUT).iter_batches(batch_size=100_000,columns=cols):
        table=pa.Table.from_batches([batch])
        count=table['n_attempts'].to_numpy()
        ga=table['GoA'].to_numpy()
        lengths=pc.list_value_length(table['attempt_times']).to_numpy()
        runway_lengths=pc.list_value_length(table['ILS']).to_numpy()
        flags['attempt_count_nonpositive']+=int((count<=0).sum())
        flags['attempt_length_mismatch']+=int((lengths!=count).sum())
        flags['ILS_length_mismatch']+=int((runway_lengths!=count).sum())
        flags['GoA_differs_from_more_than_one_attempt']+=int((ga!=(count>1)).sum())
        flags['null_attempt_element']+=pc.list_flatten(table['attempt_times']).null_count
        # Проверяем порядок и интервалы у строк с несколькими элементами
        selected=table.filter(pa.array((count>1)|(lengths>1))).to_pandas(ignore_metadata=True)
        for row in selected.itertuples(index=False):
            distribution[int(row.n_attempts)]+=1
            times=np.asarray(row.attempt_times,dtype='datetime64[ns]')
            gaps=np.diff(times)/np.timedelta64(1,'s')
            flags['descending_attempt_times']+=int((gaps<0).any())
            flags['repeated_adjacent_attempt_time']+=int((gaps==0).any())
            flags['last_attempt_after_landing']+=int(pd.Timestamp(times[-1],tz='UTC')>row.landing_time)
            key=(row.flight_id,row.icao24,row.airport,row.landing_time)
            if key in casekeys: foundkeys.add(key)
            for index,gap in enumerate(gaps):
                intervals.append(dict(flight_id=row.flight_id,icao24=row.icao24,airport=row.airport,
                    landing_time=row.landing_time,n_attempts=row.n_attempts,GoA=row.GoA,
                    attempt_number_from=index+1,attempt_number_to=index+2,
                    time_from=pd.Timestamp(times[index],tz='UTC'),time_to=pd.Timestamp(times[index+1],tz='UTC'),
                    runway_from=row.ILS[index],runway_to=row.ILS[index+1],gap_seconds=float(gap)))
        # Для одиночных попыток достаточно сравнения единственного времени с посадкой
        single=table.filter(pa.array(lengths==1))
        first=pc.list_element(single['attempt_times'],0).to_numpy().astype('datetime64[ns]')
        landing=single['landing_time'].to_numpy().astype('datetime64[ns]')
        flags['last_attempt_after_landing']+=int((first>landing).sum())
        rows+=len(table)
        positives+=int(ga.sum())
    assert rows==prior['rows_after'] and positives==prior['go_arounds_after']
    assert digest(INPUT)==before and digest(history_path)==history_hash
    REPORT.mkdir(parents=True,exist_ok=True)
    gap_frame=pd.DataFrame(intervals)
    gap_frame.to_csv(REPORT/'06_01_intervals.csv',index=False)
    # Порог 39 секунд воспроизводит исторические примеры и не является авиационной нормой
    short=gap_frame[gap_frame.gap_seconds.le(39)].copy()
    short.to_csv(REPORT/'06_01_short_intervals.csv',index=False)
    presence=[]
    for row in archived.itertuples(index=False):
        times=np.asarray(row.attempt_times,dtype='datetime64[ns]')
        first=pd.Timestamp(times[0],tz='UTC')
        reason='retained_in_current_sample' if (row.flight_id,row.icao24,row.airport,row.landing_time) in foundkeys else (
            'c40_missing' if pd.isna(row.C40_CROSS_TIME) else 'c40_not_before_first_attempt' if row.C40_CROSS_TIME>=first else 'not_found_requires_exclusion_history_check')
        presence.append(dict(flight_id=row.flight_id,icao24=row.icao24,airport=row.airport,landing_time=row.landing_time,
            n_attempts=row.n_attempts,GoA=row.GoA,C40_CROSS_TIME=row.C40_CROSS_TIME,
            first_attempt=first,present_in_current_sample=reason=='retained_in_current_sample',status=reason))
    cases_result=pd.DataFrame(presence)
    cases_result.to_csv(REPORT/'06_01_historical_cases.csv',index=False)
    counts=pd.DataFrame([dict(n_attempts=n,rows=v) for n,v in sorted(distribution.items())])
    counts.to_csv(REPORT/'06_01_multiple_attempt_counts.csv',index=False)
    journal=dict(input=str(INPUT.relative_to(ROOT)),input_sha256=before,input_unchanged=True,
        historical_records_sha256=history_hash,historical_records_unchanged=True,
        rows_checked=rows,go_arounds=positives,multiple_attempt_rows=int(counts.rows.sum()),
        more_than_two_attempt_rows=int(counts.loc[counts.n_attempts>2,'rows'].sum()),
        max_attempts=max(distribution),intervals_checked=len(gap_frame),
        min_gap_seconds=float(gap_frame.gap_seconds.min()),
        diagnostic_cutoff_seconds=39,short_interval_rows=len(short),
        short_interval_observations=len(short[['flight_id','icao24','airport','landing_time']].drop_duplicates()),
        historical_cases_retained=int(cases_result.present_in_current_sample.sum()),structural_flags=dict(flags),
        rows_changed=0,rows_deleted=0,trajectory_validation_performed=False,
        short_interval_is_not_error_proof=True,n_attempts_is_not_confirmed_operational_go_around_count=True)
    (REPORT/'06_01_audit.json').write_text(json.dumps(journal,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
    return cases_result,counts,short,journal
