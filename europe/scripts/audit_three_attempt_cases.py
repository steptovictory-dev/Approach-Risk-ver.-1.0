"""Разбор 141 записи с тремя попытками на актуальной выборке после 06_04."""
import json
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from audit_feature_availability import ROOT, digest

INPUT = ROOT / "europe/data/analytical/06_04_profiles_excluded.parquet"
REPORT = ROOT / "europe/reports/attempts"
EXTRACT = ROOT / "europe/data/review/06_05_three_attempt_records.parquet"


def main():
    prior=json.loads((REPORT / "06_04_profile_exclusion_applied.json").read_text(encoding="utf8"))
    before=digest(INPUT)
    assert before==prior["output_sha256"]
    parts=[];total=positive=more_two=0
    for batch in pq.ParquetFile(INPUT).iter_batches(batch_size=100_000):
        table=pa.Table.from_batches([batch])
        total+=len(table);positive+=pc.sum(pc.cast(table["GoA"],pa.int64())).as_py()
        more_two+=pc.sum(pc.cast(pc.greater(table["n_attempts"],2),pa.int64())).as_py()
        selected=table.filter(pc.equal(table["n_attempts"],3))
        if len(selected):parts.append(selected)
    assert total==prior["rows_after"] and positive==prior["go_arounds_after"] and more_two==143
    records=pa.concat_tables(parts)
    assert len(records)==141
    EXTRACT.parent.mkdir(parents=True,exist_ok=True)
    pq.write_table(records,EXTRACT,compression="zstd")
    assert records.equals(pq.read_table(EXTRACT))
    df=records.to_pandas(ignore_metadata=True)
    # Сравнение с фильтром позывных в опубликованном ноутбуке авторов – не доказательство назначения
    author=json.loads((ROOT/"docs/sources/files/lsgg_label_provenance/OSN_paper_code.ipynb").read_text(encoding="utf8"))
    filter_cell="".join(author["cells"][5]["source"])
    prefixes=("FCK","GAF","PPU","CFL","ECK","CALIBRAT","PHLAB","CALIBRA","ENF")
    for prefix in prefixes:assert f'"{prefix}"' in filter_cell
    summaries=[];intervals=[]
    for row in df.itertuples(index=False):
        times=np.asarray(row.attempt_times,dtype="datetime64[ns]")
        gaps=np.diff(times)/np.timedelta64(1,"s")
        assert len(times)==row.n_attempts==len(row.ILS) and bool(row.GoA) and not np.isnat(times).any() and (gaps>0).all()
        landing=np.datetime64(row.landing_time.to_datetime64(),"ns")
        last_gap=float((landing-times[-1])/np.timedelta64(1,"s"))
        assert last_gap>=0
        info=dict(flight_id=row.flight_id,icao24=row.icao24,callsign=row.callsign,registration=row.registration,typecode=row.typecode,airport=row.airport,landing_time=row.landing_time,n_attempts=row.n_attempts,GoA=bool(row.GoA),market_segment=row.market_segment,runway_sequence="|".join(row.ILS),first_attempt=pd.Timestamp(times[0],tz="UTC"),last_attempt=pd.Timestamp(times[-1],tz="UTC"),attempt_window_minutes=float((times[-1]-times[0])/np.timedelta64(1,"m")),min_gap_minutes=float(gaps.min()/60),max_gap_minutes=float(gaps.max()/60),last_attempt_to_landing_seconds=last_gap,author_callsign_prefix_filter_match=any(row.callsign.startswith(prefix) for prefix in prefixes),mission_verified=False)
        summaries.append(info)
        for i,gap in enumerate(gaps):intervals.append(dict(flight_id=row.flight_id,icao24=row.icao24,airport=row.airport,landing_time=row.landing_time,attempt_from=i+1,attempt_to=i+2,from_runway=row.ILS[i],to_runway=row.ILS[i+1],time_from=pd.Timestamp(times[i],tz="UTC"),time_to=pd.Timestamp(times[i+1],tz="UTC"),gap_seconds=float(gap)))
    summary=pd.DataFrame(summaries);intervals=pd.DataFrame(intervals)
    summary["same_runway_all_attempts"] = summary.runway_sequence.apply(lambda value: len(set(value.split("|")))==1)
    summary["has_22R_or_04L"] = summary.runway_sequence.apply(lambda value: any(label in {"22R","04L"} for label in value.split("|")))
    summary["last_attempt_to_landing_over_60_seconds"] = summary.last_attempt_to_landing_seconds.gt(60)
    airport = summary.groupby("airport").agg(records=("flight_id","size"),prefix_candidates=("author_callsign_prefix_filter_match","sum"),changed_runway=("same_runway_all_attempts",lambda values:int((~values).sum())),max_window_minutes=("attempt_window_minutes","max")).reset_index().sort_values("records",ascending=False)
    addresses = summary.groupby(["icao24","registration","typecode"],dropna=False).agg(records=("flight_id","size"),airports=("airport",lambda values:"|".join(sorted(set(values))))).reset_index().sort_values("records",ascending=False)
    patterns = summary.groupby(["airport","runway_sequence"],dropna=False).size().rename("records").reset_index().sort_values("records",ascending=False)
    summary["among_two_shortest_gaps"] = summary.index.isin(summary.nsmallest(2,"min_gap_minutes").index)
    candidates = summary[summary.author_callsign_prefix_filter_match | summary.last_attempt_to_landing_over_60_seconds | summary.among_two_shortest_gaps].copy()
    for name, frame in dict(airports=airport,addresses=addresses,patterns=patterns,review_candidates=candidates,longest_windows=summary.nlargest(10,"attempt_window_minutes"),shortest_gaps=summary.nsmallest(10,"min_gap_minutes")).items():
        frame.to_csv(REPORT/f"06_05_{name}.csv",index=False)
    REPORT.mkdir(parents=True,exist_ok=True)
    summary.to_csv(REPORT/"06_05_case_summary.csv",index=False)
    intervals.to_csv(REPORT/"06_05_case_intervals.csv",index=False)
    df.to_csv(REPORT/"06_05_case_records.csv",index=False)
    assert digest(INPUT)==before
    journal=dict(checked_date="2026-10-09",input=str(INPUT.relative_to(ROOT)),input_sha256=before,input_unchanged=True,rows_checked=total,go_arounds=positive,more_than_two_attempt_rows=more_two,selected_rows=len(df),selected_go_arounds=int(df.GoA.sum()),attempt_count_distribution={str(k):int(v) for k,v in df.n_attempts.value_counts().sort_index().items()},intervals_checked=len(intervals),min_gap_seconds=float(intervals.gap_seconds.min()),max_last_attempt_to_landing_seconds=float(summary.last_attempt_to_landing_seconds.max()),author_callsign_prefix_filter_matches=int(summary.author_callsign_prefix_filter_match.sum()),author_filter_also_requires_n_attempts_below_4=True,author_filter_applied=False,mission_verified=False,trajectory_validation_performed=False,rows_changed=0,rows_deleted=0,extract=str(EXTRACT.relative_to(ROOT)),extract_sha256=digest(EXTRACT),structural_checks_passed=True)
    journal.update(airports=int(summary.airport.nunique()),addresses=int(summary.icao24.nunique()),same_runway_records=int(summary.same_runway_all_attempts.sum()),changed_runway_records=int((~summary.same_runway_all_attempts).sum()),label_22R_or_04L_rows=int(summary.has_22R_or_04L.sum()),last_attempt_to_landing_over_60_seconds=int(summary.last_attempt_to_landing_over_60_seconds.sum()),gap_median_seconds=float(intervals.gap_seconds.median()),gap_max_seconds=float(intervals.gap_seconds.max()),window_median_minutes=float(summary.attempt_window_minutes.median()),window_min_minutes=float(summary.attempt_window_minutes.min()),window_max_minutes=float(summary.attempt_window_minutes.max()),short_intervals_at_most_39_seconds=int(intervals.gap_seconds.le(39).sum()),threshold_60_seconds_is_diagnostic_only=True,mission_filter_applied=False)
    (REPORT/"06_05_audit.json").write_text(json.dumps(journal,ensure_ascii=False,indent=2)+"\n",encoding="utf8")
    return summary, intervals, journal


if __name__=="__main__":
    summary,intervals,journal=main()
    print(json.dumps(journal,ensure_ascii=False,indent=2))
