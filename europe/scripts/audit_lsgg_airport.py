"""Полный табличный аудит LSGG после согласованного исключения одного случая."""
import json
import hashlib
from pathlib import Path

import nbformat
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from audit_feature_availability import ROOT, digest

REPORT = ROOT / "europe/reports/attempts/LSGG"
INPUT = ROOT / "europe/data/analytical/06_02_lsgg_case_excluded.parquet"
ALIASES = {"22R": "22", "04L": "04"}


def main():
    prior = json.loads((REPORT / "06_02_exclusion_applied.json").read_text(encoding="utf8"))
    before = digest(INPUT)
    assert before == prior["output_sha256"]
    columns = ["flight_id", "icao24", "callsign", "typecode", "airport", "landing_time", "attempt_times", "ILS", "n_attempts", "GoA", "AP_C_RWY"]
    pieces = []
    for batch in pq.ParquetFile(INPUT).iter_batches(batch_size=100_000, columns=columns):
        table = pa.Table.from_batches([batch])
        part = table.filter(pc.equal(table["airport"], "LSGG"))
        if len(part):
            pieces.append(part.to_pandas(ignore_metadata=True))
    df = pd.concat(pieces, ignore_index=True)
    assert len(df) == prior["lsgg_rows_after"] and int(df.GoA.sum()) == prior["lsgg_go_arounds_after"]
    assert not (df.flight_id.eq("TAP948_37820") & df.icao24.eq("495295") & df.landing_time.eq(pd.Timestamp("2019-03-01T16:29:15Z"))).any()
    df["month"] = df.landing_time.dt.strftime("%Y-%m")
    df["day"] = df.landing_time.dt.strftime("%Y-%m-%d")
    df["has_nonstandard_label"] = df.ILS.apply(lambda values: any(label in ALIASES for label in values))
    df["runway_sequence"] = df.ILS.apply(lambda values: "|".join(values))
    df["hypothetical_sequence"] = df.ILS.apply(lambda values: "|".join(ALIASES.get(label, label) for label in values))
    df["last_raw_label"] = df.ILS.apply(lambda values: values[-1])
    df["hypothetical_last_label"] = df.last_raw_label.map(lambda label: ALIASES.get(label, label))
    flags = dict(attempt_count_nonpositive=int(df.n_attempts.le(0).sum()), attempt_length_mismatch=int(sum(len(values)!=count for values,count in zip(df.attempt_times,df.n_attempts))), ILS_length_mismatch=int(sum(len(values)!=count for values,count in zip(df.ILS,df.n_attempts))), GoA_differs_from_more_than_one_attempt=int(df.GoA.ne(df.n_attempts.gt(1)).sum()))
    labels = df.copy()
    labels["runway_label"] = labels.ILS.apply(lambda values: list(dict.fromkeys(values)))
    labels = labels.explode("runway_label")
    summary = labels.groupby("runway_label").agg(observations=("GoA","size"), go_arounds=("GoA","sum"), first_seen=("landing_time","min"), last_seen=("landing_time","max")).reset_index()
    monthly = labels.groupby(["month","runway_label"]).agg(observations=("GoA","size"), go_arounds=("GoA","sum")).reset_index()
    daily = labels.groupby(["day","runway_label"]).size().unstack(fill_value=0)
    by_address = labels.groupby("icao24").runway_label.agg(set)
    by_type = labels.groupby("typecode",dropna=False).runway_label.agg(set)
    patterns = df.groupby(["runway_sequence","hypothetical_sequence","GoA"]).size().rename("observations").reset_index()
    cross = df.groupby(["last_raw_label","hypothetical_last_label","AP_C_RWY"],dropna=False).agg(observations=("GoA","size"),go_arounds=("GoA","sum")).reset_index()
    intervals=[]
    unique_reductions = adjacent_aliases = 0
    repeated = descending = last_after = 0
    for row in df.itertuples(index=False):
        raw=list(row.ILS)
        canonical=[ALIASES.get(label,label) for label in raw]
        unique_reductions += int(len(set(canonical)) < len(set(raw)))
        times=np.asarray(row.attempt_times,dtype="datetime64[ns]")
        gaps=np.diff(times)/np.timedelta64(1,"s")
        repeated+=int((gaps==0).any());descending+=int((gaps<0).any())
        last_after+=int(times[-1]>np.datetime64(row.landing_time.to_datetime64(),"ns"))
        for index,gap in enumerate(gaps):
            alias=raw[index]!=raw[index+1] and canonical[index]==canonical[index+1]
            adjacent_aliases+=int(alias)
            intervals.append(dict(flight_id=row.flight_id,icao24=row.icao24,landing_time=row.landing_time,from_label=raw[index],to_label=raw[index+1],gap_seconds=float(gap),hypothetical_same_direction=alias))
    flags.update(repeated_adjacent_attempt_time=repeated,descending_attempt_times=descending,last_attempt_after_landing=last_after)
    intervals=pd.DataFrame(intervals)
    alias_intervals=intervals[intervals.hypothetical_same_direction]
    interval_summary=intervals.groupby(["from_label","to_label"]).agg(intervals=("gap_seconds","size"),min_seconds=("gap_seconds","min"),median_seconds=("gap_seconds","median"),max_seconds=("gap_seconds","max")).reset_index()
    sensitivity=[]
    for name, subset in [("all_LSGG",df),("contains_22R_or_04L",df[df.has_nonstandard_label]),("only_04_or_22",df[~df.has_nonstandard_label])]:
        sensitivity.append(dict(group=name,rows=len(subset),GoA=int(subset.GoA.sum()),source_label_rate_per_1000=1000*subset.GoA.mean()))
    sensitivity=pd.DataFrame(sensitivity)
    review=df.loc[df.has_nonstandard_label | df.GoA,columns+["has_nonstandard_label","runway_sequence","hypothetical_sequence"]].copy()
    for name, frame in dict(labels=summary,monthly=monthly,patterns=patterns,last_label_cross=cross,intervals=intervals,interval_summary=interval_summary,sensitivity=sensitivity,review_records=review,four_L_records=df[df.ILS.apply(lambda values:"04L" in values)]).items():
        frame.to_csv(REPORT/f"06_02_airport_{name}.csv",index=False)
    flags["short_interval_at_most_39_seconds"]=int(intervals.gap_seconds.le(39).sum())
    assert not any(flags.values())
    assert digest(INPUT)==before
    monthly_pivot=monthly.pivot(index="month",columns="runway_label",values="observations").fillna(0)
    result=dict(checked_date="2026-10-09",input=str(INPUT.relative_to(ROOT)),input_sha256=before,input_unchanged=True,rows=len(df),go_arounds=int(df.GoA.sum()),months=len(monthly_pivot),months_with_both_22_labels=int(((monthly_pivot.get("22",0)>0)&(monthly_pivot.get("22R",0)>0)).sum()),days_with_both_22_labels=int(((daily.get("22",0)>0)&(daily.get("22R",0)>0)).sum()),addresses_total=len(by_address),addresses_with_both_22_labels=int(sum({"22","22R"}<=values for values in by_address)),types_with_both_22_labels=int(sum({"22","22R"}<=values for values in by_type)),nonstandard_label_rows=int(df.has_nonstandard_label.sum()),nonstandard_label_positive_rows=int(df.loc[df.has_nonstandard_label,"GoA"].sum()),positive_share_with_nonstandard_label=float(df.loc[df.has_nonstandard_label,"GoA"].sum()/df.GoA.sum()),intervals=len(intervals),min_interval_seconds=float(intervals.gap_seconds.min()),alias_intervals=len(alias_intervals),alias_min_interval_seconds=float(alias_intervals.gap_seconds.min()),observations_with_reduced_unique_labels_under_hypothesis=unique_reductions,attempt_count_changed_by_label_only_replacement=0,GoA_changed_by_label_only_replacement=0,last_label_hypothesis_matches_AP_C_RWY=int(df.hypothetical_last_label.eq(df.AP_C_RWY).sum()),structural_flags=flags,source_rows_changed=0,source_rows_deleted=0,source_labels_validated=False,raw_label_normalization_applied=False,airport_exclusion_applied=False,policy='LSGG сохранен по прямому решению пользователя; ILS и GoA не исправляются; требуется отдельная проверка чувствительности при моделировании')
    mismatch = cross[cross.hypothetical_last_label.ne(cross.AP_C_RWY)]
    share = monthly_pivot["22R"] / (monthly_pivot["22"] + monthly_pivot["22R"])
    result.update(last_label_hypothesis_mismatch_rows=int(mismatch.observations.sum()), last_label_hypothesis_mismatch_positive_rows=int(mismatch.go_arounds.sum()), monthly_22R_share_among_22_labels_min=float(share.min()), monthly_22R_share_among_22_labels_max=float(share.max()), decision_status="Решение пользователя принято", airport_retained=True, model_sensitivity_completed=False, composition_sensitivity_completed=True)
    (REPORT/"06_02_airport_audit.json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf8")
    print(json.dumps(result,ensure_ascii=False,indent=2))
    print(summary.to_string(index=False));print(sensitivity.to_string(index=False));print(interval_summary.to_string(index=False))


if __name__ == "__main__":
    main()
