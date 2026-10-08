"""Исключение CFL12 по согласованному решению о составе эксперимента."""
from datetime import datetime, timezone
import json
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
from audit_feature_availability import ROOT, digest

INPUT=ROOT/"europe/data/analytical/06_02_lsgg_case_excluded.parquet"
OUTPUT=ROOT/"europe/data/analytical/06_04_cfl12_excluded.parquet"
EXCLUDED=ROOT/"europe/data/review/06_04_cfl12_excluded_record.parquet"
REPORT=ROOT/"europe/reports/attempts"
KEY=dict(flight_id="CFL12_27984",icao24="4ab179",airport="ESSA",landing_time="2022-11-12T10:07:17+00:00")


def mask(table):
    result=pc.equal(table["flight_id"],KEY["flight_id"])
    for name in ("icao24","airport"):
        result=pc.and_(result,pc.equal(table[name],KEY[name]))
    instant=pa.scalar(pd.Timestamp(KEY["landing_time"]).to_pydatetime(),type=table.schema.field("landing_time").type)
    return pc.fill_null(pc.and_(result,pc.equal(table["landing_time"],instant)),False)


def equal(left,right):
    assert left.schema.equals(right.schema,check_metadata=True)
    if not left.equals(right):pd.testing.assert_frame_equal(left.to_pandas(ignore_metadata=True),right.to_pandas(ignore_metadata=True),check_exact=True)


def main():
    assert not OUTPUT.exists() and not EXCLUDED.exists(),"Выход уже существует; перезапись запрещена"
    prior=json.loads((REPORT/"LSGG/06_02_exclusion_applied.json").read_text(encoding="utf8"))
    before=digest(INPUT);assert before==prior["output_sha256"]
    source=pq.ParquetFile(INPUT);found=[]
    for batch in source.iter_batches(batch_size=100_000,columns=list(KEY)+["GoA","n_attempts"]):
        table=pa.Table.from_batches([batch]);found.extend(table.filter(mask(table)).to_pylist())
    assert len(found)==1 and found[0]["GoA"] is True and found[0]["n_attempts"]==9
    temp=OUTPUT.with_suffix(".pending.parquet");assert not temp.exists()
    rows=positive=more_two=more_three=0;excluded=None
    with pq.ParquetWriter(temp,source.schema_arrow,compression="zstd") as writer:
        for batch in source.iter_batches(batch_size=100_000):
            table=pa.Table.from_batches([batch]);selected=mask(table)
            if pc.any(selected).as_py():excluded=table.filter(selected)
            kept=table.filter(pc.invert(selected));writer.write_table(kept,row_group_size=100_000)
            rows+=len(kept);positive+=pc.sum(pc.cast(kept["GoA"],pa.int64())).as_py()
            more_two+=pc.sum(pc.cast(pc.greater(kept["n_attempts"],2),pa.int64())).as_py()
            more_three+=pc.sum(pc.cast(pc.greater(kept["n_attempts"],3),pa.int64())).as_py()
    assert rows==prior["rows_after"]-1 and positive==prior["go_arounds_after"]-1
    assert more_two==147 and more_three==6
    saved=pq.ParquetFile(temp);groups=0
    for index,batch in enumerate(source.iter_batches(batch_size=100_000)):
        table=pa.Table.from_batches([batch]);equal(table.filter(pc.invert(mask(table))),saved.read_row_group(index));groups+=1
    assert saved.num_row_groups==groups and saved.metadata.num_rows==rows
    saved.close();assert excluded is not None and len(excluded)==1
    pq.write_table(excluded,EXCLUDED,compression="zstd");equal(excluded,pq.read_table(EXCLUDED))
    assert digest(INPUT)==before;temp.replace(OUTPUT)
    result=dict(applied_utc=datetime.now(timezone.utc).isoformat(),input=str(INPUT.relative_to(ROOT)),input_sha256=before,input_unchanged=True,output=str(OUTPUT.relative_to(ROOT)),output_sha256=digest(OUTPUT),excluded=str(EXCLUDED.relative_to(ROOT)),excluded_sha256=digest(EXCLUDED),exclusion_key=KEY,reason="По прямому решению пользователя CFL12 исключен как нетипичный для выбранной исследовательской популяции; конкретная историческая миссия независимо не подтверждена",rows_before=prior["rows_after"],rows_after=rows,go_arounds_before=prior["go_arounds_after"],go_arounds_after=positive,more_than_two_attempt_rows=more_two,more_than_three_attempt_rows=more_three,rows_excluded=1,retained_values_changed=0,all_retained_fields_and_order_verified=True,physical_columns=len(source.schema_arrow),label_corrected=False)
    (REPORT/"06_04_cfl12_exclusion_applied.json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf8")
    print(json.dumps(result,ensure_ascii=False,indent=2),flush=True)


if __name__=="__main__":main()
