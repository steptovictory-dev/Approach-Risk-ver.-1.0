from pathlib import Path
import json
import pyarrow.parquet as pq
import pyarrow.compute as pc
root=Path(__file__).resolve().parents[2]
p=root/'europe/data/source/osn23_landings_merged.parquet.gzip'
ids=pq.read_table(p,columns=['flight_id']).column('flight_id')
counts=pc.value_counts(ids)
repeated=pc.filter(counts,pc.greater(counts.field('counts'),1))
keys=repeated.field('values')
print('Exact repeated flight_id values',len(keys),'excess rows',sum(repeated.field('counts').to_pylist())-len(keys),flush=True)
rows=[]
for batch in pq.ParquetFile(p).iter_batches(batch_size=100000):
 selected=batch.filter(pc.is_in(batch.column('flight_id'),value_set=keys))
 rows.extend(selected.to_pylist())
canonical=[json.dumps({k:v for k,v in r.items() if k!='__index_level_0__'},sort_keys=True,default=str) for r in rows]
report={'repeated_flight_ids':len(keys),'excess_identifier_rows':sum(repeated.field('counts').to_pylist())-len(keys),'duplicate_rows_excluding_storage_index':len(canonical)-len(set(canonical)),'storage_index_excluded':True,'rows_screened':len(rows),'examples':rows[:8],'no_rows_removed':True}
(root/'europe/reports/initial_snapshot/europe_duplicate_check.json').write_text(json.dumps(report,ensure_ascii=False,indent=2,default=str),encoding='utf8')
print(json.dumps({k:v for k,v in report.items() if k!='examples'},indent=2))
print(json.dumps(rows[:2],ensure_ascii=False,default=str,indent=2))
