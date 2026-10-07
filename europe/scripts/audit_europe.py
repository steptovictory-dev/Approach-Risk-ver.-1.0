from pathlib import Path
from collections import Counter
import json
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pyarrow.compute as pc
ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'europe/data/source/osn23_landings_merged.parquet.gzip'

def main():
    parquet = pq.ParquetFile(SOURCE)
    missing = Counter()
    airports = Counter()
    periods = Counter()
    labels = Counter()
    attempts = Counter()
    min_time, max_time = None, None
    mismatches = Counter()
    keys = []
    for batch in parquet.iter_batches(batch_size=100_000):
        for name in batch.schema.names:
            missing[name] += batch.column(name).null_count
        frame = batch.select(['airport','landing_time','GoA','n_attempts','flight_id']).to_pandas()
        times = frame['landing_time']
        lo, hi = times.min(), times.max()
        if pd.notna(lo):
            min_time = lo if min_time is None else min(min_time,lo)
            max_time = hi if max_time is None else max(max_time,hi)
        airports.update(frame.groupby(['airport','GoA'],dropna=False).size().to_dict())
        month = times.dt.strftime('%Y-%m')
        periods.update(frame.groupby([month,'GoA'],dropna=False).size().to_dict())
        labels.update(frame['GoA'].value_counts(dropna=False).to_dict())
        attempts.update(frame['n_attempts'].value_counts(dropna=False).to_dict())
        lengths = pc.list_value_length(batch.column('attempt_times')).to_numpy(zero_copy_only=False)
        mismatches['attempt_times_length_vs_n_attempts'] += int(np.sum(np.isfinite(lengths) & (lengths != frame['n_attempts'].to_numpy())))
        mismatches['nonpositive_n_attempts'] += int((frame['n_attempts'] < 1).sum())
        mismatches['GoA_vs_n_attempts_gt1'] += int((frame['GoA'] != (frame['n_attempts'] > 1)).sum())
        keys.append(pd.util.hash_pandas_object(frame['flight_id'],index=False).to_numpy())
    hashes = np.concatenate(keys)
    possible_duplicates = int(len(hashes) - len(np.unique(hashes)))
    airport_rows = [{'airport': airport,'records': int(sum(v for (a,g),v in airports.items() if a==airport)),'with_goaround':int(airports.get((airport,True),0))} for airport in sorted({a for a,g in airports})]
    period_rows = [{'month': month,'records':int(sum(v for (m,g),v in periods.items() if m==month)),'with_goaround':int(periods.get((month,True),0))} for month in sorted({m for m,g in periods})]
    summary = {'source':str(SOURCE.relative_to(ROOT)).replace('\\','/'),'rows':parquet.metadata.num_rows,'schema':str(parquet.schema_arrow),'min_landing_time':str(min_time),'max_landing_time':str(max_time),'airports':airport_rows,'months':period_rows,'labels':{str(k):int(v) for k,v in labels.items()},'n_attempts':{str(k):int(v) for k,v in attempts.items()},'null_counts':dict(missing),'checks':dict(mismatches),'flight_id_hash_duplicate_candidates':possible_duplicates,'duplicate_check_note':'64-bit hashes are screening only; candidates require exact identifier verification. No records filtered.'}
    (ROOT/'europe/reports/initial_snapshot/europe_initial_audit.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf8')
    pd.DataFrame(airport_rows).to_csv(ROOT/'europe/reports/initial_snapshot/europe_airports.csv',index=False)
    pd.DataFrame(period_rows).to_csv(ROOT/'europe/reports/initial_snapshot/europe_months.csv',index=False)
    print(json.dumps({k:v for k,v in summary.items() if k not in ['schema','months','null_counts']},ensure_ascii=False,indent=2),flush=True)

if __name__ == '__main__':
    main()
