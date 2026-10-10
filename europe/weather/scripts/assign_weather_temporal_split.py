"""Отдельная карта согласованного временного разделения, без обучения."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd

BASE = Path(__file__).resolve().parents[1]
REPORTS = BASE / 'reports'
application = json.loads((REPORTS / '23_final_weather_application.json').read_text(encoding='utf8'))
destination = BASE / 'data/temporal_split_v1'
destination.mkdir(parents=True, exist_ok=True)
seen = np.zeros(application['rows'], dtype=bool)
counts, files = [], []
for item in application['files']:
    source = BASE / 'data/final_c40_weather_v1' / item['name']
    assert hashlib.sha256(source.read_bytes()).hexdigest() == item['sha256']
    frame = pd.read_parquet(source, columns=['source_row', 'C40_CROSS_TIME', 'airport', 'GoA'])
    years = frame.C40_CROSS_TIME.dt.year
    assert years.between(2019, 2023).all()
    frame['temporal_split'] = np.select([years.lt(2022), years.eq(2022)], ['train', 'validation'], default='test')
    assert not seen[frame.source_row].any()
    seen[frame.source_row] = True
    output = destination / item['name']
    frame.to_parquet(output, index=False)
    pd.testing.assert_frame_equal(frame, pd.read_parquet(output))
    counts.append(frame.groupby(['temporal_split', 'airport', 'GoA']).size().rename('rows').reset_index())
    files.append(dict(name=item['name'], sha256=hashlib.sha256(output.read_bytes()).hexdigest()))
assert seen.all()
pd.concat(counts).groupby(['temporal_split', 'airport', 'GoA']).rows.sum().reset_index().to_csv(REPORTS / '24_temporal_split_counts.csv', index=False)
protocol = json.loads((REPORTS / '22_final_weather_protocol.json').read_text(encoding='utf8'))
protocol['temporal_split'] = dict(status='accepted_and_applied_as_separate_mapping', time_field='C40_CROSS_TIME', timezone='UTC', train='before 2022-01-01', validation='2022', test='2023', independent_confirmatory_test=False)
(REPORTS / '22_final_weather_protocol.json').write_text(json.dumps(protocol, ensure_ascii=False, indent=2)+'\n', encoding='utf8')
(REPORTS / '24_temporal_split_application.json').write_text(json.dumps(dict(rows=int(seen.sum()), source_input_sha256=application['input_sha256'], files=files, training_executed=False, model_table_join='drop placeholder temporal_split then merge mapping on source_row; validate one_to_one'), indent=2)+'\n', encoding='utf8')
print(pd.concat(counts).groupby('temporal_split').rows.sum().to_dict())
