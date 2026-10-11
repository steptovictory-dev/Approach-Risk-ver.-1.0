"""Independently verify the saved preparation artifacts without fitting."""
from pathlib import Path
import hashlib
import json
import nbformat
import pandas as pd
import pyarrow.parquet as pq

BASE = Path(__file__).resolve().parents[1]
manifest = json.loads((BASE / 'reports/02_model_table_manifest.json').read_text(encoding='utf8'))
rows = labels = 0
for entry in manifest['files']:
    path = BASE / 'data/model_table_v1' / entry['file']
    with path.open('rb') as stream:
        assert hashlib.file_digest(stream, 'sha256').hexdigest() == entry['sha256']
    assert pq.ParquetFile(path).metadata.num_rows == entry['rows']
    frame = pd.read_parquet(path, columns=['GoA'])
    rows += len(frame)
    labels += int(frame.GoA.sum())
assert rows == manifest['rows'] == 6331485
assert labels == manifest['GoA'] == 18349
registry = pd.read_csv(BASE / 'reports/02_feature_registry.csv')
assert registry.field.is_unique
assert list(registry.field) == manifest['baseline_features'] + manifest['weather_features']
assert not set(registry.field) & {'GoA', 'source_row', 'landing_time', 'attempt_times', 'icao24', 'flight_id', 'gap_status', 'temporal_split'}
notebook = nbformat.read(BASE / 'notebooks/02_model_table.ipynb', as_version=4)
nbformat.validate(notebook)
code = [cell for cell in notebook.cells if cell.cell_type == 'code']
assert all(cell.execution_count is not None for cell in code)
assert not any(output.output_type == 'error' for cell in code for output in cell.outputs)
assert any('image/png' in output.get('data', {}) for cell in code for output in cell.outputs)
assert manifest['training_executed'] is False and manifest['transformations_fitted'] is False
print(f'Verified {len(manifest["files"])} hashes, {rows} rows, {labels} GoA, executed notebook and predictor registry.')
