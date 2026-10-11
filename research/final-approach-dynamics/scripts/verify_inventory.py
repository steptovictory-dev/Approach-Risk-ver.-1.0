"""Verify an executed local inventory against its unchanged input hashes."""
from pathlib import Path
import hashlib,json
import nbformat
import pandas as pd
ROOT=Path(__file__).resolve().parents[3];BASE=ROOT/'research/final-approach-dynamics'
manifest=json.loads((BASE/'reports/00_inventory_manifest.json').read_text(encoding='utf8'))
assert not manifest['training_started'] and len(manifest['sources'])==11
for entry in manifest['sources']:
    with (ROOT/entry['source']).open('rb') as stream:
        assert hashlib.file_digest(stream,'sha256').hexdigest()==entry['sha256']
inventory=pd.read_csv(BASE/'reports/00_local_sources.csv')
assert len(inventory)==11 and inventory.source.is_unique and not inventory.has_GoA_label.any()
n=nbformat.read(BASE/'notebooks/00_data_inventory.ipynb',as_version=4)
code=[c for c in n.cells if c.cell_type=='code']
assert all(c.execution_count is not None for c in code)
assert not any(o.output_type=='error' for c in code for o in c.outputs)
for name in ['00_source_inventory','01_local_sequence','02_position_freshness']:
    for ext in ['png','svg']:assert (BASE/f'figures/{name}.{ext}').stat().st_size>0
result={'verified':True,'unchanged_input_hashes':11,'executed_cells':len(code),'figures':3,'fit_performed':False}
(BASE/'reports/00_verification.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
print(json.dumps(result,indent=2))
