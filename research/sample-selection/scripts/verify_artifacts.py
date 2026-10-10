"""Проверка завершенности ноутбуков и баланса опубликованных агрегатов."""
from pathlib import Path
import json
import importlib.metadata
import nbformat
import pandas as pd
from PIL import Image

root = Path(__file__).resolve().parents[1]
for path in sorted((root/'notebooks').glob('*.ipynb')):
    notebook = nbformat.read(path, as_version=4)
    nbformat.validate(notebook)
    for cell in notebook.cells:
        if cell.cell_type == 'code':
            assert cell.execution_count is not None, path
            assert not any(o.output_type == 'error' for o in cell.outputs), path
    print(path.name, 'OK')
reports = root/'reports'
summary = pd.read_csv(reports/'selection_summary.csv').set_index('stage')
for path in reports.glob('by_*_all_stages.csv'):
    grouped = pd.read_csv(path).groupby('stage')[['rows','goa']].sum()
    pd.testing.assert_frame_equal(grouped.sort_index(), summary[['rows','goa']].sort_index(),
                                  check_dtype=False)
for path in reports.glob('comparison_*.csv'):
    frame = pd.read_csv(path)
    assert frame.rows_source.sum() == summary.loc['source','rows'], path
    assert frame.rows_final.sum() == summary.loc['final','rows'], path
    assert frame.goa_source.sum() == summary.loc['source','goa'], path
    assert frame.goa_final.sum() == summary.loc['final','goa'], path
    assert (frame.rows_outside >= 0).all() and (frame.goa_outside >= 0).all(), path
reason = pd.read_csv(reports/'exclusive_reasons.csv')
assert reason.rows.sum() == summary.loc['source','rows']
assert reason.goa.sum() == summary.loc['source','goa']
verification = json.loads((reports/'verification.json').read_text(encoding='utf-8'))
assert all(row['passed'] for row in verification['ordered_checks'])
for path in (root/'figures').glob('*.png'):
    with Image.open(path) as im:
        im.verify()
    assert path.with_suffix('.svg').exists()
print('All aggregate balances and image files passed')
packages = ['numpy','pandas','pyarrow','matplotlib','nbformat','nbclient','ipykernel','Pillow']
(root/'requirements-lock.txt').write_text(
    '\n'.join(f'{name}=={importlib.metadata.version(name)}' for name in packages)+'\n',
    encoding='utf-8')
