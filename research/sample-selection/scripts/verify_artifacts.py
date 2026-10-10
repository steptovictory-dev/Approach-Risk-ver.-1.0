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
if (reports/'structure_verification.json').exists():
    structure=json.loads((reports/'structure_verification.json').read_text(encoding='utf-8'))
    assert structure['all_stage_totals_match'] and structure['actual_final_joint_groups_match']
    joint=pd.read_csv(reports/'joint_all_stages.csv').groupby('stage')[['rows','goa']].sum()
    pd.testing.assert_frame_equal(joint.sort_index(),summary[['rows','goa']].sort_index(),check_dtype=False)
    decomposition=pd.read_csv(reports/'rate_decomposition.csv')
    assert ((decomposition.composition_pp+decomposition.within_pp-decomposition.delta_pp).abs()<1e-10).all()
    assert ((decomposition.absent_group_support_pp+decomposition.common_support_delta_pp-
             decomposition.delta_pp).abs()<1e-10).all()
    for path in reports.glob('structure_*.csv'):
        frame=pd.read_csv(path)
        if {'rows_source','goa_source','rows_final','goa_final'} <= set(frame.columns):
            assert frame.rows_source.sum()==summary.loc['source','rows'],path
            assert frame.goa_source.sum()==summary.loc['source','goa'],path
            assert frame.rows_final.sum()==summary.loc['final','rows'],path
            assert frame.goa_final.sum()==summary.loc['final','goa'],path
if (reports/'temporal_verification.json').exists():
    temporal=json.loads((reports/'temporal_verification.json').read_text(encoding='utf-8'))
    assert temporal['input_unchanged'] and temporal['group_balances_passed']
    assert temporal['all_leads_positive'] and temporal['fields_classified']==35
    assert temporal['rows']==summary.loc['final','rows']
    assert temporal['goa']==summary.loc['final','goa']
    contract=pd.read_csv(reports/'temporal_field_contract.csv')
    assert contract.field.is_unique and len(contract)==35
    assert not contract.operationally_verified.any()
    coverage=pd.read_csv(reports/'temporal_scenario_coverage.csv').set_index('scenario')
    grouped=pd.read_csv(reports/'temporal_group_coverage.csv')
    for dimension in ['airport','month']:
        actual=grouped[grouped.dimension==dimension].groupby('scenario')[['rows','goa']].sum()
        pd.testing.assert_frame_equal(actual.sort_index(),coverage[['rows','goa']].sort_index())
    horizons=pd.read_csv(reports/'temporal_horizon_coverage.csv')
    assert (horizons[['rows','goa']].diff().iloc[1:]<=0).all().all()
    assert horizons.iloc[0]['rows']==temporal['rows']
    assert horizons.iloc[0]['goa']==temporal['goa']
for path in (root/'figures').glob('*.png'):
    with Image.open(path) as im:
        im.verify()
    assert path.with_suffix('.svg').exists()
print('All aggregate balances and image files passed')
packages = ['numpy','pandas','pyarrow','matplotlib','nbformat','nbclient','ipykernel','Pillow']
(root/'requirements-lock.txt').write_text(
    '\n'.join(f'{name}=={importlib.metadata.version(name)}' for name in packages)+'\n',
    encoding='utf-8')
