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
if (reports/'metadata_verification.json').exists():
    metadata=json.loads((reports/'metadata_verification.json').read_text(encoding='utf-8'))
    assert metadata['balances_passed'] and metadata['scenario_matches_notebook04']
    monthly=pd.read_csv(reports/'metadata_monthly.csv')
    for entry in metadata['inputs']:
        assert entry['input_unchanged']
        base=monthly[monthly.stage==entry['stage']].set_index('month')[['rows','goa']].sort_index()
        assert base.rows.sum()==entry['rows'] and base.goa.sum()==entry['goa']
        for name in ['patterns','exclusive_reasons']:
            table=pd.read_csv(reports/f'metadata_{name}.csv')
            actual=table[table.stage==entry['stage']].groupby('month')[['rows','goa']].sum().sort_index()
            pd.testing.assert_frame_equal(actual,base)
if (reports/'protocol_manifest.json').exists():
    protocol=json.loads((reports/'protocol_manifest.json').read_text(encoding='utf-8'))
    assert protocol['balances_passed'] and protocol['input_unchanged']
    split=pd.read_csv(reports/'protocol_split_summary.csv')
    assert split.rows.sum()==summary.loc['final','rows']
    assert split.goa.sum()==summary.loc['final','goa']
    monthly=pd.read_csv(reports/'protocol_month_coverage.csv').groupby('split')[['rows','goa']].sum()
    pd.testing.assert_frame_equal(monthly.sort_index(),split.set_index('split')[['rows','goa']].sort_index())
    airport=pd.read_csv(reports/'protocol_airport_coverage.csv').groupby('split')[['rows','goa']].sum()
    expected=split[split.split.isin(['train','validation','test'])].set_index('split')[['rows','goa']]
    pd.testing.assert_frame_equal(airport.sort_index(),expected.sort_index())
    fields=pd.read_csv(reports/'protocol_feature_sets.csv')
    assert not fields.field.isin(['GoA','landing_time','attempt_times','market_segment','icao24','flight_id']).any()
    assert not protocol['training_performed'] and not protocol['operational_availability_verified']
if (reports/'preparation_plan.json').exists():
    preparation=json.loads((reports/'preparation_plan.json').read_text(encoding='utf-8'))
    assert preparation['category_balances_passed'] and preparation['input_unchanged']
    assert not preparation['classifier_training_performed'] and not preparation['test_metrics_computed']
    vocabulary=json.loads((reports/'preparation_train_vocabulary.json').read_text(encoding='utf-8'))
    scaler=json.loads((reports/'preparation_numeric_scaler.json').read_text(encoding='utf-8'))
    assert vocabulary['fit_split']=='train' and scaler['fit_split']=='train'
    counts=pd.read_csv(reports/'preparation_category_counts.csv',keep_default_na=False)
    states=pd.read_csv(reports/'preparation_category_states.csv')
    for field,values in vocabulary['vocabulary'].items():
        train=counts[counts.split.eq('train')&counts.field.eq(field)]
        assert values==sorted(train.raw_value.unique().tolist())
    assert states[states.split.eq('train')&states.state.eq('unseen')].rows.sum()==0
    split=pd.read_csv(reports/'protocol_split_summary.csv').set_index('split')
    for part in ['train','validation','test']:
        assert states[states.split.eq(part)].groupby('field').rows.sum().eq(split.loc[part,'rows']).all()
    assert all(x['fit_rows']==split.loc['train','rows'] and x['scale']>0 for x in scaler['parameters'].values())
for path in (root/'figures').glob('*.png'):
    with Image.open(path) as im:
        im.verify()
    assert path.with_suffix('.svg').exists()
print('All aggregate balances and image files passed')
packages = ['numpy','pandas','pyarrow','matplotlib','nbformat','nbclient','ipykernel','Pillow']
(root/'requirements-lock.txt').write_text(
    '\n'.join(f'{name}=={importlib.metadata.version(name)}' for name in packages)+'\n',
    encoding='utf-8')
