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
for path in [reports/f'comparison_{group}.csv' for group in ['airport','year','month','typecode','market_segment']]:
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
if (reports/'baseline_verification.json').exists():
    baseline=json.loads((reports/'baseline_verification.json').read_text(encoding='utf-8'))
    assert baseline['input_unchanged'] and baseline['converged_all']
    assert baseline['fit_split']=='train' and baseline['selection_split']=='validation'
    assert not baseline['test_used_for_selection'] and not baseline['operational_availability_verified']
    assert baseline['airport_balances_passed'] and baseline['weighted_metrics_match_sklearn']
    search=pd.read_csv(reports/'baseline_validation_search.csv')
    selection=json.loads((reports/'baseline_frozen_selection.json').read_text(encoding='utf-8'))
    assert search.converged.all() and len(search)==8
    for name,choice in selection['chosen'].items():
        best=search[search.model.eq(name)].sort_values(['validation_AP','C'],ascending=[False,True]).iloc[0]
        assert choice['C']==best.C and abs(choice['validation_AP']-best.validation_AP)<1e-12
    metrics=pd.read_csv(reports/'baseline_metrics.csv')
    airport=pd.read_csv(reports/'baseline_airport_metrics.csv')
    for row in metrics.itertuples():
        assert row.rows==row.tp+row.fp+row.fn+row.tn
        assert row.goa==row.tp+row.fn
        assert 0<=row.AP<=1 and 0<=row.ROC_AUC<=1 and 0<=row.Brier<=1
        if row.split=='test':
            group=airport[airport.model.eq(row.model)]
            assert group.rows.sum()==row.rows and group.goa.sum()==row.goa
            assert row.rows==baseline['rows_test'] and row.goa==baseline['goa_test']
    boot=pd.read_csv(reports/'baseline_bootstrap.csv')
    assert len(boot)==3000 and boot.replicate.nunique()==1000
    assert boot.groupby('replicate').model.nunique().eq(3).all()
    intervals=pd.read_csv(reports/'baseline_intervals.csv')
    assert (intervals.lower<=intervals.upper).all() and intervals.valid_replicates.le(1000).all()
if (reports/'errors_verification.json').exists():
    errors=json.loads((reports/'errors_verification.json').read_text(encoding='utf-8'))
    assert errors['group_balances_passed'] and errors['Brier_decomposition_passed']
    assert errors['input_unchanged'] and errors['prediction_unchanged'] and errors['thresholds_unchanged']
    assert not errors['retraining_performed'] and not errors['threshold_search_performed']
    baseline=pd.read_csv(reports/'baseline_metrics.csv')
    for group in errors['diagnostic_groups']:
        table=pd.read_csv(reports/f'errors_by_{group}.csv')
        for model in ['geometry','geometry_aircraft']:
            a=table[table.model.eq(model)]
            b=baseline[baseline.split.eq('test')&baseline.model.eq(model)].iloc[0]
            for column in ['rows','goa','tp','fp','fn','tn']:assert int(a[column].sum())==int(b[column])
            assert (a.loc[a.goa.eq(0),'recall'].isna()).all()
            assert (a.loc[a.goa.eq(0),'AP'].isna()).all()
    overlap=pd.read_csv(reports/'errors_alert_overlap.csv')
    assert overlap.rows.sum()==errors['rows_test']
    assert overlap.loc[overlap.GoA,'rows'].sum()==errors['goa_test']
if (reports/'selection_evaluation_verification.json').exists():
    audit=json.loads((reports/'selection_evaluation_verification.json').read_text(encoding='utf-8'))
    assert audit['exact_join_passed'] and audit['group_balances_passed'] and audit['coverage_identity_passed']
    assert audit['baseline_metrics_match'] and audit['inputs_unchanged'] and audit['predictions_unchanged']
    assert not audit['retraining_performed'] and not audit['source_model_AP_identified']
    assert audit['attempt_order_audit']['first_not_minimum']==0
    population=pd.read_csv(reports/'selection_evaluation_population.csv').set_index('population')
    table=pd.read_csv(reports/'selection_evaluation_metrics.csv')
    for row in table.itertuples():
        assert row.rows==row.tp+row.fp+row.fn_in_coverage+row.tn
        assert row.goa==row.tp+row.fn_in_coverage
        assert row.goa+row.goa_outside_test_coverage==audit['scored_goa']
        assert row.goa+row.goa_outside_source_coverage==audit['source_goa']
        assert abs(row.found_fraction_source-row.tp/audit['source_goa'])<1e-12
        if row.goa:
            assert abs(row.positive_coverage_source*row.recall_in_coverage-row.found_fraction_source)<1e-12
        else:
            assert pd.isna(row.AP) and pd.isna(row.recall_in_coverage)
    assert population.loc['source_landing_period','goa']==audit['source_goa']
if (reports/'comparison_verification.json').exists():
    audit=json.loads((reports/'comparison_verification.json').read_text(encoding='utf-8'))
    assert audit['inputs_unchanged'] and audit['models_unchanged'] and not audit['retraining_performed']
    assert audit['coefficient_replay_passed'] and audit['weighted_metric_parity_passed']
    assert audit['airport_prior_fit_split']=='train' and audit['budget_selection_split']=='validation'
    assert not audit['budget_selection_uses_labels']
    boot=pd.read_csv(reports/'comparison_block_bootstrap.csv')
    assert len(boot)==9000 and boot.groupby(['block_weeks','replicate']).model.nunique().eq(3).all()
    assert set(boot.block_weeks)=={1,2,4}
    ranking=pd.read_csv(reports/'comparison_equal_budget.csv')
    assert ranking.groupby('budget_test').positions.nunique().eq(1).all()
    assert (ranking.expected_tp+ranking.expected_fp-ranking.positions).abs().lt(1e-8).all()
    budgets=pd.read_csv(reports/'comparison_budget_metrics.csv')
    assert budgets.validation_alert_rate.le(budgets.budget_validation+1e-12).all()
    expected_goa=int(pd.read_csv(reports/'baseline_metrics.csv').query('split=="test"').goa.iloc[0])
    assert (budgets.tp+budgets.fn).eq(expected_goa).all()
if (reports/'study_summary.json').exists():
    study=json.loads((reports/'study_summary.json').read_text(encoding='utf-8'))
    audit=json.loads((reports/'selection_evaluation_verification.json').read_text(encoding='utf-8'))
    assert study['source_goa_period']==audit['source_goa']
    assert study['scored_goa_period']==audit['scored_goa']
    assert study['uncovered_goa_period']+study['scored_goa_period']==study['source_goa_period']
    assert abs(study['positive_coverage_period']-audit['scored_goa']/audit['source_goa'])<1e-12
    assert not study['source_model_AP_identified'] and not study['new_training_performed']
    assert not study['operational_availability_verified']
for path in (root/'figures').glob('*.png'):
    with Image.open(path) as im:
        im.verify()
    assert path.with_suffix('.svg').exists()
print('All aggregate balances and image files passed')
packages = ['numpy','pandas','pyarrow','matplotlib','nbformat','nbclient','ipykernel','Pillow',
            'scikit-learn','scipy','threadpoolctl','joblib','narwhals','cloudpickle']
(root/'requirements-lock.txt').write_text(
    '\n'.join(f'{name}=={importlib.metadata.version(name)}' for name in packages)+'\n',
    encoding='utf-8')
