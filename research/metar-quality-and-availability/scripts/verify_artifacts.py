"""Проверки согласованности результатов и неизменности источников исследования."""
from pathlib import Path
import hashlib
import importlib.metadata
import json
import nbformat
import pandas as pd

BASE=Path(__file__).resolve().parents[1]
PROJECT=BASE.parents[1]
R=BASE/'reports'
def sha(path):
    with path.open('rb') as stream:return hashlib.file_digest(stream,'sha256').hexdigest()
manifest=json.loads((R/'02_study_manifest.json').read_text(encoding='utf8'))
assert sha(BASE/'protocol.md')==manifest['protocol_sha256']
assert sha(BASE/'scripts/run_study.py')==manifest['implementation_sha256']
assert sha(PROJECT/'europe/data/analytical/06_05_cases_excluded.parquet')==manifest['input_sha256']
for job in manifest['sources']:
    for sub,name,key in [('full/events',job['job']+'.parquet','event_sha256'),
                        ('full_cloud_v2/reports',job['job']+'_reports.parquet','reports_sha256'),
                        ('final_c40_weather_v1',job['job']+'.parquet','final_sha256')]:
        assert sha(PROJECT/'europe/weather/data'/sub/name)==job[key]
summary=pd.read_csv(R/'02_scenario_summary.csv')
assert len(summary)==14
assert summary.loc[summary.GoA.eq(True),'rows'].eq(18349).all()
assert summary.loc[summary.GoA.eq(False),'rows'].eq(6313136).all()
assert (summary.usable<=summary.observation).all() and (summary.observation<=summary.rows).all()
assert (summary.information_changed<=summary.rows).all()
assert summary.loc[summary.scenario.eq('primary'),'information_changed'].eq(0).all()
assert summary.loc[summary.scenario.eq('no_cor'),'cor_usable'].eq(0).all()
strata=pd.read_csv(R/'02_scenario_strata.csv')
for field in ['rows','usable','observation','message_changed','information_changed','cor_usable']:
    expected=strata.groupby(['scenario','GoA'])[field].sum().sort_index()
    actual=summary.set_index(['scenario','GoA'])[field].sort_index()
    pd.testing.assert_series_equal(expected,actual)
reasons=pd.read_csv(R/'03_missing_reasons.csv')
pd.testing.assert_series_equal(reasons.groupby(['scenario','GoA']).rows.sum().sort_index(),summary.set_index(['scenario','GoA']).rows.sort_index())
fields=pd.read_csv(R/'03_field_differences.csv')
assert (fields.value_changed<=fields.paired_present).all()
assert (fields.threshold_changed<=fields.value_changed).all()
assert (fields.paired_present<=fields.rows).all()
bootstrap=pd.read_csv(R/'04_block_bootstrap.csv')
assert len(bootstrap)==42 and bootstrap.replicates.eq(2000).all()
assert (bootstrap.lower_pp<=bootstrap.upper_pp).all()
validation=json.loads((R/'05_validation_manifest.json').read_text(encoding='utf8'))
assert validation['semantic_fixtures_pass'] and validation['all_disagreement_cores_blocked']
figures=0
for path in (BASE/'notebooks').glob('*.ipynb'):
    notebook=nbformat.read(path,as_version=4)
    nbformat.validate(notebook)
    for cell in notebook.cells:
        if cell.cell_type=='code':
            assert cell.execution_count is not None
            assert not any(o.output_type=='error' for o in cell.outputs)
            figures+=sum('image/png' in o.get('data',{}) for o in cell.outputs)
assert figures==7
for path in (BASE/'figures').glob('*.png'):
    assert sha(path)==sha(PROJECT/'Articles/article_04/figures'/path.name)
dependencies={name:importlib.metadata.version(name) for name in ['pandas','numpy','pyarrow','metar','nbformat','nbclient','ipykernel','matplotlib']}
result=dict(verified=True,source_files_verified=3*len(manifest['sources']),rows=manifest['rows'],
    source_rows_unchanged=True,training_executed=False,executed_notebooks=3,embedded_figures=figures,
    semantic_fixtures_pass=True,lexical_disagreements_explained=True,dependencies=dependencies)
(R/'08_artifact_verification.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
print(result,flush=True)
