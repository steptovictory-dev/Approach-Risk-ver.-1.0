"""Independently verify saved candidate provenance and selected predictions."""
import hashlib,json,sys
from pathlib import Path
import nbformat
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'src'))
from approach_risk.data.registry import dataset
from approach_risk.models.boosting import pool
from approach_risk.evaluation.classification import metrics

BASE=ROOT/'research/boosting-weather-comparison'
DEST=ROOT/'artifacts/boosting-weather-comparison'
manifest=json.loads((BASE/'reports/05_artifact_manifest.json').read_text(encoding='utf8'))
for item in manifest['files']:
    with (ROOT/item['path']).open('rb') as stream:
        assert hashlib.file_digest(stream,'sha256').hexdigest()==item['sha256'],item['path']
rows=pd.read_csv(BASE/'reports/01_validation_candidates.csv')
selection=json.loads((BASE/'reports/02_selection.json').read_text(encoding='utf8'))
assert len(rows)==10 and not selection['test_used_for_selection']
for name in ['baseline','weather']:
    best=rows[rows.variant==name].sort_values(['AP','depth','learning_rate'],ascending=[False,True,True]).iloc[0]
    assert best.candidate==selection['selected'][name]
    for row in rows[rows.variant==name].itertuples():
        parameters=json.loads((DEST/name/row.candidate/'actual_parameters.json').read_text(encoding='utf8'))
        assert parameters['depth']==row.depth and parameters['iterations']==500
        assert parameters['bootstrap_type']=='No' and parameters['has_time']
        assert parameters['random_seed']==20261011
        learning=pd.read_csv(BASE/f'reports/01_{name}_{row.candidate}_learning.csv')
        assert len(learning)==500 and learning.train_Logloss.notna().all()
        pred=pd.read_parquet(DEST/name/row.candidate/'validation_predictions.parquet')
        assert len(pred)==1619848 and pred.GoA.sum()==4984 and pred.source_row.is_unique
        actual=metrics(pred.GoA,pred.probability)
        for metric in ['AP','ROC_AUC','Brier']:
            np.testing.assert_allclose(actual[metric],getattr(row,metric),rtol=1e-12)
    pred=pd.read_parquet(DEST/f'{name}_test_predictions.parquet').set_index('source_row')
    model=CatBoostClassifier()
    model.load_model(str(DEST/name/best.candidate/'model.cbm'))
    assert model.tree_count_==500 and len(model.feature_names_)==(34 if name=='weather' else 8)
    frame=next(dataset('europe_c40_model_v1').iter_split('test')).head(1000)
    np.testing.assert_allclose(model.predict_proba(pool(frame,name=='weather'))[:,1],pred.loc[frame.source_row,'probability'],rtol=1e-12)
notebook=nbformat.read(BASE/'notebooks/01_catboost_training.ipynb',as_version=4)
code=[c for c in notebook.cells if c.cell_type=='code']
assert all(c.execution_count is not None for c in code)
assert not any(o.output_type=='error' for c in code for o in c.outputs)
report={'verified':True,'candidate_count':len(rows),'hashed_files':len(manifest['files']),'executed_cells':len(code),'validation_metrics_reproduced':True,'selected_test_predictions_reproduced_rows':1000,'selection_matches_validation_AP':True}
(BASE/'reports/06_independent_verification.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf8')
print(json.dumps(report,indent=2))
