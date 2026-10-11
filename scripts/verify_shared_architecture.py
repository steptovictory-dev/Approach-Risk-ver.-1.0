"""Verify extraction against frozen legacy code and registered data; no fit."""
from pathlib import Path
import sys,json
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
sys.path.insert(0,str(ROOT/'research/weather-model-comparison/scripts'))
import comparison_pipeline as old
from approach_risk.data.registry import dataset,sha
from approach_risk.features import c40
from approach_risk.models.logistic import build_pipeline
from approach_risk.evaluation import classification as new

def main():
    ds=dataset();manifest=ds.manifest();counts={};labels={}
    for entry in manifest['files']:
        p=ds.directory/entry['file'];assert sha(p)==entry['sha256']
        f=pd.read_parquet(p,columns=['source_row','GoA','temporal_split'])
        assert len(f)==entry['rows']
        for split,g in f.groupby('temporal_split'):
            counts[split]=counts.get(split,0)+len(g);labels[split]=labels.get(split,0)+int(g.GoA.sum())
    assert counts=={'train':4165060,'validation':1619848,'test':546577}
    assert sum(labels.values())==18349
    frame=next(ds.iter_split('train')).head(1000)
    for aircraft in [True,False]:
        for weather in [True,False]:
            pd.testing.assert_frame_equal(old.prepare_features(frame,weather,aircraft),c40.prepare_features(frame,weather,aircraft))
            assert build_pipeline(weather,aircraft).named_steps['model'].get_params()==old.build_pipeline(weather,aircraft).named_steps['model'].get_params()
    for y,p in [([1,0,1,0],[.9,.8,.8,.1]),([0,0],[.1,.2]),([0,1],[1.,1.])]:
        assert old.metrics(y,p)==new.metrics(y,p)
        for t in [.5,1.01]:assert old.threshold_metrics(y,p,t)==new.threshold_metrics(y,p,t)
        if len(set(y))==2:assert old.choose_thresholds(y,p)==new.choose_thresholds(y,p)
        pd.testing.assert_frame_equal(old.calibration_table(y,p),new.calibration_table(y,p))
    # Recompute saved test metrics through the new package, not new training.
    expected=pd.read_csv(ROOT/'research/weather-model-comparison/reports/04_test_evaluation.csv')
    for name in ['baseline','weather']:
        p=pd.read_parquet(ROOT/f'research/weather-model-comparison/data/primary_models_v1/{name}_test_predictions.parquet')
        actual=new.metrics(p.GoA,p.probability)
        row=expected[expected.dimension.eq('all')&expected.model.eq(name)&expected.AP.notna()].iloc[0]
        for metric in ['AP','ROC_AUC','Brier']:np.testing.assert_allclose(actual[metric],row[metric],rtol=1e-12)
    result={'verified':True,'dataset':ds.name,'files_verified':len(manifest['files']),'rows':sum(counts.values()),'split_rows':counts,'labels':labels,
            'legacy_feature_equivalence':True,'saved_test_metrics_reproduced':True,'fit_executed':False}
    dest=ROOT/'research/boosting-weather-comparison/reports/00_architecture_verification.json'
    dest.write_text(json.dumps(result,indent=2)+'\n');print(result)
if __name__=='__main__':main()
