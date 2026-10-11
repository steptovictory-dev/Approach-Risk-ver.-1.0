"""Compare optimized block intervals against explicit resampling and sklearn."""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score,roc_auc_score,brier_score_loss
sys.path.insert(0,str(Path(__file__).resolve().parents[3]/'src'))
from approach_risk.evaluation.bootstrap import paired_intervals

rng=np.random.default_rng(42)
for tied in [False,True]:
    days=np.repeat(np.arange(20),15)
    y=(rng.random(len(days))<.12).astype(int)
    a=rng.random(len(days));b=rng.random(len(days))
    if tied:a=np.round(a,1);b=np.round(b,1)
    result,metadata=paired_intervals(y,a,b,pd.Timestamp('2023-01-01',tz='UTC')+pd.to_timedelta(days,unit='D'),replicates=200,seed=7)
    reference=[];sampling=np.random.default_rng(7)
    def score(labels,p):return np.array([average_precision_score(labels,p),roc_auc_score(labels,p),brier_score_loss(labels,p)])
    for _ in range(200):
        draw=sampling.integers(0,20,20)
        ids=np.concatenate([np.flatnonzero(days==d) for d in draw])
        if len(np.unique(y[ids]))==2:reference.append(score(y[ids],b[ids])-score(y[ids],a[ids]))
    lo,hi=np.quantile(reference,[.025,.975],axis=0)
    np.testing.assert_allclose(result.lower_95,lo,rtol=1e-10,atol=1e-12)
    np.testing.assert_allclose(result.upper_95,hi,rtol=1e-10,atol=1e-12)
    np.testing.assert_allclose(result.difference,score(y,b)-score(y,a),rtol=1e-10,atol=1e-12)
    assert metadata['valid']==len(reference)
    series_result,_=paired_intervals(y,a,b,pd.Series(pd.Timestamp('2023-01-01',tz='UTC')+pd.to_timedelta(days,unit='D')),replicates=200,seed=7)
    pd.testing.assert_frame_equal(series_result,result)
print('Block intervals match explicit sklearn resampling, with and without ties.')
