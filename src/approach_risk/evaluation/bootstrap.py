"""Exact paired day bootstrap using score ordering once, including ties."""
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score,roc_auc_score,brier_score_loss
from threadpoolctl import threadpool_limits
from .classification import validate_scores

def _prepared(y,p,day):
    order=np.argsort(-p,kind='stable');scores=p[order]
    starts=np.r_[0,np.flatnonzero(np.diff(scores))+1]
    return y[order],day[order],starts,(p-y)**2

def _weighted(pre,counts,day):
    y,d,starts,error=pre;w=counts[d]
    pos=np.add.reduceat(w*y,starts);neg=np.add.reduceat(w*(1-y),starts)
    tp=np.cumsum(pos);fp=np.cumsum(neg);P=tp[-1];N=fp[-1]
    if not P or not N:return None
    ap=np.sum(pos/P*np.divide(tp,tp+fp,out=np.zeros_like(tp,dtype=float),where=(tp+fp)>0))
    auc=np.sum(neg*(tp-pos/2))/(P*N);weights=counts[day]
    return np.array([ap,auc,np.dot(weights,error)/weights.sum()])

def paired_intervals(y,a,b,times,replicates=2000,seed=20261011):
    y,a=validate_scores(y,a);_,b=validate_scores(y,b);y=y.astype(float)
    days=pd.to_datetime(times,utc=True).floor('D')
    if len(days)!=len(y) or days.isna().any():raise ValueError('Invalid UTC blocks')
    day,dates=pd.factorize(days);A=_prepared(y,a,day);B=_prepared(y,b,day)
    rng=np.random.default_rng(seed)
    with threadpool_limits(limits=1):
        for _ in range(3):
            counts=np.bincount(rng.integers(0,len(dates),len(dates)),minlength=len(dates));w=counts[day]
            if (y*w).sum()==0 or ((1-y)*w).sum()==0:continue
            for pre,p in [(A,a),(B,b)]:
                expected=[average_precision_score(y,p,sample_weight=w),roc_auc_score(y,p,sample_weight=w),brier_score_loss(y,p,sample_weight=w)]
                np.testing.assert_allclose(_weighted(pre,counts,day),expected,rtol=1e-10,atol=1e-12)
        values=[];skipped=0;rng=np.random.default_rng(seed)
        for i in range(replicates):
            counts=np.bincount(rng.integers(0,len(dates),len(dates)),minlength=len(dates))
            x=_weighted(A,counts,day);z=_weighted(B,counts,day)
            if x is None:skipped+=1
            else:values.append(z-x)
            if (i+1)%250==0:print(f'UTC-day bootstrap: {i+1}/{replicates}',flush=True)
        if not values:raise ValueError('No valid replicates')
        lower,upper=np.quantile(values,[.025,.975],axis=0)
        difference=_weighted(B,np.ones(len(dates)),day)-_weighted(A,np.ones(len(dates)),day)
    return pd.DataFrame({'metric':['AP','ROC_AUC','Brier'],'difference':difference,'lower_95':lower,'upper_95':upper}),{'replicates':replicates,'valid':len(values),'skipped':skipped,'days':len(dates),'seed':seed,'weighted_formulas_checked':True}
