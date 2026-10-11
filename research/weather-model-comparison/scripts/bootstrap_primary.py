"""Paired UTC-day bootstrap, fixed classifiers, exact weighted AP/AUC with ties."""
import json,time
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score,roc_auc_score,brier_score_loss
from comparison_pipeline import BASE

def prepared(y,p,day):
    order=np.argsort(-p,kind='stable'); s=p[order]
    starts=np.r_[0,np.flatnonzero(np.diff(s))+1]
    return y[order],day[order],starts,(p-y)**2

def weighted(pre,counts,day):
    y,d,starts,error=pre; w=counts[d]
    pos=np.add.reduceat(w*y,starts); neg=np.add.reduceat(w*(1-y),starts)
    tp=np.cumsum(pos); fp=np.cumsum(neg); P=tp[-1]; N=fp[-1]
    if not P or not N:return None
    ap=np.sum(pos/P*np.divide(tp,tp+fp,out=np.zeros_like(tp,dtype=float),where=(tp+fp)>0))
    auc=np.sum(neg*(tp-pos/2))/(P*N)
    weights=counts[day]; brier=np.dot(weights,error)/weights.sum()
    return np.array([ap,auc,brier])

def main():
    dest=BASE/'data/primary_models_v1'
    a=pd.read_parquet(dest/'baseline_test_predictions.parquet').sort_values('source_row').reset_index(drop=True)
    b=pd.read_parquet(dest/'weather_test_predictions.parquet').sort_values('source_row').reset_index(drop=True)
    assert a.drop(columns='probability').equals(b.drop(columns='probability'))
    y=a.GoA.to_numpy(dtype=float); day,dates=pd.factorize(pd.to_datetime(a.C40_CROSS_TIME,utc=True).dt.floor('D'))
    pa=a.probability.to_numpy();pb=b.probability.to_numpy(); A=prepared(y,pa,day);B=prepared(y,pb,day)
    # Independently compare tie-aware weighted formulas to sklearn, including missing days.
    rng=np.random.default_rng(20261011)
    for _ in range(3):
        counts=np.bincount(rng.integers(0,len(dates),len(dates)),minlength=len(dates)); w=counts[day]
        for pre,p in [(A,pa),(B,pb)]:
            expected=[average_precision_score(y,p,sample_weight=w),roc_auc_score(y,p,sample_weight=w),brier_score_loss(y,p,sample_weight=w)]
            np.testing.assert_allclose(weighted(pre,counts,day),expected,rtol=1e-10,atol=1e-12)
    rng=np.random.default_rng(20261011);values=[]; skipped=0;start=time.time()
    for i in range(2000):
        counts=np.bincount(rng.integers(0,len(dates),len(dates)),minlength=len(dates))
        x=weighted(A,counts,day);z=weighted(B,counts,day)
        if x is None:skipped+=1
        else:values.append(z-x)
        if (i+1)%250==0:print(f'Bootstrap {i+1}/2000',flush=True)
    values=np.array(values);lower,upper=np.quantile(values,[.025,.975],axis=0)
    observed=weighted(B,np.ones(len(dates)),day)-weighted(A,np.ones(len(dates)),day)
    result=pd.DataFrame({'metric':['AP','ROC_AUC','Brier'],'difference':observed,'lower_95':lower,'upper_95':upper})
    result.to_csv(BASE/'reports/05_primary_day_bootstrap.csv',index=False)
    (BASE/'reports/05_bootstrap_manifest.json').write_text(json.dumps({'replicates':2000,'valid':len(values),'skipped':skipped,'days':len(dates),'seed':20261011,'rows':len(a),'seconds':time.time()-start,'weighted_formulas_checked':True},indent=2)+'\n')
    print(result.to_string(index=False),flush=True)
if __name__=='__main__':main()
