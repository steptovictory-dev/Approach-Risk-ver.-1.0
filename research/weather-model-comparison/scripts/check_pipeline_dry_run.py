"""Check preparation contracts and evaluation on known fixtures. Never call fit."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from comparison_pipeline import (build_pipeline, prepare_features, feature_names,
    metrics, choose_thresholds, threshold_metrics, calibration_table, paired_day_bootstrap,
    evaluate_pair)

BASE = Path(__file__).resolve().parents[1]
m = json.loads((BASE/'reports/02_model_table_manifest.json').read_text(encoding='utf8'))
frame = pd.read_parquet(BASE/'data/model_table_v1'/m['files'][0]['file']).head(1000)
variants = []
for aircraft in [True, False]:
    for weather in [False, True]:
        x = prepare_features(frame, weather, aircraft)
        pipe = build_pipeline(weather, aircraft)
        assert len(x) == len(frame) and list(x) == feature_names(weather, aircraft)
        assert not hasattr(pipe.named_steps['model'], 'coef_')
        variants.append({'weather':weather, 'aircraft':aircraft, 'predictors':len(x.columns)})

# Exact-score ties must not be separated. The fixture admits no false positive.
y = [1, 0, 1, 0]; p = [.9, .8, .8, .1]
thresholds = choose_thresholds(y, p)
assert all(value == .9 for value in thresholds.values())
no_alarm = choose_thresholds([0,1], [1.,1.])
assert all(np.isfinite(v) and v>1. for v in no_alarm.values())
assert all(threshold_metrics([0,1],[1.,1.],v)['FP']==0 for v in no_alarm.values())
assert threshold_metrics(y, p, .9) == {'TP':1,'FP':0,'FN':1,'TN':2,'recall':.5,'FPR':0.,'precision':1.}
assert metrics([0,0], [.1,.2])['AP'] is None
assert threshold_metrics([0,0], [.1,.2], .9)['precision'] is None
assert calibration_table(y,p).rows.sum() == 4
assert metrics([0,1], [0.,1.])['AP'] == 1.
assert metrics([0,1], [0.,1.])['Brier'] == 0.
interval = paired_day_bootstrap(y, p, p, ['2023-01-01']*2+['2023-01-02']*2, replicates=30)
assert all(v['lower']==v['upper']==0. for v in interval['differences'].values())
pred = pd.DataFrame({'source_row':[0,1,2,3], 'GoA':y, 'C40_CROSS_TIME':pd.to_datetime(['2023-01-01']*4,utc=True),
                     'airport':['AAA']*4,'weather_core_usable':[True,False,True,False],'probability':p})
assert len(evaluate_pair(pred,pred.iloc[::-1],thresholds,thresholds)) > 0
try:
    evaluate_pair(pred,pred.iloc[:3],thresholds,thresholds)
except ValueError:
    pass
else:
    raise AssertionError('Unequal populations accepted')

result = {'verified':True,'variants':variants,'rows_checked':len(frame),
          'fit_called':False,'training_executed':False,
          'checks':['registered predictors only','unfitted pipelines','tied thresholds',
                    'undefined metrics','calibration counts','known perfect scores','paired UTC blocks',
                    'prediction alignment','reject unequal populations']}
(BASE/'reports/03_pipeline_dry_run.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
print(json.dumps(result,ensure_ascii=False))
