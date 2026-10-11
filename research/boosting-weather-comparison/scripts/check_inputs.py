"""Pretraining architecture check for supervised binary GoA classification."""
from pathlib import Path
import sys,json
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT/'src'))
from approach_risk.data.registry import dataset
from approach_risk.features.c40 import prepare_features

config=json.loads((ROOT/'configs/boosting_weather.json').read_text())
ds=dataset(config['dataset']);f=next(ds.iter_split('train'))
for variant in config['variants']:
    x=prepare_features(f,weather=variant['weather'])
    assert len(x)==len(f)
    print(variant['name'],len(x),'rows',len(x.columns),'predictors')
assert config['task']=='supervised_binary_classification'
print('Shared interfaces ready. No fitting.')
