"""Supervised binary GoA classification with CatBoost; no implicit fitting."""
from catboost import CatBoostClassifier,Pool
from approach_risk.features.c40 import prepare_features,feature_registry

def boosting_features(frame,weather=False):
    x=prepare_features(frame,weather=weather)
    r=feature_registry().set_index('field')
    categories=[c for c in x if r.loc[c,'kind']=='categorical']
    for c in categories:x[c]=x[c].fillna('__MISSING__').astype(str)
    return x,categories

def pool(frame,weather=False,with_labels=False):
    x,cats=boosting_features(frame,weather)
    return Pool(x,label=frame.GoA.to_numpy(dtype=int) if with_labels else None,cat_features=cats)

def classifier(parameters,train_dir):
    return CatBoostClassifier(**parameters,train_dir=str(train_dir),verbose=50,
                              allow_writing_files=True,use_best_model=False)
