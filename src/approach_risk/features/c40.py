"""Explicit predictor registry and deterministic selection; no fitting."""
import numpy as np
import pandas as pd
from approach_risk.data.registry import dataset
def feature_registry():
    r = pd.read_csv(dataset().feature_registry_path)
    if not r.field.is_unique or len(r) != 34: raise ValueError('Unexpected registry')
    return r


def feature_names(weather=False, aircraft=True):
    r = feature_registry()
    return [row.field for row in r.itertuples()
            if (weather or row.block == 'nonweather') and (aircraft or row.field != 'typecode')]


def prepare_features(frame, weather=False, aircraft=True):
    """Select only registered predictors; normalize nullable pandas values."""
    fields = feature_names(weather, aircraft)
    out = frame.loc[:, fields].copy()
    registry = feature_registry().set_index('field')
    for col in fields:
        if registry.loc[col, 'kind'] == 'categorical':
            # Missing sentinel is reserved; fail rather than merge it with a real category.
            if out[col].eq('__MISSING__').any(): raise ValueError('Reserved category in data')
            out[col] = out[col].astype(object).where(out[col].notna(), np.nan)
        else:
            out[col] = pd.to_numeric(out[col], errors='raise').astype(float)
    return out

