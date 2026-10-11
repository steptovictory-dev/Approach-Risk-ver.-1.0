"""Binary GoA classification: unfitted logistic L2 baseline."""
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from approach_risk.features.c40 import feature_registry, feature_names
class NumericPreparation(TransformerMixin, BaseEstimator):
    """Train medians, fixed zero for empty columns, and indicators for every field."""
    def fit(self, X, y=None):
        a = np.asarray(X, dtype=float)
        if np.isinf(a).any(): raise ValueError('Infinite numeric feature')
        self.medians_ = np.array([np.nanmedian(c) if np.isfinite(c).any() else 0. for c in a.T])
        self.n_features_in_ = a.shape[1]
        return self

    def transform(self, X):
        a = np.asarray(X, dtype=float)
        if a.shape[1] != self.n_features_in_ or np.isinf(a).any():
            raise ValueError('Invalid numeric features')
        missing = np.isnan(a)
        return np.column_stack([np.where(missing, self.medians_, a), missing.astype(float)])


def build_pipeline(weather=False, aircraft=True):
    """Construct only: calling this function does not fit any transformation."""
    r = feature_registry().set_index('field')
    fields = feature_names(weather, aircraft)
    numeric = [c for c in fields if r.loc[c, 'kind'] != 'categorical']
    categorical = [c for c in fields if r.loc[c, 'kind'] == 'categorical']
    pre = ColumnTransformer([
        ('numeric', Pipeline([('missing', NumericPreparation()), ('scale', StandardScaler())]), numeric),
        ('categorical', Pipeline([
            ('missing', SimpleImputer(strategy='constant', fill_value='__MISSING__', keep_empty_features=True)),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=True))]), categorical)
    ], remainder='drop', sparse_threshold=1.0)
    return Pipeline([('preprocess', pre), ('model', LogisticRegression(
        C=1.0, solver='lbfgs', max_iter=2000, random_state=20261011))])

