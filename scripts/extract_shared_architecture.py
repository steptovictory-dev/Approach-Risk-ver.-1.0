"""One-time extraction preserving the frozen legacy research implementation."""
from pathlib import Path
import ast
ROOT=Path(__file__).resolve().parents[1]
source=(ROOT/'research/weather-model-comparison/scripts/comparison_pipeline.py').read_text(encoding='utf8')
tree=ast.parse(source); lines=source.splitlines(keepends=True)
def functions(names):
    return '\n\n'.join(''.join(lines[n.lineno-1:n.end_lineno]) for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in names)+'\n'
base=ROOT/'src/approach_risk'
for name in ['data','features','models','evaluation']:
    (base/name).mkdir(parents=True,exist_ok=True)
    (base/name/'__init__.py').write_text('',encoding='utf8')
(base/'__init__.py').write_text('"""Shared Approach Risk research infrastructure."""\n',encoding='utf8')
evaluation='''"""Classification metrics, validation thresholds and paired evaluation."""
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss, roc_curve
FPR_LIMITS=(0.001,0.005,0.01)
'''+functions({'validate_scores','metrics','choose_thresholds','threshold_metrics','calibration_table','paired_day_bootstrap','evaluate_pair'})
(base/'evaluation/classification.py').write_text(evaluation,encoding='utf8')
feature='''"""Explicit predictor registry and deterministic selection; no fitting."""
import numpy as np
import pandas as pd
from approach_risk.data.registry import dataset
'''+functions({'feature_registry','feature_names','prepare_features'})
feature=feature.replace("pd.read_csv(BASE / 'reports/02_feature_registry.csv')","pd.read_csv(dataset().feature_registry_path)")
(base/'features/c40.py').write_text(feature,encoding='utf8')
model='''"""Binary GoA classification: unfitted logistic L2 baseline."""
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from approach_risk.features.c40 import feature_registry, feature_names
'''+functions({'NumericPreparation','build_pipeline'})
(base/'models/logistic.py').write_text(model,encoding='utf8')
print('Extracted shared code; legacy implementation unchanged')
