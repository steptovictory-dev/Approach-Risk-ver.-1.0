"""Classification metrics, validation thresholds and paired evaluation."""
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, brier_score_loss, roc_curve
FPR_LIMITS=(0.001,0.005,0.01)
def validate_scores(y, scores):
    y = np.asarray(y)
    p = np.asarray(scores, dtype=float)
    if len(y) == 0 or y.ndim != 1 or p.shape != y.shape:
        raise ValueError('Empty or misaligned evaluation')
    if not np.isin(y, [0, 1]).all() or not np.isfinite(p).all() or ((p < 0) | (p > 1)).any():
        raise ValueError('Invalid labels or probabilities')
    return y.astype(bool), p


def metrics(y, scores):
    y, p = validate_scores(y, scores)
    both = len(np.unique(y)) == 2
    return {'rows': len(y), 'GoA': int(y.sum()),
            'AP': float(average_precision_score(y, p)) if both else None,
            'ROC_AUC': float(roc_auc_score(y, p)) if both else None,
            'Brier': float(brier_score_loss(y, p))}


def choose_thresholds(y_validation, scores_validation):
    """Validation only; ties are never split to satisfy the FPR constraint."""
    y, p = validate_scores(y_validation, scores_validation)
    if len(np.unique(y)) != 2: raise ValueError('Validation requires both classes')
    fpr, recall, thresholds = roc_curve(y, p, drop_intermediate=False)
    result = {}
    for limit in FPR_LIMITS:
        allowed = np.flatnonzero(fpr <= limit)
        best_recall = recall[allowed].max()
        # Highest threshold among equal-recall candidates minimizes false alarms.
        idx = allowed[recall[allowed] == best_recall][0]
        threshold = float(thresholds[idx])
        # A finite value above 1 represents no alarms and is valid JSON.
        result[limit] = threshold if np.isfinite(threshold) else float(np.nextafter(1., np.inf))
    return result


def threshold_metrics(y, scores, threshold):
    y, p = validate_scores(y, scores)
    alarm = p >= threshold
    tp = int((alarm & y).sum()); fp = int((alarm & ~y).sum())
    positives = int(y.sum()); negatives = len(y) - positives
    return {'TP': tp, 'FP': fp, 'FN': positives-tp, 'TN': negatives-fp,
            'recall': tp/positives if positives else None,
            'FPR': fp/negatives if negatives else None,
            'precision': tp/(tp+fp) if tp+fp else None}


def calibration_table(y, scores, bins=10):
    y, p = validate_scores(y, scores)
    groups = np.minimum((p*bins).astype(int), bins-1)
    return pd.DataFrame([{'bin': i, 'rows': int((groups == i).sum()),
                         'mean_probability': float(p[groups == i].mean()) if (groups == i).any() else None,
                         'observed_rate': float(y[groups == i].mean()) if (groups == i).any() else None}
                        for i in range(bins)])


def paired_day_bootstrap(y, baseline, weather, times, replicates=2000, seed=20261011):
    """Weather minus baseline; resample UTC days identically for both models."""
    y, a = validate_scores(y, baseline); _, b = validate_scores(y, weather)
    days = pd.to_datetime(times, utc=True).floor('D')
    if len(days) != len(y) or days.isna().any(): raise ValueError('Invalid day blocks')
    codes, unique = pd.factorize(days)
    blocks = [np.flatnonzero(codes == i) for i in range(len(unique))]
    rng = np.random.default_rng(seed); values = []; skipped = 0
    for _ in range(replicates):
        ids = np.concatenate([blocks[i] for i in rng.integers(0, len(blocks), len(blocks))])
        if len(np.unique(y[ids])) != 2:
            skipped += 1; continue
        ma, mb = metrics(y[ids], a[ids]), metrics(y[ids], b[ids])
        values.append([mb[k]-ma[k] for k in ['AP', 'ROC_AUC', 'Brier']])
    if not values: raise ValueError('No valid bootstrap replicates')
    bounds = np.quantile(values, [0.025, 0.975], axis=0)
    return {'replicates': replicates, 'valid': len(values), 'skipped': skipped,
            'differences': {k: {'lower': float(bounds[0,i]), 'upper': float(bounds[1,i])}
                            for i,k in enumerate(['AP', 'ROC_AUC', 'Brier'])}}


def evaluate_pair(baseline, weather, thresholds_baseline, thresholds_weather):
    """Score aligned predictions; thresholds must have been fixed on validation."""
    required = ['source_row','GoA','C40_CROSS_TIME','airport','weather_core_usable','probability']
    for frame in [baseline, weather]:
        if not set(required).issubset(frame.columns) or not frame.source_row.is_unique:
            raise ValueError('Invalid prediction registry')
        if frame[required[:-1]].isna().any().any(): raise ValueError('Missing audit fields')
    a = baseline.sort_values('source_row').reset_index(drop=True)
    b = weather.sort_values('source_row').reset_index(drop=True)
    if not a[required[:-1]].equals(b[required[:-1]]): raise ValueError('Different evaluation populations')
    rows = []
    groupings = [('all', pd.Series('all',index=a.index)), ('airport',a.airport),
                 ('period',pd.to_datetime(a.C40_CROSS_TIME,utc=True).dt.strftime('%Y-%m')),
                 ('weather_coverage',a.weather_core_usable.astype(str))]
    for dimension, groups in groupings:
        for group, ids in groups.groupby(groups).groups.items():
            for model, frame, thresholds in [('baseline',a,thresholds_baseline),('weather',b,thresholds_weather)]:
                q = frame.loc[ids]; result = metrics(q.GoA,q.probability)
                rows.append({'dimension':dimension,'group':group,'model':model,**result})
                for limit in FPR_LIMITS:
                    if limit not in thresholds: raise ValueError('Incomplete validation thresholds')
                    rows.append({'dimension':dimension,'group':group,'model':model,'target_validation_FPR':limit,
                                 'rows':len(q),'GoA':int(q.GoA.sum()),
                                 **threshold_metrics(q.GoA,q.probability,thresholds[limit])})
    return pd.DataFrame(rows)

