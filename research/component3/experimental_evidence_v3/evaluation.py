"""Matched prediction metrics and separate machine/date cluster sensitivity."""
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


def top_membership(frame, p, k=10):
    if frame.duplicated(['machineID', 'prediction_time']).any():
        raise ValueError('Duplicate machine-date keys')
    result = np.zeros(len(frame), dtype=bool)
    for _, positions in frame.groupby('prediction_time', sort=False).indices.items():
        order = np.lexsort((frame.iloc[positions].machineID.to_numpy(), -np.asarray(p)[positions]))
        result[np.asarray(positions)[order[:k]]] = True
    return result


def metrics(frame, y, p):
    y, p = np.asarray(y), np.asarray(p)
    selected = top_membership(frame, p)
    positive = int(y.sum()); captured = int(y[selected].sum())
    bins = []; assigned = np.minimum((p * 10).astype(int), 9)
    for b in range(10):
        mask = assigned == b
        bins.append(dict(bin=b + 1, lower=b/10, upper=(b+1)/10, count=int(mask.sum()),
                         mean_probability=float(p[mask].mean()) if mask.any() else np.nan,
                         observed_rate=float(y[mask].mean()) if mask.any() else np.nan))
    ece = sum(b['count'] * abs(b['mean_probability'] - b['observed_rate']) for b in bins if b['count']) / len(y)
    return dict(cases=len(y), positive_cases=positive, prevalence=float(y.mean()), brier=float(np.mean((p-y)**2)),
                average_precision=float(average_precision_score(y,p)) if positive else np.nan,
                roc_auc=float(roc_auc_score(y,p)) if len(np.unique(y)) == 2 else np.nan,
                ece_10_equal_width=ece, mean_probability=float(p.mean()), calibration_bias=float(p.mean()-y.mean()),
                precision_at_10=captured/int(selected.sum()), recall_at_10=captured/positive if positive else np.nan,
                top10_captured=captured, flagged_windows=int(selected.sum())), bins, selected


def cluster_weights(labels, draws=1000, seed=42):
    _, code = np.unique(np.asarray(labels), return_inverse=True)
    n = code.max() + 1
    counts = np.random.default_rng(seed).multinomial(n, np.full(n, 1/n), size=draws)
    return counts[:, code].astype(float)


def weighted_ap(y, p, weights):
    """Vectorized sklearn-equivalent weighted AP, preserving score ties."""
    y, p, w = np.asarray(y), np.asarray(p), np.asarray(weights)
    order = np.argsort(-p, kind='stable')
    ends = np.r_[np.flatnonzero(np.diff(p[order])), len(p)-1]
    positive = np.cumsum(w[:,order] * y[order], axis=1)[:,ends]
    total = np.cumsum(w[:,order], axis=1)[:,ends]
    increments = np.diff(positive, prepend=0, axis=1)
    precision = np.divide(positive, total, out=np.zeros_like(positive), where=total>0)
    denom = positive[:,-1]
    return np.divide((increments*precision).sum(axis=1), denom, out=np.full(len(w), np.nan), where=denom>0)


def bootstrap_metrics(y, p, selected, weights):
    denom = weights.sum(axis=1); positives = weights @ y
    captured = weights @ (y * selected); flagged = weights @ selected.astype(float)
    return dict(brier=(weights @ ((p-y)**2))/denom,
                average_precision=weighted_ap(y,p,weights),
                precision_at_10=np.divide(captured,flagged,out=np.full(len(weights),np.nan),where=flagged>0),
                recall_at_10=np.divide(captured,positives,out=np.full(len(weights),np.nan),where=positives>0))


def intervals(current, reference):
    rows=[]
    for name in current:
        delta=current[name]-reference[name]; valid=delta[np.isfinite(delta)]
        rows.append(dict(metric=name, low=float(np.quantile(valid,.025)) if len(valid) else np.nan,
                         high=float(np.quantile(valid,.975)) if len(valid) else np.nan, valid_draws=len(valid)))
    return rows
