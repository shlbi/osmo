"""Metrics use observations only; synthetic or fitted outputs are not truth."""
import numpy as np
import pandas as pd


def evaluate(frame, predictions, bootstrap=1000, seed=4107):
    required = {'concentration', 'compound_id', 'curve_id', 'tissue', 'cluster_id'}
    if not required.issubset(frame):
        raise ValueError(f'Missing metric fields: {sorted(required - set(frame))}')
    d = frame.reset_index(drop=True).copy()
    pred = np.asarray(predictions, dtype=float)
    if pred.shape != (len(d),):
        raise ValueError('One prediction is required for every observation')
    y = d.concentration.to_numpy(float)
    if len(y) == 0 or not np.all(np.isfinite(y) & (y > 0)):
        raise ValueError('Primary ratio metric requires nonempty positive finite observations')
    valid = np.isfinite(pred) & (pred > 0)
    errs = np.full(len(y), np.inf)
    errs[valid] = np.abs(np.log2(pred[valid]) - np.log2(y[valid]))
    d['hit'] = valid & (errs <= 1 + 1e-12)
    curves = d.groupby(['cluster_id', 'compound_id', 'tissue', 'curve_id'], dropna=False).hit.mean()
    pairs = curves.groupby(level=['cluster_id', 'compound_id', 'tissue']).mean().reset_index()
    tissues = pairs.groupby('tissue').hit.mean()
    score = float(tissues.mean())
    clusters = sorted(pairs.cluster_id.unique())
    rng = np.random.default_rng(seed)
    samples = []
    if len(clusters) >= 2:
        grouped = {c: pairs[pairs.cluster_id == c] for c in clusters}
        for _ in range(bootstrap):
            sampled = pd.concat([grouped[c] for c in rng.choice(clusters, len(clusters), replace=True)])
            samples.append(float(sampled.groupby('tissue').hit.mean().mean()))
    return {
        'metric': 'within_2fold_curve_compound_tissue_macro',
        'score': score, 'pooled_point_score': float(d.hit.mean()),
        'by_tissue': {str(k): float(v) for k, v in tissues.items()},
        'n_points': len(d), 'n_curves': int(d.curve_id.nunique()),
        'n_compounds': int(d.compound_id.nunique()), 'n_clusters': len(clusters),
        'failed_predictions': int((~valid).sum()),
        'log2_mae': float(errs.mean()) if valid.all() else None,
        'gm_absolute_fold_error': float(2 ** errs.mean()) if valid.all() else None,
        'cluster_bootstrap_95ci': [float(x) for x in np.quantile(samples,[.025,.975])] if samples else None,
        'target_70_met_point_estimate': score >= .7,
        'note': 'Resampled chemical clusters; this interval does not account for every source/population bias.'
    }
