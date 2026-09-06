"""Frozen-development IV-bolus comparison. Never scores sealed test chemistry.

No restricted paper code/data is imported. This is a new experiment, not a
reproduction of published percentages. Source rows remain provisionally curated.
"""
import argparse
import copy
import json
import time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import torch
from scipy.optimize import least_squares
from scipy.special import logsumexp
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.model_selection import GroupKFold
from osmo.architectures import CurveModel
from osmo.chemistry import features
from osmo.io import digest, write_json
from osmo.metrics import evaluate


def fit_exponentials(t, y, n=2):
    """Positive exponentials fitted to dose-normalized observed concentrations.

    Used only as training-derived proxy labels or capacity diagnostics. These
    fitted parameters are not measured pharmacokinetic ground truth.
    """
    t, y = np.asarray(t, float), np.asarray(y, float)
    def predict(p, times):
        return logsumexp(p[:n, None] - np.exp(p[n:, None]) * times[None, :], axis=0) / np.log(10)
    best = None
    for rate_scale in [0.03, 0.5, 3.]:
        init = np.r_[np.full(n, np.log(np.max(y) / n)), np.log(np.geomspace(rate_scale, rate_scale * 10, n))]
        init = np.clip(init, np.r_[np.full(n, -15), np.full(n, -14)] + 1e-5,
                       np.r_[np.full(n, 25), np.full(n, 5)] - 1e-5)
        fit = least_squares(lambda p: predict(p, t) - np.log10(y), init,
                            bounds=(np.r_[np.full(n, -15), np.full(n, -14)],
                                    np.r_[np.full(n, 25), np.full(n, 5)]),
                            loss='soft_l1', max_nfev=200)
        if best is None or fit.cost < best.cost:
            best = fit
    amps, rates = np.exp(best.x[:n]), np.exp(best.x[n:])
    auc = np.sum(amps / rates)
    cl = 1000 / auc
    vss = cl * np.sum(amps / rates**2) / auc
    vc = 1000 / amps.sum()
    return dict(params=best.x, log_pk=np.log10([cl, vss, vc]), success=bool(best.success),
                predict=lambda times: predict(best.x, np.asarray(times)))


def macro_weights(frame):
    w = 1 / (frame.groupby('compound_id').curve_id.transform('nunique') *
             frame.groupby(['compound_id', 'curve_id']).curve_id.transform('size'))
    return (w / w.mean()).to_numpy(np.float32)


def training_diagnostics(pk, output):
    tr = pk[pk.split.eq('train')].copy()
    tr['normalized'] = tr.concentration / tr.dose_mg_kg
    proxies = []
    for compound, group in tr.groupby('compound_id'):
        curve = group.groupby('time_h').normalized.median()
        fit = fit_exponentials(curve.index.to_numpy(), curve.to_numpy())
        proxies.append(dict(compound_id=compound, cluster_id=group.cluster_id.iloc[0],
                            log_cl_proxy=fit['log_pk'][0], log_vss_proxy=fit['log_pk'][1],
                            log_vc_proxy=fit['log_pk'][2], fit_converged=fit['success']))
    pd.DataFrame(proxies).to_csv(output / 'training-pk-proxies.csv', index=False)
    records, predictions = [], []
    # Study arms retain dose and sex; median across subjects at each time.
    # This is a population shape diagnostic, not individual longitudinal data.
    for key, group in tr.groupby(['compound_id', 'study_id', 'dose_mg_kg', 'sex'], dropna=False):
        curve = group.groupby('time_h').normalized.median().sort_index()
        if len(curve) < 6:
            continue
        times = curve.index.to_numpy()
        cut = max(4, int(np.floor(len(curve) * .75)))
        for n in [1, 2, 3]:
            fitted = fit_exponentials(times[:cut], curve.to_numpy()[:cut], n)
            pred = fitted['predict'](times)
            hit = np.abs(pred - np.log10(curve.to_numpy())) <= np.log10(2)
            records.append(dict(compound_id=key[0], study_id=str(key[1]), dose=float(key[2]), sex=str(key[3]),
                                exponentials=n, n_fit=cut, n_late=len(curve)-cut,
                                fit_score=float(hit[:cut].mean()), late_score=float(hit[cut:].mean()),
                                converged=fitted['success']))
            for i, t in enumerate(times):
                predictions.append(dict(compound_id=key[0], study_id=str(key[1]), dose=float(key[2]), sex=str(key[3]),
                                         exponentials=n, time_h=float(t), role='fit' if i < cut else 'late_holdout',
                                         observed_normalized=float(curve.iloc[i]), prediction_log10_normalized=float(pred[i])))
    pd.DataFrame(records).to_csv(output / 'capacity-diagnostics.csv', index=False)
    pd.DataFrame(predictions).to_csv(output / 'capacity-predictions.csv', index=False)
    return pd.DataFrame(proxies)


def run(a):
    a.output.mkdir(parents=True, exist_ok=True)
    if (a.output / 'protocol.json').exists():
        raise ValueError('Output already has a protocol; use a new directory to preserve prior runs')
    torch.set_num_threads(2)
    ds = json.loads((a.root / 'processed/latest.json').read_text())['dataset_id']
    source = a.root / 'processed' / ds / 'observations.parquet'
    columns = ['record_id', 'compound_id', 'cluster_id', 'curve_id', 'species', 'tissue', 'route',
               'infusion_h', 'split', 'smiles', 'time_h', 'dose_mg_kg', 'concentration', 'study_id', 'sex', 'formulation']
    # Predicate pushdown excludes test records from the comparison frame.
    development = pd.read_parquet(source, columns=columns, filters=[('split', 'in', ['train', 'validation']),
                                                                   ('species', '==', 'rat'), ('tissue', '==', 'plasma')])
    eligible = development.route.eq('iv') & development.infusion_h.eq(0)
    ledger = development[['record_id', 'compound_id', 'split', 'route', 'infusion_h']].copy()
    ledger['comparison_eligible'] = eligible
    ledger['reason'] = np.where(eligible, 'IV bolus development cohort', 'outside prespecified IV bolus scope')
    ledger.to_csv(a.output / 'cohort-ledger.csv', index=False)
    pk = development[eligible].reset_index(drop=True)
    if not (pk.concentration.gt(0) & pk.dose_mg_kg.gt(0) & pk.time_h.ge(0)).all():
        raise ValueError('Invalid positive-concentration IV data')
    train = np.flatnonzero(pk.split.eq('train'))
    val = np.flatnonzero(pk.split.eq('validation'))
    assert not set(pk.iloc[train].cluster_id) & set(pk.iloc[val].cluster_id)
    cache = {s: features(s) for s in pk.smiles.unique()}
    ref = np.stack([cache[s] for s in pk.iloc[train].smiles.unique()])
    mean, std = ref.mean(0), ref.std(0)
    std[std < .01] = 1
    x = ((np.stack([cache[s] for s in pk.smiles]) - mean) / std).astype(np.float32)
    tx = torch.tensor(x)
    times = torch.tensor(pk.time_h.to_numpy(np.float32))
    dose = torch.tensor(pk.dose_mg_kg.to_numpy(np.float32))
    target = torch.tensor(np.log10(pk.concentration.to_numpy()).astype(np.float32))
    weights = np.zeros(len(pk), np.float32)
    weights[train], weights[val] = macro_weights(pk.iloc[train]), macro_weights(pk.iloc[val])
    tw = torch.tensor(weights)
    scales = dict(time_scale=max(float(np.log1p(pk.iloc[train].time_h).max()), 1.),
                  dose_center=float(np.log10(pk.iloc[train].dose_mg_kg).mean()),
                  dose_scale=max(float(np.log10(pk.iloc[train].dose_mg_kg).std()), .1))
    counts = {s: dict(points=len(g), compounds=g.compound_id.nunique(), clusters=g.cluster_id.nunique(),
                      curves=g.curve_id.nunique()) for s, g in pk.groupby('split')}
    counts = {s: {k: int(v) for k, v in c.items()} for s, c in counts.items()}
    write_json(a.output / 'protocol.json', dict(dataset_id=ds, input_sha256=digest(source), counts=counts,
               scope='provisional rat IV bolus plasma; all eligible development records; no error-based exclusions',
               seeds=a.seeds, arms=a.arms, epochs=a.epochs, seconds_per_neural_arm=a.seconds,
               neural_lr=.001, batch_size=128, patience=15, encoder=[64, 32],
               selection='curve/compound-weighted validation log10 MAE',
               nonlinear_train_steps=32, nonlinear_selection_steps=64, nonlinear_final_steps=128,
               nonlinear_convergence_steps=256, nonlinear_tolerance_max_log10=.02,
               numerical_failures='reported, never removed from scoring',
               budget_note='Same neural epoch/time ceilings; trees use fixed 200 estimators. This is not equal-FLOP tuning.',
               chemical_split='existing frozen global clusters; no validation or test auxiliary targets',
               test_evaluated=False, human_organ_target_established=False,
               data_review='source rows retain automated provisional eligibility; sparse original curve IDs',
               code_sha256={p.name: digest(p) for p in
                            [Path(__file__), Path(__file__).parents[1] / 'src/osmo/architectures.py']}))
    np.savez(a.output / 'feature-transform.npz', mean=mean, std=std)
    proxies = training_diagnostics(pk, a.output)
    results = []

    def record(kind, seed, pred, metadata):
        vf = pk.iloc[val].copy()
        vf['prediction'] = pred
        vf.to_csv(a.output / f'{kind}-{seed}-predictions.csv', index=False)
        score = evaluate(vf, pred)
        score.update(arm=kind, seed=seed, test_evaluated=False, human_organ_target_established=False, **metadata)
        write_json(a.output / f'{kind}-{seed}-metrics.json', score)
        results.append({k: v for k, v in score.items() if k != 'history'})
        write_json(a.output / 'comparison.json', dict(results=results, test_evaluated=False))
        print(json.dumps(dict(arm=kind, seed=seed, macro=score['score'], pooled=score['pooled_point_score'],
                              seconds=metadata['seconds'])), flush=True)

    for seed in a.seeds:
        for kind in a.arms:
            torch.manual_seed(seed)
            rng = np.random.default_rng(seed)
            started = time.monotonic()
            if kind == 'hierarchical':
                compounds = pk.iloc[train].drop_duplicates('compound_id').set_index('compound_id').loc[proxies.compound_id]
                cx = np.stack([(cache[s] - mean) / std for s in compounds.smiles])
                cy = proxies[['log_cl_proxy', 'log_vss_proxy', 'log_vc_proxy']].to_numpy()
                oof = np.full_like(cy, np.nan)
                folds = []
                for fold, (fitids, heldids) in enumerate(GroupKFold(n_splits=min(4, compounds.cluster_id.nunique())).split(cx, cy, compounds.cluster_id)):
                    forest = ExtraTreesRegressor(n_estimators=200, min_samples_leaf=2, random_state=seed, n_jobs=2)
                    forest.fit(cx[fitids], cy[fitids])
                    oof[heldids] = forest.predict(cx[heldids])
                    assert not set(compounds.iloc[fitids].cluster_id) & set(compounds.iloc[heldids].cluster_id)
                    folds.append(dict(fold=fold, train_compounds=compounds.iloc[fitids].index.tolist(),
                                      held_compounds=compounds.iloc[heldids].index.tolist()))
                assert np.isfinite(oof).all()
                stage1 = ExtraTreesRegressor(n_estimators=200, min_samples_leaf=2, random_state=seed, n_jobs=2).fit(cx, cy)
                lookup = dict(zip(compounds.index, oof))
                latent_train = np.stack([lookup[c] for c in pk.iloc[train].compound_id])
                latent_val = stage1.predict(x[val])
                def downstream(ids, latent):
                    return np.column_stack([x[ids], latent, np.log1p(pk.iloc[ids].time_h), np.log10(pk.iloc[ids].dose_mg_kg)])
                stage2 = ExtraTreesRegressor(n_estimators=200, min_samples_leaf=3, random_state=seed, n_jobs=2)
                stage2.fit(downstream(train, latent_train), (target - dose.log10()).numpy()[train], sample_weight=weights[train])
                pred = 10 ** stage2.predict(downstream(val, latent_val)) * dose.numpy()[val]
                joblib.dump(dict(stage1=stage1, stage2=stage2, mean=mean, std=std,
                                 scope='rat IV bolus plasma', proxy_columns=['log_cl_proxy','log_vss_proxy','log_vc_proxy']),
                            a.output / f'{kind}-{seed}.joblib')
                write_json(a.output / f'{kind}-{seed}-oof.json', dict(folds=folds, measured_validation_properties_used=False))
                record(kind, seed, pred, dict(seconds=time.monotonic()-started, parameters=None,
                       note='Training-curve-derived latent PK proxies, not measured CL/Vss. Stage 2 uses out-of-fold stage 1 predictions.'))
                continue
            model = CurveModel(kind, **scales)
            opt = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.001)
            best, beststate, bestepoch, stale = float('inf'), None, -1, 0
            history = []
            def predict(ids, steps):
                model.steps = steps
                return torch.cat([model(tx[j], times[j], dose[j]) for j in np.array_split(ids, max(1, int(np.ceil(len(ids)/256))))])
            for epoch in range(a.epochs):
                model.train()
                model.steps = 32
                for j in np.array_split(rng.permutation(train), int(np.ceil(len(train) / 128))):
                    opt.zero_grad(set_to_none=True)
                    pred = model(tx[j], times[j], dose[j])
                    loss = (torch.nn.functional.smooth_l1_loss(pred, target[j], reduction='none') * tw[j]).mean()
                    if not torch.isfinite(loss):
                        raise RuntimeError(f'Nonfinite loss {kind} {seed}')
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 5)
                    opt.step()
                model.eval()
                with torch.no_grad():
                    vl = float(((predict(val, 64) - target[val]).abs() * tw[val]).mean())
                history.append(dict(epoch=epoch, validation_log10_mae=vl, seconds=time.monotonic()-started))
                if vl < best - 1e-5:
                    best, beststate, bestepoch, stale = vl, copy.deepcopy(model.state_dict()), epoch, 0
                else:
                    stale += 1
                if epoch % 10 == 0:
                    print(kind, seed, 'epoch', epoch, 'validation_log10_mae', round(vl, 4), flush=True)
                if stale >= 15 or time.monotonic() - started >= a.seconds:
                    break
            model.load_state_dict(beststate)
            model.eval()
            with torch.no_grad():
                pv = predict(val, 128)
                ptr = predict(train, 128)
                convergence = None
                if kind == 'flux':
                    delta = (pv - predict(val, 256)).abs()
                    coarse_delta = (pv - predict(val, 32)).abs()
                    convergence = dict(max_log10_difference=float(delta.max()), mean_log10_difference=float(delta.mean()),
                                       train_solver_to_final_max_log10=float(coarse_delta.max()),
                                       passed=bool(delta.max() <= .02))
            torch.save(dict(model=beststate, kind=kind, scales=scales, feature_mean=mean, feature_std=std,
                            evaluation_steps=128, scope='rat IV bolus plasma', seed=seed, dataset_id=ds),
                       a.output / f'{kind}-{seed}.pt')
            record(kind, seed, 10 ** pv.numpy(), dict(seconds=time.monotonic()-started, best_epoch=bestepoch,
                   epochs_completed=len(history), history=history, parameters=sum(p.numel() for p in model.parameters()),
                   training_macro=evaluate(pk.iloc[train], 10 ** ptr.numpy(), bootstrap=0)['score'],
                   solver_convergence=convergence))
    summary = pd.DataFrame(results).groupby('arm').agg(macro_mean=('score','mean'), macro_std=('score','std'),
                macro_min=('score','min'), macro_max=('score','max'), pooled_mean=('pooled_point_score','mean'))
    summary.to_csv(a.output / 'summary.csv')
    print(summary.to_string(), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--seeds', nargs='+', type=int, default=[4107, 4108, 4109])
    p.add_argument('--arms', nargs='+', choices=['direct','cmt1','cmt2','cmt3','hierarchical','flux'],
                   default=['direct','cmt1','cmt2','cmt3','hierarchical','flux'])
    p.add_argument('--epochs', type=int, default=100)
    p.add_argument('--seconds', type=int, default=90)
    run(p.parse_args())
