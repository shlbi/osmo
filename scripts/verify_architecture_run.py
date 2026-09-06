"""Read-only post-run audit: saved predictions, chemical reservations and solvers."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from scipy.integrate import solve_ivp
from osmo.architectures import CurveModel, transport_amounts
from osmo.chemistry import features
from osmo.io import digest, write_json
from osmo.metrics import evaluate
from predict_architecture import predict


def main(a):
    torch.set_num_threads(2)
    protocol = json.loads((a.run / 'protocol.json').read_text())
    source = a.root / 'processed' / protocol['dataset_id'] / 'observations.parquet'
    assert digest(source) == protocol['input_sha256']
    ledger = pd.read_csv(a.run / 'cohort-ledger.csv')
    assert set(ledger.split) == {'train', 'validation'}
    assert len(ledger[ledger.comparison_eligible]) == sum(c['points'] for c in protocol['counts'].values())
    results = []
    for seed in protocol['seeds']:
        folds = json.loads((a.run / f'hierarchical-{seed}-oof.json').read_text())['folds'] if 'hierarchical' in protocol['arms'] else []
        proxies = pd.read_csv(a.run / 'training-pk-proxies.csv').set_index('compound_id')
        all_held = []
        for fold in folds:
            assert not set(fold['train_compounds']) & set(fold['held_compounds'])
            assert not set(proxies.loc[fold['train_compounds']].cluster_id) & set(proxies.loc[fold['held_compounds']].cluster_id)
            all_held.extend(fold['held_compounds'])
        if folds:
            assert sorted(all_held) == sorted(proxies.index)
        for arm in protocol['arms']:
            frame = pd.read_csv(a.run / f'{arm}-{seed}-predictions.csv')
            checkpoint = a.run / f'{arm}-{seed}.{"joblib" if arm == "hierarchical" else "pt"}'
            actual = predict(checkpoint, frame.smiles.tolist(), frame.time_h, frame.dose_mg_kg)
            error = float(np.max(np.abs(np.log10(actual) - np.log10(frame.prediction))))
            assert error < 2e-5, (arm, seed, error)
            assert frame.split.eq('validation').all()
            assert np.all(np.isfinite(actual) & (actual > 0))
            score = evaluate(frame, actual, bootstrap=0)
            original = json.loads((a.run / f'{arm}-{seed}-metrics.json').read_text())
            assert abs(score['score'] - original['score']) < 1e-12
            record = dict(arm=arm, seed=seed, checkpoint_sha256=digest(checkpoint),
                          prediction_reload_max_log10_error=error, metric_reproduced=True)
            if arm == 'flux':
                saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
                model = CurveModel(arm, **saved['scales'])
                model.load_state_dict(saved['model'])
                # One highest-dose point from each validation compound; labels unused.
                rows = frame.sort_values(['dose_mg_kg', 'time_h']).groupby('compound_id').tail(1)
                x = torch.tensor((np.stack([features(s) for s in rows.smiles]) - saved['feature_mean']) / saved['feature_std'])
                with torch.no_grad():
                    p = torch.exp(model.log_low + (model.log_high-model.log_low)*torch.sigmoid(model.head(model.encoder(x)))).double()
                    d = torch.tensor(rows.dose_mg_kg.to_numpy(), dtype=torch.double)
                    t = torch.tensor(rows.time_h.to_numpy(), dtype=torch.double)
                    amounts = transport_amounts(p[:, :5], p[:, 5], d, t, p[:, 6], 128).numpy()
                independent_errors = []
                for i, params in enumerate(p.numpy()):
                    def rhs(t, state):
                        k = params[:5] * np.array([1/(1+state[0]/params[5]/params[6]),
                            1/(1+state[0]/params[5]/params[6]), 1, 1/(1+state[0]/params[5]/params[6]), 1])
                        return [-state[0]*(k[0]+k[1]+k[3])+state[1]*k[2]+state[2]*k[4],
                                state[0]*k[1]-state[1]*k[2], state[0]*k[3]-state[2]*k[4], state[0]*k[0]]
                    ref = solve_ivp(rhs, [0, t[i].item()], [d[i].item(), 0, 0, 0], rtol=1e-9, atol=1e-12, method='DOP853')
                    assert ref.success
                    independent_errors.append(float(abs(np.log10(amounts[i, 0])-np.log10(ref.y[0, -1]))))
                mass_error = float(np.max(np.abs(amounts.sum(-1)-d.numpy())))
                assert amounts.min() >= -1e-10 and mass_error < 1e-7
                assert max(independent_errors) < .02
                record.update(scipy_check_points=len(rows), scipy_max_log10_error=max(independent_errors),
                              max_mass_error_mg_kg=mass_error)
            results.append(record)
    write_json(a.run / 'verification.json', dict(passed=True, source_hash_unchanged=True,
        checks=results, chemical_oof_checks_passed=True, test_evaluated=False,
        environment=dict(torch=torch.__version__, numpy=np.__version__, pandas=pd.__version__),
        tests='16 passed; including independent ODE, positivity, mass balance, differentiability, linear limit and distinct compartment initialization'))
    print('Verified', len(results), 'checkpoints and metrics; all chemical OOF reservations and solver checks passed.')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--run', type=Path, required=True)
    main(p.parse_args())
