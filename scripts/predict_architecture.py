"""Predict a reference rat IV-bolus plasma curve using a trusted local checkpoint.

This interface intentionally rejects unsupported routes/species rather than
presenting a rat plasma prototype as the intended human-organ product.
"""
import argparse
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import torch
from osmo.architectures import CurveModel
from osmo.chemistry import features


def predict(checkpoint, smiles, time_h, dose_mg_kg):
    checkpoint = Path(checkpoint)
    times, doses = np.asarray(time_h, dtype=np.float32), np.asarray(dose_mg_kg, dtype=np.float32)
    if len(smiles) != len(times) or doses.shape != times.shape:
        raise ValueError('Each structure needs a time and dose')
    if not np.all(np.isfinite(times) & (times >= 0) & np.isfinite(doses) & (doses > 0)):
        raise ValueError('Times must be finite nonnegative hours; doses finite positive mg/kg')
    raw = np.stack([features(s) for s in smiles])
    if checkpoint.suffix == '.joblib':
        saved = joblib.load(checkpoint)
        x = (raw - saved['mean']) / saved['std']
        latent = saved['stage1'].predict(x)
        downstream = np.column_stack([x, latent, np.log1p(times), np.log10(doses)])
        return 10 ** saved['stage2'].predict(downstream) * doses
    saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
    x = torch.tensor((raw - saved['feature_mean']) / saved['feature_std'])
    model = CurveModel(saved['kind'], **saved['scales'])
    model.load_state_dict(saved['model'])
    model.steps = saved['evaluation_steps']
    model.eval()
    with torch.no_grad():
        return 10 ** model(x, torch.tensor(times), torch.tensor(doses)).numpy()


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', type=Path, required=True, help='Trusted local .pt or .joblib artifact only')
    p.add_argument('--smiles', required=True)
    p.add_argument('--dose-mg-kg', type=float, required=True)
    p.add_argument('--times-h', type=float, nargs='+', required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    torch.set_num_threads(2)
    prediction = predict(a.checkpoint, [a.smiles] * len(a.times_h), a.times_h, [a.dose_mg_kg] * len(a.times_h))
    frame = pd.DataFrame(dict(time_h=a.times_h, prediction_ng_ml=prediction, species='rat', tissue='plasma',
                              route='iv_bolus', dose_mg_kg=a.dose_mg_kg, status='research_unvalidated'))
    a.output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(a.output, index=False)
