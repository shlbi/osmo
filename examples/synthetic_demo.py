"""Synthetic mass-balance demonstration. No measured data or trained model."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch
from osmo.mechanistic import amounts

def main():
    torch.set_num_threads(2)
    time=torch.linspace(0,24,97,dtype=torch.float64)
    # ka [1/h], CL [L/h/kg], Vc/Vp [L/kg], Q [L/h/kg], F [fraction].
    parameters=torch.tensor([[1.,.2,.3,.8,.5,.7]],dtype=torch.float64).repeat(len(time),1)
    state=amounts(parameters,time,torch.zeros_like(time),torch.ones_like(time))
    delivered=torch.minimum(time,torch.ones_like(time))
    error=float((state.sum(-1)-delivered).abs().max())
    assert error<1e-8 and state.min()>-1e-10
    out=Path('demo-output');out.mkdir(exist_ok=True)
    pd.DataFrame({'time_h':time.numpy(),'synthetic_plasma_ng_ml':state[:,1].numpy()/.3*1000,
        'total_accounted_amount_mg_kg':state.sum(-1).numpy()}).to_csv(out/'synthetic-curve.csv',index=False)
    summary={'input_type':'synthetic','scenario':'1 mg/kg over a one-hour IV infusion',
             'mass_balance_max_absolute_error':error,'points':len(time),'clinical_accuracy_measured':False}
    (out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf8')
    print(json.dumps(summary,indent=2))

if __name__=='__main__':main()
