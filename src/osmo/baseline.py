"""First experimental baseline. Sealed test labels are never scored by this module."""
import json
import platform
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import ExtraTreesRegressor
from .chemistry import features
from .io import write_json, digest
from .metrics import evaluate

FORMS=['unknown','solution','suspension','tablet','capsule']


def inputs(d, molecular=True):
    time=np.log1p(d.time_h.to_numpy(float))
    context=np.column_stack([time,np.log1p(d.infusion_h.to_numpy(float)),
        d.route.eq('oral').to_numpy(float),*[d.formulation.eq(f).to_numpy(float) for f in FORMS]])
    if not molecular:return context.astype(np.float32)
    cache={s:features(s) for s in d.smiles.unique()}
    return np.column_stack([np.stack([cache[s] for s in d.smiles]),context]).astype(np.float32)


def train(root, reports, species='rat', tissue='plasma', trees=200):
    root,reports=Path(root),Path(reports)
    latest=json.loads((root/'processed/latest.json').read_text())['dataset_id']
    p=root/'processed'/latest
    d=pd.read_parquet(p/'observations.parquet')
    d=d[d.species.eq(species)&d.tissue.eq(tissue)].copy()
    training=d[d.split.eq('train')].copy()
    validation=d[d.split.eq('validation')].copy()
    if training.compound_id.nunique()<5 or validation.compound_id.nunique()<2:
        raise ValueError('Insufficient independent training/validation compounds for this endpoint')
    if species=='human':
        raise ValueError('Current human records require cohort review before a reference-human baseline')
    out=reports/f'{species}-{tissue}'; out.mkdir(parents=True,exist_ok=True)
    target=np.log10(training.concentration.to_numpy(float)/training.dose_mg_kg.to_numpy(float))
    # Equal compound weight, then equal source curve weight within each compound.
    curve_counts=training.groupby('compound_id').curve_id.transform('nunique')
    points=training.groupby(['compound_id','curve_id']).curve_id.transform('size')
    weights=1/(curve_counts*points)
    weights=weights/weights.mean()
    results=[]
    for molecular,label in [(False,'time_route_reference'),(True,'structure_time_extra_trees')]:
        model=ExtraTreesRegressor(n_estimators=trees,min_samples_leaf=3,
            max_features=.7,random_state=4107,n_jobs=4)
        model.fit(inputs(training,molecular),target,sample_weight=weights)
        pred=10**model.predict(inputs(validation,molecular))*validation.dose_mg_kg.to_numpy(float)
        score=evaluate(validation,pred)
        score.update(model=label,species=species,tissue=tissue,split='validation',dataset_id=latest,
            training_compounds=int(training.compound_id.nunique()),training_points=len(training),
            human_organ_target_established=False,
            limitation='Provisional source records and automated structure mappings; developmental animal endpoint, not human validation.')
        results.append(score)
        output=validation[['record_id','compound_id','curve_id','cluster_id','time_h','dose_mg_kg','route','concentration','concentration_unit']].copy()
        output['prediction']=pred
        output.to_csv(out/f'{label}-validation-predictions.csv',index=False)
        joblib.dump({'estimator':model,'molecular':molecular,'dataset_id':latest,
            'species':species,'tissue':tissue,'output_unit':str(d.concentration_unit.iloc[0]),
            'architecture':'Dose-linear time-conditioned ExtraTrees baseline; not physiological.',
            'reference_body_mass_kg':70.,'forms':FORMS},out/f'{label}.joblib')
        write_json(out/f'{label}-metrics.json',score)
    train_docs=set(training.document_id.dropna());val_docs=set(validation.document_id.dropna())
    summary={'dataset_id':latest,'results':results,'test_evaluated':False,
        'overlapping_train_validation_documents':len(train_docs&val_docs),
        'split_warning':'Scaffold and >=0.70 Morgan similarity grouping; documents can cross chemistry partitions. Source-held-out validation remains required.',
        'python':platform.python_version(),'baseline_code_sha256':digest(Path(__file__)),
        'observations_sha256':digest(p/'observations.parquet')}
    write_json(out/'run.json',summary)
    return summary


def predict(model_path, structure, formulation, dose, times):
    # joblib loading is for trusted, locally generated checkpoints only.
    checkpoint=joblib.load(model_path)
    if dose['unit'] not in {'mg/kg','mg'}:raise ValueError('Supported dose units: mg/kg, mg')
    if dose.get('schedule','single')!='single':raise ValueError('Baseline supports single dose only')
    amount=float(dose['amount'])
    if not np.isfinite(amount) or amount<=0:raise ValueError('Dose must be positive and finite')
    if formulation['route'] not in {'iv','oral'}:raise ValueError('Unsupported route')
    if len(times)==0 or any(not np.isfinite(t) or t<0 for t in times):raise ValueError('Invalid output time grid')
    if dose['unit']=='mg':
        if checkpoint['species']!='human':raise ValueError('Animal pilot requires mg/kg; no animal body weight assumed')
        amount/=checkpoint['reference_body_mass_kg']
    d=pd.DataFrame({'smiles':[structure]*len(times),'time_h':times,
        'infusion_h':float(formulation.get('infusion_h',0)),
        'route':formulation['route'],'formulation':formulation.get('dosage_form','unknown')})
    return {'species':checkpoint['species'],'tissue':checkpoint['tissue'],
        'unit':checkpoint['output_unit'],'time_h':list(times),
        'concentration':(10**checkpoint['estimator'].predict(inputs(d,checkpoint['molecular']))*amount).tolist(),
        'status':'research baseline; no individual clinical or human-organ validation'}
