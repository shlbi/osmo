"""CvTdb v2 adapter. Raw measurements remain immutable; all exclusions are explicit."""
import hashlib
import json
import sqlite3
import time
from pathlib import Path
from urllib.parse import quote
import numpy as np
import pandas as pd
from .io import download, write_json, session, digest
from .chemistry import identity, assign_splits

ROOT_URL='https://raw.githubusercontent.com/USEPA/CompTox-PK-CvTdb/master/v2.0.0/'
FILE='cvtdb_v2_0_0_no_audit.sqlite'
LICENSE_API='https://api.figshare.com/v2/articles/29610452'


def acquire(root):
    root=Path(root)
    s=session(); r=s.get(LICENSE_API,timeout=60); r.raise_for_status()
    meta=r.json()
    if meta.get('license',{}).get('name')!='CC0':
        raise ValueError('CvTdb release license changed; review before training')
    write_json(root/'raw/cvtdb/release_metadata.json',meta)
    record=download(ROOT_URL+FILE,root/'raw/cvtdb'/FILE)
    record.update(source_id='EPA-CvTdb-2.0.0',license='CC0',license_evidence=LICENSE_API,
                  permitted_use='public release, including research model training',
                  observation_kind='experimental; calculated TK parameters excluded')
    write_json(root/'manifests/cvtdb.json',record)
    return record


def load_joined(root):
    # The distributed SQLite lacks useful join indexes. Pandas joins avoid expensive SQL scans.
    c=sqlite3.connect(f'file:{(Path(root)/"raw/cvtdb"/FILE).as_posix()}?mode=ro',uri=True)
    def table(name): return pd.read_sql_query(f'SELECT * FROM "{name}"',c)
    v=table('conc_time_values').rename(columns={'id':'observation_id','qc_status':'point_qc'})
    series=table('series').rename(columns={'id':'source_curve_id','qc_status':'series_qc'})
    studies=table('studies').rename(columns={'id':'study_id','qc_status':'study_qc'})
    subjects=table('subjects').rename(columns={'id':'subject_id','qc_status':'subject_qc'})
    chemicals=table('chemicals')
    d=v.merge(series,on=None,left_on='fk_series_id',right_on='source_curve_id',suffixes=('','_series'),validate='many_to_one')
    d=d.merge(studies,left_on='fk_study_id',right_on='study_id',suffixes=('','_study'),validate='many_to_one')
    d=d.merge(subjects,left_on='fk_subject_id',right_on='subject_id',suffixes=('','_subject'),validate='many_to_one')
    d=d.merge(chemicals[['id','preferred_name','dsstox_casrn','dsstox_substance_id']],left_on='fk_dosed_chemical_id',right_on='id',validate='many_to_one')
    for name,key,value,out in [
        ('conc_medium_dict','fk_conc_medium_id','conc_medium_normalized','tissue'),
        ('administration_route_dict','fk_administration_route_id','administration_route_normalized','route')]:
        mapping=table(name).set_index('id')[value]
        d[out]=d[key].map(mapping)
    docs=table('documents').set_index('id')
    d['document_id']=d.fk_reference_document_id.fillna(d.fk_extraction_document_id)
    for col in ['doi','pmid','title','url']:
        d['source_'+col]=d.document_id.map(docs[col])
    c.close()
    return d


def unit_conversion(unit,tissue):
    u=str(unit).strip().lower().replace('μ','u').replace('µ','u').replace(' ','')
    liquid={'pg/ml':.001,'ng/ml':1.,'ug/ml':1000.,'mg/ml':1e6,'ng/l':.001,'ug/l':1.,'mg/l':1000.}
    solid={'pg/g':.001,'ng/g':1.,'ug/g':1000.,'mg/g':1e6,'mg/kg':1000.}
    # Never silently equate a gram of tissue with a millilitre of fluid.
    if tissue=='plasma' and u in liquid: return liquid[u], 'ng/mL'
    if tissue in {'liver','kidney','lung','heart','brain'} and u in solid: return solid[u], 'ng/g'
    return None,None


def resolve_chemical(cas, root):
    p=Path(root)/'raw/pubchem'/f'{cas}.json'
    url=f'https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{quote(cas,safe="")}/property/IsomericSMILES,MolecularWeight/JSON'
    if not p.exists():
        r=session().get(url,timeout=40)
        if r.status_code==404: return None,'CAS not found in PubChem'
        r.raise_for_status(); write_json(p,r.json()); time.sleep(.25)
    raw=json.loads(p.read_text(encoding='utf-8'))
    rows=raw.get('PropertyTable',{}).get('Properties',[])
    if len(rows)!=1: return None,'Ambiguous PubChem CAS mapping'
    row=rows[0]; smiles=row.get('SMILES') or row.get('IsomericSMILES') or row.get('ConnectivitySMILES')
    if not smiles:return None,'Missing structure'
    # Multi-component salts need administered-form/dose-basis review before inclusion.
    if '.' in smiles:return None,'Multicomponent administered substance needs dose-basis review'
    ident=identity(smiles)
    ident.update(cas=cas,pubchem_cid=row['CID'],structure_source_url=url,
                 structure_source_sha256=digest(p),mapping_method='single-result CAS lookup; review pending')
    return ident,None


def prepare(root, reports):
    root,reports=Path(root),Path(reports)
    reports.mkdir(parents=True,exist_ok=True)
    d=load_joined(root)
    parent=d.fk_analyzed_chemical_id.eq(d.fk_dosed_chemical_id)
    coverage=d[parent].groupby(['species','tissue'],dropna=False).agg(
        points=('observation_id','count'),source_series=('source_curve_id','nunique'),
        chemicals=('fk_dosed_chemical_id','nunique')).reset_index()
    coverage.to_csv(reports/'cvtdb-parent-coverage.csv',index=False)
    d['exclusion_reason']=''
    def exclude(mask,reason):
        d.loc[mask,'exclusion_reason'] += reason+';'
    exclude(~parent,'analyzed_and_dosed_chemical_ids_differ')
    exclude(~d.radiolabeled.eq(0),'radiolabel_or_unspecified')
    exclude(~d.route.isin(['iv','oral']),'route_outside_initial_scope')
    exclude(~d.tissue.isin(['plasma','liver','kidney','lung','heart','brain']),'matrix_outside_initial_scope')
    exclude(~d.dose_frequency_original.fillna('').str.lower().str.strip().isin(['single dose','1']),'not_explicit_single_dose')
    exclude(~d.dose_level_units_normalized.eq('mg/kg BW'),'dose_not_normalized_mg_per_kg')
    d['dose_mg_kg']=pd.to_numeric(d.dose_level_normalized,errors='coerce')
    d['time_h']=pd.to_numeric(d.time_hr,errors='coerce')
    d['concentration_raw_numeric']=pd.to_numeric(d.conc_original,errors='coerce')
    exclude(~np.isfinite(d.dose_mg_kg)|(d.dose_mg_kg<=0),'invalid_dose')
    exclude(~np.isfinite(d.time_h)|(d.time_h<0),'invalid_time')
    exclude(~np.isfinite(d.concentration_raw_numeric)|(d.concentration_raw_numeric<=0),'nonpositive_or_missing_observation')
    exclude(d.no_conc_val_type.notna(),'censored_or_missing_measurement')
    exclude(d.log_concentration_values.eq(1),'log_encoded_original_requires_review')
    exclude(d.conc_cumulative.eq(1),'cumulative_measurement')
    conversions=[unit_conversion(u,t) for u,t in zip(d.conc_units_original,d.tissue)]
    d['conversion_factor']=[a for a,b in conversions]
    d['concentration_unit']=[b for a,b in conversions]
    exclude(d.conversion_factor.isna(),'original_concentration_unit_requires_review')
    d['concentration']=d.concentration_raw_numeric*d.conversion_factor
    for col in ['point_qc','series_qc','study_qc','subject_qc']:
        exclude(d[col].notna()&~d[col].eq('pass'),'source_qc_not_pass')
    # Exact formulations are not present in the normalized form dictionary of this release.
    d['formulation']=d.administration_form_original.fillna('unknown').str.lower()
    d['administration_method']=d.administration_method_original.fillna('unknown').str.lower()
    dur=pd.to_numeric(d.dose_duration,errors='coerce')
    dur_factor=d.dose_duration_units.str.lower().map({'minutes':1/60,'minute':1/60,'seconds':1/3600,'hours':1.,'h':1.})
    d['infusion_h']=(dur*dur_factor).where(d.administration_method.str.contains('infusion'),0.)
    exclude(d.infusion_h.isna(),'unknown_infusion_duration')
    exclude(d.administration_method.str.contains('arterial'),'intraarterial_requires_separate_model')
    exclude(d.dsstox_casrn.isna(),'missing_structure_identifier')
    candidates=d[d.exclusion_reason.eq('')]
    ids=[]; mapping_errors={}
    cas_values=sorted(candidates.dsstox_casrn.unique())
    for i,cas in enumerate(cas_values):
        try:
            ident,error=resolve_chemical(cas,root)
        except Exception as exc:
            ident,error=None,f'{type(exc).__name__}: structure lookup failed'
        if ident:ids.append(ident)
        else:mapping_errors[cas]=error
        if (i+1)%10==0: print(f'Structure resolution {i+1}/{len(cas_values)}',flush=True)
    for cas,error in mapping_errors.items():
        exclude(d.dsstox_casrn.eq(cas),'structure_mapping_'+error)
    if not ids:raise ValueError('No uniquely mapped structures')
    compounds=pd.DataFrame(ids)
    split=assign_splits(compounds)
    compounds=compounds.merge(split[['compound_id','cluster_id','split']],on='compound_id',validate='many_to_one')
    d=d.merge(compounds,left_on='dsstox_casrn',right_on='cas',how='left',validate='many_to_one')
    # Preserve each source series. One-point series are not silently reconstructed into curves.
    d['curve_id']='EPA:'+d.source_curve_id.astype(str)
    d['record_id']='EPA:'+d.observation_id.astype(str)
    d['source_id']='EPA-CvTdb-2.0.0'
    d['measurement_kind']='experimental'
    d['review_status']='automated_provisional'
    d['quality_note']='Original numeric values; unspecified log encoding and CAS mapping need source review.'
    d['eligible']=d.exclusion_reason.eq('')
    ready=d[d.eligible].copy()
    rawhash=digest(root/'raw/cvtdb'/FILE)
    codehash=hashlib.sha256(Path(__file__).read_bytes()+Path(__file__).with_name('chemistry.py').read_bytes()).hexdigest()
    split_payload=compounds[['cas','compound_id','cluster_id','split']].sort_values('cas').to_json(orient='records')
    dsid=hashlib.sha256((rawhash+codehash+split_payload).encode()).hexdigest()[:20]
    build=root/'processed'/dsid; build.mkdir(parents=True,exist_ok=True)
    ready.to_parquet(build/'observations.parquet',index=False)
    compounds.to_parquet(build/'compounds.parquet',index=False)
    d[['record_id','exclusion_reason','eligible']].to_parquet(build/'eligibility.parquet',index=False)
    compounds[['cas','compound_id','cluster_id','split']].to_csv(reports/'compound-splits.csv',index=False)
    summary={'dataset_id':dsid,'source_sha256':rawhash,'adapter_sha256':codehash,
        'raw_points':len(d),'eligible_points':len(ready),'eligible_compounds':int(ready.compound_id.nunique()),
        'eligible_source_series':int(ready.curve_id.nunique()),'structure_mapping_failures':mapping_errors,
        'by_species_matrix':ready.groupby(['species','tissue']).agg(points=('record_id','count'),compounds=('compound_id','nunique')).reset_index().to_dict('records'),
        'splits':ready.groupby('split').agg(points=('record_id','count'),compounds=('compound_id','nunique'),clusters=('cluster_id','nunique')).reset_index().to_dict('records'),
        'exclusion_counts':d.exclusion_reason.str.split(';').explode().value_counts().drop('',errors='ignore').to_dict(),
        'readiness':'Provisional development data; not a source-reviewed human organ benchmark',
        'test_policy':'test split sealed; supervised records from these compounds excluded across species and auxiliary sources'}
    write_json(build/'manifest.json',summary); write_json(root/'processed/latest.json',{'dataset_id':dsid})
    write_json(reports/'data-audit.json',summary)
    return summary
