"""Source ADME normalization and frozen-chemistry exclusion for auxiliary training."""
import hashlib,json
from pathlib import Path
import numpy as np,pandas as pd
from rdkit import Chem,DataStructs
from .chemistry import identity,FP
from .io import write_json,digest

ASSAYS={'CHEMBL3301365':('human_ppbr','%','PPB','Homo sapiens'),
 'CHEMBL3301366':('rat_ppbr','%','PPB','Rattus norvegicus'),
 'CHEMBL3301370':('human_microsome_clint','mL.min-1.g-1','CL','Homo sapiens'),
 'CHEMBL3301371':('rat_hepatocyte_clint','uL.min-1.(10^6cells)-1','CL','Rattus norvegicus'),
 'CHEMBL3301372':('human_hepatocyte_clint','uL.min-1.(10^6cells)-1','CL','Homo sapiens')}

def guarded_splits(new, frozen, threshold=.7):
    """Connected components over all supplied chemistry; existing splits never change.

    Aux components touching any frozen validation/test compound are excluded from
    training, including transitive bridges. Labels play no part in assignment.
    """
    d=pd.concat([frozen,new],ignore_index=True).drop_duplicates('compound_id').sort_values('compound_id').reset_index(drop=True)
    parent=list(range(len(d)));fps=[FP.GetFingerprint(Chem.MolFromSmiles(s)) for s in d.smiles]
    def root(i):
        while parent[i]!=i:parent[i]=parent[parent[i]];i=parent[i]
        return i
    def union(a,b):
        a,b=root(a),root(b);parent[max(a,b)]=min(a,b)
    seen={}
    for i,row in enumerate(d.itertuples()):
        for key in [('scaffold',row.scaffold),('family',row.family_id)]:
            if not key[1]:continue
            if key in seen:union(i,seen[key])
            else:seen[key]=i
        for j in np.flatnonzero(np.array(DataStructs.BulkTanimotoSimilarity(fps[i],fps[:i]))>=threshold):union(i,int(j))
    frozen_map=dict(zip(frozen.compound_id,frozen.split));groups={}
    for i,cid in enumerate(d.compound_id):
        groups.setdefault(root(i),set()).add(frozen_map.get(cid,'new'))
    records=[]
    for i,cid in enumerate(d.compound_id):
        group=root(i);members=groups[group];anchor=d.iloc[group].compound_id
        if 'test' in members or 'validation' in members:split='excluded_reserved_chemistry'
        elif 'train' in members:split='train'
        else:split='aux_validation' if int(hashlib.sha256(anchor.encode()).hexdigest()[:8],16)%5==0 else 'train'
        records.append({'compound_id':cid,'aux_split':split,'aux_cluster':hashlib.sha256(anchor.encode()).hexdigest()[:16]})
    return pd.DataFrame(records)

def prepare(root,out):
    root,out=Path(root),Path(out);out.mkdir(parents=True,exist_ok=True)
    latest=json.loads((root/'processed/latest.json').read_text())['dataset_id']
    frozen=pd.read_parquet(root/'processed'/latest/'compounds.parquet')
    raw=root/'raw/chembl-az';rows=[];ledger=[];cache={}
    for aid,(task,unit,kind,species) in ASSAYS.items():
        assay=json.loads((raw/f'{aid}-assay.json').read_text())
        if assay['assay_organism']!=species:raise ValueError('Unexpected assay species')
        for path in sorted(raw.glob(aid+'-activities-*.json')):
            for a in json.loads(path.read_text())['activities']:
                reason='eligible';sm=a.get('canonical_smiles');v=pd.to_numeric(a.get('standard_value'),errors='coerce')
                if a.get('standard_units')!=unit or a.get('standard_type')!=kind:reason='unit_or_endpoint'
                elif a.get('data_validity_comment') or a.get('potential_duplicate'):reason='source_flag'
                elif a.get('standard_relation') not in [None,'=']:reason='censored'
                elif not np.isfinite(v):reason='missing_value'
                elif kind=='CL' and not 3<v<150:reason='assay_boundary'
                elif kind=='PPB' and not 10<v<99.95:reason='assay_boundary'
                elif not sm or '.' in sm:reason='missing_or_multicomponent_structure'
                if reason=='eligible':
                    try:
                        if sm not in cache:cache[sm]=identity(sm)
                        ident=cache[sm]
                        # PPBR target = log10 unbound fraction; CL stays an in-vitro endpoint.
                        target=np.log10(1-v/100) if kind=='PPB' else np.log10(v)
                        rows.append(dict(**ident,task=task,source_id=str(a['activity_id']),assay_id=aid,
                            species=species,value=float(v),unit=unit,target=float(target),
                            relation=a.get('standard_relation') or 'unspecified_interior',license='CC-BY-SA-3.0'))
                    except ValueError:reason='invalid_structure'
                ledger.append({'activity_id':a['activity_id'],'task':task,'status':reason})
    d=pd.DataFrame(rows);m=guarded_splits(d,frozen);d=d.merge(m,on='compound_id',validate='many_to_one')
    # One row per molecule/assay, retain source IDs and aggregate replicate targets with median.
    d=d.groupby(['compound_id','task'],as_index=False).agg({c:('median' if c in ['value','target'] else (lambda x:';'.join(sorted(set(x)))) if c=='source_id' else 'first') for c in d.columns if c not in ['compound_id','task']})
    d.to_parquet(out/'adme.parquet',index=False);pd.DataFrame(ledger).to_csv(out/'adme-eligibility.csv',index=False)
    m.to_csv(out/'auxiliary-chemical-registry.csv',index=False)
    report={'source':'AstraZeneca CHEMBL3301361 via ChEMBL','license':'CC-BY-SA-3.0',
        'license_evidence':'https://www.ebi.ac.uk/chembl/','frozen_pk_dataset':latest,
        'raw_activities':len(ledger),'eligible_rows':len(d),'eligible_compounds':int(d.compound_id.nunique()),
        'counts':d.groupby(['task','aux_split']).size().rename('rows').reset_index().to_dict('records'),
        'exclusions':pd.DataFrame(ledger).status.value_counts().to_dict(),
        'assumption':'Unspecified relation accepted only strictly inside stated assay range; boundary and censored records excluded.',
        'normalization':'Microsome mL/min/g numerically equals uL/min/mg; hepatocyte values are per million cells. Never treated as in-vivo whole-body clearance.',
        'adme_sha256':digest(out/'adme.parquet'),'sealed_labels_used':False}
    write_json(out/'adme-audit.json',report);return report
