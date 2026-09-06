"""Molecular identity and deterministic scaffold/similarity grouping."""
import hashlib
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem.MolStandardize import rdMolStandardize

FP = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)


def identity(smiles):
    m = Chem.MolFromSmiles(smiles)
    if m is None:
        raise ValueError('Invalid SMILES')
    parent = rdMolStandardize.FragmentParent(m)
    canonical = Chem.MolToSmiles(parent, isomericSmiles=True)
    nonstereo = Chem.MolToSmiles(parent, isomericSmiles=False)
    scaffold = MurckoScaffold.MurckoScaffoldSmiles(mol=parent, includeChirality=False)
    return {'smiles':canonical,'compound_id':Chem.MolToInchiKey(parent),
            'family_id':hashlib.sha256(nonstereo.encode()).hexdigest()[:20],
            'scaffold':scaffold, 'molecular_weight':Descriptors.MolWt(parent)}


def features(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError('Invalid molecular input')
    fp = np.asarray(FP.GetFingerprintAsNumPy(mol), dtype=np.float32)
    desc = np.array([Descriptors.MolWt(mol),Descriptors.MolLogP(mol),
        Descriptors.TPSA(mol),Descriptors.NumHDonors(mol),Descriptors.NumHAcceptors(mol),
        Descriptors.NumRotatableBonds(mol),Descriptors.RingCount(mol),
        Descriptors.FractionCSP3(mol)],dtype=np.float32)
    return np.concatenate([fp, desc])


def chemical_clusters(compounds, threshold=.70):
    d=compounds.drop_duplicates('compound_id').sort_values('compound_id').reset_index(drop=True)
    fps=[FP.GetFingerprint(Chem.MolFromSmiles(s)) for s in d.smiles]
    parents=list(range(len(d)))
    def root(i):
        while parents[i]!=i:
            parents[i]=parents[parents[i]]; i=parents[i]
        return i
    def union(a,b):
        a,b=root(a),root(b)
        parents[max(a,b)]=min(a,b)
    for i in range(len(d)):
        sims=DataStructs.BulkTanimotoSimilarity(fps[i],fps[:i])
        for j,sim in enumerate(sims):
            same_scaffold=bool(d.iloc[i].scaffold) and d.iloc[i].scaffold==d.iloc[j].scaffold
            if sim >= threshold or same_scaffold or d.iloc[i].family_id==d.iloc[j].family_id:
                union(i,j)
    d['cluster_id']=[hashlib.sha256(d.iloc[root(i)].compound_id.encode()).hexdigest()[:16] for i in range(len(d))]
    return d


def assign_splits(compounds, seed=4107):
    d=chemical_clusters(compounds)
    clusters=sorted(d.cluster_id.unique())
    if len(clusters)<10:
        raise ValueError('Fewer than 10 chemical clusters: cannot make a useful train/dev/sealed-test split')
    rng=np.random.default_rng(seed)
    rng.shuffle(clusters)
    ntest=max(2,round(len(clusters)*.20))
    ndev=max(2,round(len(clusters)*.20))
    mapping={c:('test' if i<ntest else 'validation' if i<ntest+ndev else 'train') for i,c in enumerate(clusters)}
    d['split']=d.cluster_id.map(mapping)
    return d
