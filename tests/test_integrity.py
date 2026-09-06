import numpy as np
import pandas as pd
import pytest
from osmo.metrics import evaluate
from osmo.cvtdb import unit_conversion
from osmo.chemistry import identity,chemical_clusters


def test_fold_boundaries_and_failures_count():
    d=pd.DataFrame({'concentration':[10.]*5,'compound_id':['a']*5,'curve_id':['a']*5,
                    'tissue':['plasma']*5,'cluster_id':['a']*5})
    r=evaluate(d,[5,20,21,np.nan,-1],bootstrap=0)
    assert r['score']==.4 and r['failed_predictions']==2


def test_dense_curve_does_not_dominate_macro():
    d=pd.DataFrame({'concentration':[10.]*101,'compound_id':['a']*100+['b'],
        'curve_id':['a']*100+['b'],'tissue':['liver']*101,'cluster_id':['a']*100+['b']})
    r=evaluate(d,[10.]*100+[1000.],bootstrap=10)
    assert r['score']==.5 and r['pooled_point_score']>.99-1e-4


def test_invalid_truth_cannot_inflate_metric():
    d=pd.DataFrame({'concentration':[0.],'compound_id':['a'],'curve_id':['a'],
                    'tissue':['plasma'],'cluster_id':['a']})
    with pytest.raises(ValueError):evaluate(d,[0.])


def test_units_do_not_equate_tissue_and_fluid():
    assert unit_conversion('ug/g','liver')==(1000.,'ng/g')
    assert unit_conversion('ug/mL','plasma')==(1000.,'ng/mL')
    assert unit_conversion('ug/mL','liver')==(None,None)
    assert unit_conversion('%ID/g','kidney')==(None,None)
    assert unit_conversion('ug/g','plasma')==(None,None)


def test_stereoisomers_share_split_cluster():
    d=pd.DataFrame([identity('C[C@H](O)C(=O)O'),identity('C[C@@H](O)C(=O)O'),identity('c1ccccc1')])
    c=chemical_clusters(d).set_index('compound_id')
    assert c.loc[d.iloc[0].compound_id].cluster_id==c.loc[d.iloc[1].compound_id].cluster_id
