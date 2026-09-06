import numpy as np
import pandas as pd
import pytest
from osmo.expanded import auxiliary_rows,concentration_rows

def test_auxiliary_conflicts_and_ppbr_transform():
    d=pd.DataFrame([
        dict(dataset='previous_ADME',task='human_ppbr',value=90.,target=-1.,compound_id='a',aux_cluster='a',aux_split='train'),
        dict(dataset='PKSmart',task='human_fup',value=.1,target=np.nan,compound_id='b',aux_cluster='b',aux_split='aux_validation',valid_range=True,conflicting_values=False,exact_duplicate=False,source_role='development'),
        dict(dataset='PKSmart',task='human_fup',value=.2,target=np.nan,compound_id='c',aux_cluster='c',aux_split='train',valid_range=True,conflicting_values=True,exact_duplicate=False,source_role='development')])
    out=auxiliary_rows(d)
    assert out.compound_id.tolist()==['a','b']
    np.testing.assert_allclose(out.target,[-1,-1])
    d.loc[1,'aux_cluster']='a'
    with pytest.raises(ValueError,match='leakage'):auxiliary_rows(d)

def test_tissue_and_dose_basis_are_not_converted():
    d=pd.DataFrame([dict(integration_candidate=True,route='intraperitoneal',unit=u,dose_unit=b,value=10.,dose=5.,time_h=.5,species='mouse',matrix=m,compound_id='a',arm='sham')
        for u,b,m in [('ng/g','mg/kg','brain'),('ng/mL','mg','plasma')]])
    out=concentration_rows(d)
    assert out.output_head.nunique()==2
    assert out.dose_basis.tolist()==['mg/kg','mg']
    np.testing.assert_allclose(out.target_log10,[1,1])
    assert out.loss_weight.sum()==1
    d.loc[0,'unit']='unknown'
    with pytest.raises(ValueError,match='unit'):concentration_rows(d)
