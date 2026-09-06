"""Explicit adapters for expanded auxiliary and concentration observations."""
import numpy as np

def auxiliary_rows(data):
    d=data.copy()
    eligible=d.dataset.eq('previous_ADME') | (d.valid_range.fillna(False).astype(bool)
        & ~d.conflicting_values.fillna(False).astype(bool) & ~d.exact_duplicate.fillna(False).astype(bool)
        & d.source_role.eq('development'))
    d=d[eligible & d.aux_split.isin(['train','aux_validation'])].copy()
    # Prior PPBR targets are log10 fraction unbound, not log10 percent bound.
    new=d.dataset.eq('PKSmart')
    d.loc[new,'target']=np.log10(d.loc[new,'value'])
    if not np.isfinite(d.target).all():raise ValueError('Nonfinite auxiliary target')
    if d.duplicated(['compound_id','task']).any():raise ValueError('Repeated compound/task')
    if set(d.loc[d.aux_split.eq('train'),'aux_cluster']) & set(d.loc[d.aux_split.eq('aux_validation'),'aux_cluster']):
        raise ValueError('Chemical cluster leakage')
    return d.reset_index(drop=True)

def concentration_rows(data):
    d=data[data.integration_candidate].copy()
    if not d.route.isin(['oral','intraperitoneal','intranasal','iv']).all():raise ValueError('Unsupported route')
    if not d.unit.isin(['ng/mL','ng/g']).all():raise ValueError('Unsupported concentration unit')
    if not d.dose_unit.isin(['mg','mg/kg']).all():raise ValueError('Unsupported dose basis')
    if not ((d.value>0)&(d.dose>0)&(d.time_h>=0)).all():raise ValueError('Invalid observation')
    d['target_log10']=np.log10(d.value)
    d['dose_log10']=np.log10(d.dose)
    # Output heads have an explicit measurement basis, not an assumed tissue density.
    d['output_head']=d.species+'|'+d.matrix+'|'+d.unit
    d['dose_basis']=d.dose_unit
    d['curve_group']=d.compound_id+'|'+d.output_head+'|'+d.arm
    # Equal compounds, then tissue/arm groups, then observations within each group.
    d['loss_weight']=1/(d.groupby('compound_id').curve_group.transform('nunique')*
        d.groupby('curve_group').curve_group.transform('size'))
    return d.reset_index(drop=True)
