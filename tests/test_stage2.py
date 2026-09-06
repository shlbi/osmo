import numpy as np,pandas as pd,torch
from osmo.chemistry import identity
from osmo.auxiliary import guarded_splits
from osmo.mechanistic import amounts
from osmo.hybrid import Hybrid

def test_reserved_chemistry_blocks_auxiliary_and_preserves_training():
    frozen=pd.DataFrame([dict(**identity('CCO'),split='test'),dict(**identity('c1ccccc1'),split='train')])
    new=pd.DataFrame([identity('CCO'),identity('Cc1ccccc1')])
    result=guarded_splits(new,frozen).set_index('compound_id')
    assert result.loc[identity('CCO')['compound_id'],'aux_split']=='excluded_reserved_chemistry'
    assert result.loc[identity('Cc1ccccc1')['compound_id'],'aux_split']=='train'
    assert frozen.split.tolist()==['test','train']

def test_infusion_mass_tracks_delivered_dose_and_has_gradients():
    p=torch.tensor([[1.,.2,.3,.8,.5,.7]]*4,dtype=torch.float64,requires_grad=True)
    t=torch.tensor([0.,.5,1.,5.],dtype=torch.float64)
    a=amounts(p,t,torch.zeros_like(t),torch.ones_like(t))
    assert torch.allclose(a.sum(-1),torch.tensor([0.,.5,1.,1.],dtype=torch.float64),atol=1e-9)
    assert a.min()>=-1e-10
    a[:,1].sum().backward();assert torch.isfinite(p.grad).all()

def test_hybrid_continuous_time_dose_scaling_and_parameter_bounds():
    m=Hybrid();x=torch.zeros(3,1032);form=torch.zeros(3,5);form[:,0]=1
    t=torch.tensor([0.,1.,20.]);oral=torch.zeros(3);infusion=torch.zeros(3)
    p=m.parameters_from_structure(x,form)
    assert ((p>=m.log_low.exp())&(p<=m.log_high.exp())).all()
    logc=m(x,form,t,oral,infusion)
    assert torch.isfinite(logc).all() and logc[-1]<logc[0]
    logc.sum().backward();assert torch.isfinite(m.kinetics.weight.grad).all()

def test_infusion_matches_independent_numerical_ode():
    from scipy.integrate import solve_ivp
    params=[1.,.2,.3,.8,.5,.7];ka,cl,vc,vp,q,f=params
    def rhs(t,y):
        gut,central,peripheral,sink=y
        return [-ka*gut,ka*f*gut-(cl+q)/vc*central+q/vp*peripheral+1,
                q/vc*central-q/vp*peripheral,ka*(1-f)*gut+cl/vc*central]
    ref=solve_ivp(rhs,[0,.7],[0,0,0,0],rtol=1e-10,atol=1e-12).y[:,-1]
    actual=amounts(torch.tensor([params],dtype=torch.float64),torch.tensor([.7],dtype=torch.float64),
                   torch.zeros(1,dtype=torch.float64),torch.ones(1,dtype=torch.float64))[0].numpy()
    assert np.allclose(actual,ref,atol=1e-9)
