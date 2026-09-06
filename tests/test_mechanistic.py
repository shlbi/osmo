import pytest
torch=pytest.importorskip('torch')
from osmo.mechanistic import amounts


def test_mass_balance_nonnegative_and_iv_initial_state():
    p=torch.tensor([[1.,.2,.3,.8,.5,.7]]*5,dtype=torch.float64)
    t=torch.tensor([0.,.1,1.,10.,100.],dtype=torch.float64)
    for oral in [torch.zeros(5,dtype=torch.float64),torch.ones(5,dtype=torch.float64)]:
        a=amounts(p,t,oral)
        assert torch.allclose(a.sum(-1),torch.ones(5,dtype=torch.float64),atol=1e-8)
        assert a.min()>=-1e-10
    assert amounts(p,t,torch.zeros(5,dtype=torch.float64))[0,1]==1.


def test_differentiable_clearance_reduces_late_central_amount():
    p=torch.tensor([[1.,.2,.3,.8,.5,.7]],dtype=torch.float64,requires_grad=True)
    a=amounts(p,torch.tensor([10.],dtype=torch.float64),torch.tensor([0.],dtype=torch.float64))
    a[0,1].backward()
    assert torch.isfinite(p.grad).all()
    assert p.grad[0,1]<0
