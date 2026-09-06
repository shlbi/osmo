import numpy as np
import torch
from scipy.integrate import solve_ivp
from osmo.architectures import CurveModel, generator, transport_amounts


def test_linear_one_compartment_matches_analytic_and_mass_conservation():
    t = torch.tensor([0., .1, 5., 1200.], dtype=torch.double)
    dose = torch.tensor([1., 2., 3., 4.], dtype=torch.double)
    r = torch.full((4, 1), .2, dtype=torch.double)
    a = transport_amounts(r, torch.ones_like(t), dose, t)
    assert torch.allclose(a[:, 0], dose * torch.exp(-.2 * t), atol=1e-12)
    assert torch.allclose(a.sum(-1), dose, atol=1e-9)
    assert a.min() >= 0


def test_nonlinear_solver_against_independent_scipy_and_gradients():
    rates = torch.tensor([[.7, 1.2, .4, .5, .2]], dtype=torch.double, requires_grad=True)
    volume = torch.tensor([.5], dtype=torch.double)
    dose = torch.tensor([20.], dtype=torch.double)
    time = torch.tensor([12.], dtype=torch.double)
    km = torch.tensor([4.], dtype=torch.double, requires_grad=True)
    def rhs(t, a):
        k = np.array([.7, 1.2, .4, .5, .2]) * np.array([1/(1+a[0]/.5/4), 1/(1+a[0]/.5/4), 1, 1/(1+a[0]/.5/4), 1])
        return np.array([-a[0]*(k[0]+k[1]+k[3])+a[1]*k[2]+a[2]*k[4],
                         a[0]*k[1]-a[1]*k[2], a[0]*k[3]-a[2]*k[4], a[0]*k[0]])
    ref = solve_ivp(rhs, [0, 12], [20, 0, 0, 0], rtol=1e-10, atol=1e-12).y[:, -1]
    with torch.no_grad():
        coarse = transport_amounts(rates, volume, dose, time, km, steps=128).numpy()[0]
        refined = transport_amounts(rates, volume, dose, time, km, steps=256).numpy()[0]
    assert np.max(np.abs(coarse-ref)) > 3 * np.max(np.abs(refined-ref))
    a = transport_amounts(rates, volume, dose, time, km, steps=1024)
    np.testing.assert_allclose(a.detach().numpy()[0], ref, rtol=1e-4, atol=1e-6)
    assert torch.allclose(a.sum(-1), dose, atol=1e-9)
    assert a.min() >= 0
    a[0, 0].backward()
    assert torch.isfinite(rates.grad).all() and torch.isfinite(km.grad).all()
    assert km.grad.item() < 0


def test_three_compartment_linear_limit_and_dose_scaling():
    r = torch.tensor([[.2, .5, .15, .3, .1]], dtype=torch.double)
    v = torch.tensor([1.], dtype=torch.double)
    d = torch.tensor([2.], dtype=torch.double)
    t = torch.tensor([10.], dtype=torch.double)
    a = transport_amounts(r, v, d, t)
    b = transport_amounts(r, v, d * 2, t)
    limit = transport_amounts(r, v, d, t, torch.tensor([1e12], dtype=torch.double), 16)
    assert torch.allclose(b, 2 * a, atol=1e-9)
    assert torch.allclose(limit, a, atol=1e-9)
    assert torch.allclose(generator(r, 3).sum(1), torch.zeros((1, 4), dtype=torch.double), atol=1e-12)


def test_initial_concentration_and_model_backward():
    for kind in ['cmt1', 'cmt2', 'cmt3', 'flux', 'direct']:
        model = CurveModel(kind, width=8)
        x = torch.zeros((2, 8))
        y = model(x, torch.tensor([0., 2.]), torch.tensor([1., 2.]))
        assert torch.isfinite(y).all()
        if kind != 'direct':
            assert abs(y[0].item() - 3.) < 1e-5
        y.sum().backward()
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_three_compartment_initialization_has_distinct_peripheral_phases():
    model = CurveModel('cmt3', width=8)
    with torch.no_grad():
        p = torch.exp(model.log_low + (model.log_high-model.log_low)*torch.sigmoid(model.head(model.encoder(torch.zeros(1, 8))))).double()
    g = generator(p[:, :5], 3)[0, :3, :3]
    decay = torch.linalg.eigvals(g).real.sort().values
    assert (torch.diff(decay) > 1e-3).all()
    # Both peripheral directions must start differently; otherwise exact
    # symmetry can trap optimization in an effectively smaller model.
    assert p[0, 1] / p[0, 3] > 5
    assert p[0, 2] / p[0, 4] > 5
