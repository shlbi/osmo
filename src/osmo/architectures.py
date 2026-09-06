"""Independent IV-bolus research decoders; concentrations are ng/mL.

Amounts are mg/kg, volumes L/kg, rates 1/h. Compartments are latent, not organs.
The nonlinear decoder freezes rates at an exponential midpoint each step. Each
matrix is a conservative Markov generator including an eliminated-mass sink.
"""
import torch
from torch import nn


def generator(rates, compartments):
    """Column-oriented transfer generator: elimination, then outward/return pairs."""
    n = compartments + 1
    edges = [(0, compartments)]
    for j in range(1, compartments):
        edges.extend([(0, j), (j, 0)])
    basis = rates.new_zeros((len(edges), n, n))
    for k, (donor, receiver) in enumerate(edges):
        basis[k, donor, donor] = -1
        basis[k, receiver, donor] = 1
    return torch.einsum('bk,kij->bij', rates, basis)


def advance(a, g, dt):
    return (torch.matrix_exp(g * dt[:, None, None]) @ a.unsqueeze(-1)).squeeze(-1)


def transport_amounts(rates, volume, dose, time, km=None, steps=32):
    """Exact linear solution, or conservative exponential-midpoint integration.

    Central-origin elimination and outward transport saturate with central
    concentration. Return rates stay linear. km is in mg/L.
    """
    compartments = (rates.shape[-1] + 1) // 2
    a = torch.cat([dose[:, None], dose.new_zeros((len(dose), compartments))], -1)
    if km is None:
        return advance(a, generator(rates, compartments), time)
    mask = rates.new_tensor([1] + [v for _ in range(compartments - 1) for v in (1, 0)])

    def rate_matrix(state):
        saturation = 1 / (1 + state[:, 0] / volume / km)
        effective = rates * (1 - mask + mask * saturation[:, None])
        return generator(effective, compartments)

    previous = torch.zeros_like(time)
    for j in range(steps):
        current = torch.expm1(torch.log1p(time) * ((j + 1) / steps))
        dt = current - previous
        midpoint = advance(a, rate_matrix(a), dt / 2)
        a = advance(a, rate_matrix(midpoint), dt)
        previous = current
    return a


class CurveModel(nn.Module):
    def __init__(self, kind, width=1032, time_scale=1., dose_center=0., dose_scale=1.):
        super().__init__()
        self.kind = kind
        self.time_scale = float(time_scale)
        self.dose_center = float(dose_center)
        self.dose_scale = float(dose_scale)
        self.encoder = nn.Sequential(nn.Linear(width, 64), nn.SiLU(), nn.Linear(64, 32), nn.SiLU())
        self.steps = 32
        if kind == 'direct':
            self.head = nn.Sequential(nn.Linear(34, 32), nn.SiLU(), nn.Linear(32, 1))
        else:
            self.compartments = 3 if kind == 'flux' else int(kind[-1])
            nr = 2 * self.compartments - 1
            low = [1e-4] * nr + [.02] + ([.01] if kind == 'flux' else [])
            high = [20.] * nr + [100.] + ([1e4] if kind == 'flux' else [])
            # Distinct fast/slow peripheral initializations avoid a symmetric
            # subspace in which identical compartments cannot specialize.
            peripheral = [r for j in range(self.compartments - 1) for r in (.5 / 10**j, .15 / 10**j)]
            initial = [.2] + peripheral + [1.] + ([30.] if kind == 'flux' else [])
            self.register_buffer('log_low', torch.tensor(low).log())
            self.register_buffer('log_high', torch.tensor(high).log())
            self.head = nn.Linear(32, len(low))
            nn.init.zeros_(self.head.weight)
            with torch.no_grad():
                f = (torch.tensor(initial).log() - self.log_low) / (self.log_high - self.log_low)
                self.head.bias.copy_(torch.logit(f))

    def forward(self, x, time, dose):
        encoded = self.encoder(x)
        if self.kind == 'direct':
            temporal = torch.log1p(time) / self.time_scale
            dosing = (dose.log10() - self.dose_center) / self.dose_scale
            return self.head(torch.cat([encoded, temporal[:, None], dosing[:, None]], -1)).squeeze(-1) + dose.log10()
        p = torch.exp(self.log_low + (self.log_high - self.log_low) * torch.sigmoid(self.head(encoded))).double()
        nr = 2 * self.compartments - 1
        volume = p[:, nr]
        a = transport_amounts(p[:, :nr], volume, dose.double(), time.double(),
                              p[:, nr + 1] if self.kind == 'flux' else None, self.steps)
        return (1000 * a[:, 0] / volume).clamp_min(1e-20).log10().float()
