"""Differentiable two-compartment baseline, not a validated organ PBPK model."""
import torch


def amounts(parameters, times, oral, infusion_h=None):
    """Unit dose mass in [gut, central, peripheral, eliminated].

    Parameters: ka (1/h), CL (L/h/kg), Vc/Vp (L/kg), Q (L/h/kg), F.
    Unabsorbed fraction is carried to the eliminated/unavailable accounting sink.
    Single oral, IV bolus or constant IV infusion. No concentration residual is added.
    """
    ka,cl,vc,vp,q,f=parameters.unbind(-1)
    z=torch.zeros_like(ka)
    a=torch.stack([
        torch.stack([-ka,z,z,z],-1),
        torch.stack([ka*f,-(cl+q)/vc,q/vp,z],-1),
        torch.stack([z,q/vc,-q/vp,z],-1),
        torch.stack([ka*(1-f),cl/vc,z,z],-1)],-2)
    initial=torch.stack([oral,1-oral,z,z],-1)
    bolus=(torch.matrix_exp(a*times[:,None,None])@initial[:,:,None]).squeeze(-1)
    if infusion_h is None:return bolus
    use=(infusion_h>0)&(oral==0)
    if not use.any():return bolus
    ai=a[use];ti=times[use];duration=infusion_h[use]
    # Augmented constant input state integrates the infusion exactly without A^-1.
    rate=torch.zeros_like(initial[use]);rate[:,1]=1/duration
    top=torch.cat([ai,rate[:,:,None]],-1)
    aug=torch.cat([top,torch.zeros_like(top[:,:1,:])],-2)
    ini=torch.zeros_like(aug[:,:,0]);ini[:,-1]=1
    active=torch.minimum(ti,duration)
    delivered=(torch.matrix_exp(aug*active[:,None,None])@ini[:,:,None]).squeeze(-1)[:,:4]
    after=(torch.matrix_exp(ai*(ti-active)[:,None,None])@delivered[:,:,None]).squeeze(-1)
    result=bolus.clone();result[use]=after;return result
