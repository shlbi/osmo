"""Structure encoder -> bounded kinetic parameters -> mass-conserving amounts.

This is a two-compartment plasma research model, not a whole-body organ PBPK model.
"""
import numpy as np
import torch
from torch import nn
from .mechanistic import amounts

TASKS=['human_hepatocyte_clint','human_microsome_clint','human_ppbr','rat_hepatocyte_clint','rat_ppbr']
FORMS=['unknown','solution','suspension','tablet','capsule']

class Hybrid(nn.Module):
    def __init__(self,width=1032,tasks=None):
        super().__init__()
        self.encoder=nn.Sequential(nn.Linear(width,128),nn.SiLU(),nn.Linear(128,64),nn.SiLU())
        self.tasks=list(TASKS if tasks is None else tasks)
        self.aux=nn.Linear(64,len(self.tasks))
        self.kinetics=nn.Linear(64+len(FORMS),6)
        # Broad development constraints, not measured compound inputs.
        low=np.array([.01,.001,.02,.02,.001,.001]);high=np.array([30,10,10,100,10,.999])
        self.register_buffer('log_low',torch.tensor(np.log(low),dtype=torch.float32))
        self.register_buffer('log_high',torch.tensor(np.log(high),dtype=torch.float32))
        initial=np.array([1,.2,.3,1,.5,.6]);frac=(np.log(initial)-np.log(low))/(np.log(high)-np.log(low))
        nn.init.zeros_(self.kinetics.weight)
        with torch.no_grad():self.kinetics.bias.copy_(torch.tensor(np.log(frac/(1-frac)),dtype=torch.float32))

    def parameters_from_structure(self,x,form):
        logits=self.kinetics(torch.cat([self.encoder(x),form],-1))
        return torch.exp(self.log_low+(self.log_high-self.log_low)*torch.sigmoid(logits))

    def forward(self,x,form,time,oral,infusion):
        p=self.parameters_from_structure(x,form)
        # mg/kg divided by L/kg = mg/L; convert to ng/mL with factor 1000.
        a=amounts(p.double(),time.double(),oral.double(),infusion.double())
        c=(a[:,1]/p[:,2].double()*1000).clamp_min(1e-12)
        return c.log10().float()

    def auxiliary(self,x):return self.aux(self.encoder(x))
