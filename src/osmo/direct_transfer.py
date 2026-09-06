"""Transfer-compatible direct plasma predictor; no physiological organ claim."""
import torch
from torch import nn
from .hybrid import Hybrid,FORMS

class DirectTransfer(Hybrid):
    def __init__(self,tasks=None):
        super().__init__(tasks=tasks)
        self.kinetics=nn.Sequential(nn.Linear(64+len(FORMS)+3,64),nn.SiLU(),nn.Linear(64,1))

    def forward(self,x,form,time,oral,infusion):
        context=torch.stack([torch.log1p(time),oral,torch.log1p(infusion)],-1)
        return self.kinetics(torch.cat([self.encoder(x),form,context],-1)).squeeze(-1)
