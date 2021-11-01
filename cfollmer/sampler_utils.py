import torch
from collections import OrderedDict
from cfollmer.layers import ResBlock, get_timestep_embedding
from cfollmer.drifts import *



def detach_state_dict(state_dict):
    d = OrderedDict()
    
    for k,v in state_dict.items():
        d[k] = v.detach()
    return v



class FollmerSDE(torch.nn.Module):

    def __init__(self, state_size=1, brownian_size=1, 
                 batch_size=10, γ=1.0, drift=SimpleForwardNet,
                device="cpu", diffusion_type="uniform", γ_max=0.5, γ_min=0.04):
        super().__init__()
        self.noise_type = 'diagonal'
        self.sde_type = 'ito'
        
        drift = SimpleForwardNetBN if drift is None else drift
        
        self.diffusion_type = diffusion_type
        
        self.state_size = state_size
        self.brownian_size = brownian_size
        self.batch_size = batch_size
#         
        self.device = device

        self.γ = γ
        self.γ_max = γ_max
        self.γ_min = γ_min
        self.μ = drift(input_dim=state_size).to(device)
        
        self.μ_detached = drift(input_dim=state_size).to(device)

    # Drift
    def f(self, t, y):
        
        d = y.shape[0] if len(y.shape) == 2 else y.shape[1]
        t_ = t.to(self.device) * torch.ones(d,1).to(self.device)
        t_ = t_ if len(y.shape) == 2 else t_.T[...,None]
        
        return self.μ(y, t_)

    def f_detached(self, t, y):
        # For STL estimator
        
        d = y.shape[0] if len(y.shape) == 2 else y.shape[1]
        t_ = t.to(self.device) * torch.ones(d,1).to(self.device)
        t_ = t_ if len(y.shape) == 2 else t_.T[...,None]
        
#         if "ResNetScoreNetwork" in str(self.μ):
#             return self.μ_detached(y, t_)  
        
#         y = torch.cat((y, t_), dim=-1)

        return self.μ_detached(y, t_)  

    # Diffusion
    def g(self, t, y):
       
        if self.diffusion_type == "uniform":
            γ_t = self.γ
        elif self.diffusion_type == "linear":
            Δγ = (self.γ_max- self.γ_min)
            pos_slope = 2 * t * Δγ * (t < 0.5 )
            neg_slope = 2 * (1-t) * Δγ * (t >= 0.5 )
            γ_t = (self.γ_min + pos_slope + neg_slope)
        else:
            raise BaseException(f"{diffusion_type} schedule not implemented")
            
        diffusion = (torch.ones_like(y).to(self.device) * γ_t) 
        return torch.sqrt(diffusion)