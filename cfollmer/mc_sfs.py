import torch
import torch.nn.functional as F
import torchsde

import math
import numpy as np

from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import functorch



class MCFollmerDrift:
    
    def __init__(self, log_posterior, X,y, dim, device, n_samp=300, gamma=torch.tensor(1), debug=False):
        self.log_posterior = log_posterior
        self.debug = debug
        self.log_posterior = log_posterior
        self.device = device
        self.X = X
        self.dim = dim
        self.y = y
        self.gamma = gamma
        self.n_samp = n_samp
        self.distrib = torch.distributions.multivariate_normal.MultivariateNormal(
            loc=torch.zeros(dim),
            covariance_matrix=torch.eye(dim) * torch.sqrt(gamma)
        )
        
    def g(self, thet):
        func = lambda params: self.log_posterior(self.X, self.y, params)
        func = functorch.vmap(func)
        lp = func(thet)
        reg = 0.5 * (thet**2).sum(dim=-1) / self.gamma
        

        out = torch.exp(lp + reg)
        isnan = torch.isinf(torch.abs(out)) | torch.isnan(out)
        if self.debug and torch.any(isnan):
            import pdb; pdb.set_trace()
        return out # nans exp(reg)

    def ln_g(self, thet):
        func = lambda params: self.log_posterior(self.X, self.y, params)
        func = functorch.vmap(func)
        lp = func(thet)
        reg = 0.5 * (thet**2).sum(dim=-1) / self.gamma
        
        out = lp + reg
        isnan = torch.isinf(torch.abs(out)) | torch.isnan(out)
        if self.debug and torch.any(isnan):
            import pdb; pdb.set_trace()
            
        return out # nans exp(reg)
        
    def mc_follmer_drift_(self, t, params, Z):
        # Using Stein Estimator for SFS drift

        g_YZt = self.g(params[None, ...] + torch.sqrt(1-t) * Z)
        num = (Z * g_YZt[..., None]).mean(dim=0)
        denom = torch.sqrt(1-t) * (g_YZt).mean(dim=0)
        
        out = num / denom[...,None]
        
        isnan = torch.isinf(torch.abs(out)) | torch.isnan(out)
        
        return out
    
    def mc_follmer_drift_stable(self, t, params, Z):
        # Using Stein Estimator for SFS drift
        N, d = Z.shape
        lnN = torch.log(torch.tensor(N)).to(self.device)
        
        ln_g_YZt = self.ln_g(params[None, ...] + torch.sqrt(1-t) * Z)
        
        Z_plus = torch.nn.functional.relu(Z)
        Z_minus = torch.nn.functional.relu(-Z)        
        
        ln_num_plus = torch.logsumexp(
            (torch.log(Z_plus) + ln_g_YZt[..., None]) - lnN,
            dim=0,
        )
        ln_num_minus = torch.logsumexp(
            (torch.log(Z_minus) + ln_g_YZt[..., None]) - lnN,
            dim=0
        )
        
        ln_denom = torch.logsumexp(
            torch.log(torch.sqrt(1-t))  + (ln_g_YZt) - lnN,
            dim=0
        )
        
        out =  torch.exp(ln_num_plus-ln_denom) - torch.exp(ln_num_minus-ln_denom)
        
        
        isnan = torch.isinf(torch.abs(out)) | torch.isnan(out)
        
        return out 
    
    def mc_follmer_drift_debug(self, t, params):
        # Using Stein Estimator for SFS drift
        
        Z = self.distrib.rsample((self.n_samp,)).to(self.device)
        params = params[0]

        g_YZt = self.g(params[None, ...] + torch.sqrt(1-t) * Z)
        num = (Z * g_YZt[..., None]).mean(dim=0)
        denom = torch.sqrt(1-t) * (g_YZt).mean(dim=0)
        
        out = num / denom[...,None]
        
        isnan = torch.isinf(torch.abs(out)) | torch.isnan(out)
        
        if self.debug and torch.any(isnan):
            import pdb; pdb.set_trace()
        
        return out.reshape(1,-1)
    
    def mc_follmer_drift(self, t , params_batch):
        Z = self.distrib.rsample((params_batch.shape[0], self.n_samp)).to(self.device)
        
        func = lambda params, z: self.mc_follmer_drift_stable(t, params, z)
        func = functorch.vmap(func, in_dims=(0,0) )
        out = func(params_batch, Z)
#         import pdb; pdb.set_trace()
        return out

    

class MCFollmerSDE(torch.nn.Module):

    def __init__(self, gamma, dim, log_posterior, X_train, y_train, device, debug=False):
        super().__init__()

        self.noise_type = 'diagonal'
        self.sde_type = 'ito'
        self.gamma = gamma
        if debug:
            self.drift =  MCFollmerDrift(log_posterior, X_train, y_train, dim, device, gamma=gamma, debug=debug).mc_follmer_drift_debug
        else:
            self.drift =  MCFollmerDrift(log_posterior, X_train, y_train, dim, device, gamma=gamma).mc_follmer_drift
        self.dim = dim
        
    def f(self, t, y, detach=False):
        return self.drift(t, y)
        
    def g(self, t, y):
        return torch.sqrt(self.gamma )* torch.ones_like(y)

    def sample_trajectory(self, batch_size, dt=0.05, device=None):
        param_init = torch.zeros((batch_size, self.dim), device=device)

        n_steps = int(1.0 / dt)

        ts = torch.linspace(0, 1, n_steps, device=device)

        param_trajectory = torchsde.sdeint(self, param_init, ts, method="euler", dt=dt)

        return param_trajectory, ts

    def sample(self, batch_size, dt=0.05, device=None):
        return self.sample_trajectory(batch_size, dt=dt, device=device)[0] [-1]#[-1]
    

# mcfol = MCFollmerDrift(log_posterior, X_train, y_train, dim, device)
# sde_sfs = MCFollmerSDE(torch.tensor(gamma), dim, log_posterior, X_train, y_train, device)