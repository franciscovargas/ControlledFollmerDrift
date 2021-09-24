import torch
import torch.nn.functional as F

import torchsde
import math
import matplotlib.pyplot as plt

import numpy as np

from torch import _vmap_internals

from cfollmer.sampler_utils import detach_state_dict

def log_g(Θ, ln_prior, ln_like, γ=1.0):
    """
    g function in control objective
    
    g is the Radon-Nikodym derivtive between
    the joint and N(0, γ I)
    """
    normal_term = -0.5 * (Θ**2).sum(axis=1) / γ

    return ln_like(Θ) + ( ln_prior(Θ) - normal_term)


def log_g_direct(Θ, ln_prior, ln_like, γ=1.0):
    """
    g function in control objective
    
    g is the Radon-Nikodym derivtive ln_prior and N(0, γ I)
    """
    normal_term = -0.5 * (Θ**2).sum(axis=1) / γ

    return ( ln_prior(Θ) - normal_term)



def relative_entropy_control_cost(
        sde, Θ_0, X, y, ln_prior,
        ln_like, Δt=0.05, γ=1.0,
        device="cpu", batchnorm=False, method="euler", adjoint=False
    ):
    """
    Objective for the Hamilton-Bellman-Jacobi Follmer Sampler
    """
    n = int(1.0 / Δt)
    ts = torch.linspace(0, 1, n).to(device)
    
    ln_like_partial = lambda Θ: ln_like(Θ, X, y)
    
    if not adjoint:
        Θs =  torchsde.sdeint(sde, Θ_0, ts, method=method,dt=Δt)
    else:
        Θs =  torchsde.sdeint_adjoint(sde, Θ_0, ts, method=method,dt=Δt)

    if not batchnorm:
        μs = sde.f(ts, Θs)
    else:
        def f_(t, x):
            return sde.f(t, x)
        f = _vmap_internals.vmap(f_) 
        μs = f(ts, Θs)
        
    ΘT = Θs[-1] 
    lng = log_g(ΘT, ln_prior, ln_like_partial, γ)
    girsanov_factor = (0.5 / γ) * ((μs**2).sum(axis=-1)).sum(axis=0) * Δt
    
    return (girsanov_factor  - lng).mean()


def stl_relative_entropy_control_cost(
        sde, Θ_0, X, y, ln_prior,
        ln_like, Δt=0.05, γ=1.0,
        device="cpu", batchnorm=False, method="euler", adjoint=False
    ):
    """
    Stick the landing objective (Xu et al. 2021) for the
    Hamilton-Bellman-Jacobi Follmer Sampler
    """
    n = int(1.0 / Δt)
    ts = torch.linspace(0, 1, n).to(device)
    
    ln_like_partial = lambda Θ: ln_like(Θ, X, y)
    
    if not adjoint:
        Θs =  torchsde.sdeint(sde, Θ_0, ts, method=method,dt=Δt)
    else:
        Θs =  torchsde.sdeint_adjoint(sde, Θ_0, ts, method=method,dt=Δt)
    
    if not batchnorm:
        μs = sde.f(ts, Θs)
        μs_detached = sde.f_detached(ts, Θs)
    else:
        def f_(t, x):
            return sde.f(t, x)
        
        f = _vmap_internals.vmap(f_) 
        f_detached = _vmap_internals.vmap(sde.f_detached) 
        μs = f(ts, Θs)
        μs_detached = f_detached(ts, Θs).to(device)
        
    ΘT = Θs[-1] 
    lng = log_g(ΘT, ln_prior, ln_like_partial, γ)
    girsanov_factor_dt = (0.5 / γ) * ((μs**2).sum(axis=-1)).sum(axis=0) * Δt
    
#     import pdb; pdb.set_trace()
    dW = torch.normal(mean=0.0, std=math.sqrt(Δt), size=μs_detached.shape).to(device)
    girsanov_factor_dW = (1.0 / γ) * (torch.einsum("ijk,ijk->ij", μs_detached, dW)).sum(axis=0).mean()

    girsanov_factor = girsanov_factor_dt + girsanov_factor_dW
    
    return (girsanov_factor  - lng).mean()


def relative_entropy_control_cost_direct(sde, Θ_0, ln_prior, Δt=0.05, γ=1.0, device="cpu"):
    """
    Objective for the Hamilton-Bellman-Jacobi Follmer Sampler
    """
    n = int(1.0 / Δt)
    ts = torch.linspace(0, 1, n).to(device)
        
    Θs =  torchsde.sdeint(sde, Θ_0, ts, method="euler",dt=Δt)
    μs = sde.f(ts, Θs)
    ΘT = Θs[-1] 
    lng = log_g_direct(ΘT, ln_prior, γ)
    girsanov_factor = (0.5 / γ) * ((μs**2).sum(axis=-1)).sum(axis=0) * Δt
    
    return (girsanov_factor - lng).mean()