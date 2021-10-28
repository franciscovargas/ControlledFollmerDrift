import torch
import torch.nn.functional as F

import torchsde
import math
import matplotlib.pyplot as plt

import numpy as np

from torch import _vmap_internals

from cfollmer.sampler_utils import detach_state_dict


def log_g(Θ, ln_prior, ln_like, γ=1.0, debug=False):
    """
    g function in control objective
    
    g is the Radon-Nikodym derivtive between
    the joint and N(0, γ I)
    """
    if torch.any(torch.isnan(Θ)) or torch.any(torch.isinf(Θ)):
        import pdb; pdb.set_trace()
    normal_term = -0.5 * ((Θ / γ)**2).sum(axis=1) 

    ll =ln_like(Θ)
    lp =  ln_prior(Θ)
    
    if debug:
        print("Gl", ll.min().item(), ll.max().item())
        print("Gp", lp.min().item(), lp.max().item())
        print("Gp", normal_term.min().item(), normal_term.max().item())
    if torch.any(torch.isnan(normal_term)) or torch.any(torch.isinf(normal_term)):
        import pdb; pdb.set_trace()
    
    return ll + ( lp - normal_term)


def log_g_direct(Θ, ln_prior, ln_like, γ=1.0):
    """
    g function in control objective
    
    g is the Radon-Nikodym derivtive ln_prior and N(0, γ I)
    """
    normal_term = -0.5 * (Θ**2).sum(axis=1) / γ

    return ( ln_prior(Θ) - normal_term)



def simplified(
        sde, Θ_0, X, y, ln_prior,
        ln_like, Δt=0.05, γ=1.0,
        device="cpu", batchnorm=False, method="euler", adjoint=False, debug=False
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
        
    ΘT = Θs[-1] 
    sde.last_samples = ΘT
    lng = ln_like_partial(ΘT)
    
    return  - (lng).mean()  / X.shape[0]


def relative_entropy_control_cost(
        sde, Θ_0, X, y, ln_prior,
        ln_like, Δt=0.05, γ=1.0,
        device="cpu", batchnorm=False, method="euler", adjoint=False, debug=False
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
    sde.last_samples = ΘT
    lng = log_g(ΘT, ln_prior, ln_like_partial, γ_t[-1,0,0], debug=debug)
    
    γ_t = sde.g(ts, Θs)
    import pdb;pdb.set_trace()
    girsanov_factor = (0.5 * (( (μs / γ_t)**2).sum(axis=-1)) ).sum(axis=0) * Δt
    
    return (girsanov_factor  - lng).mean()  / X.shape[0]



def stl_relative_entropy_control_cost_xu(
        sde, Θ_0, X, y, ln_prior,
        ln_like, Δt=0.05, γ=1.0,
        device="cpu", batchnorm=False, method="euler", adjoint=False, debug=False
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
    sde.last_samples = ΘT
    
    γ_t = sde.g(ts, Θs)
    import pdb;pdb.set_trace()
    lng = log_g(ΘT, ln_prior, ln_like_partial, γ_t[-1,0,0], debug=debug)
    girsanov_factor_dt =  0.5 * (((μs / γ_t)**2).sum(axis=-1)).sum(axis=0) * Δt
    
#     import pdb; pdb.set_trace()
    
    dW = torch.normal(mean=0.0, std=math.sqrt(Δt), size=μs_detached.shape).to(device)
    girsanov_factor_dW =  (torch.einsum("ijk,ijk->ij", μs_detached / γ_t, dW)).sum(axis=0).mean()

    girsanov_factor = girsanov_factor_dt + girsanov_factor_dW
    
    return (girsanov_factor  - lng).mean()  / X.shape[0]


def stl_relative_entropy_control_cost_nik(
        sde, Θ_0, X, y, ln_prior,
        ln_like, Δt=0.05, γ=1.0,
        device="cpu", batchnorm=False, method="euler",
        adjoint=False, debug=False, dw=False
    ):
    """
    Stick the landing objective proposed by Nik Nuesken for the
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
#         μs = f(ts, Θs)
        μs_detached = f_detached(ts, Θs).to(device)
    
    γ_t = sde.g(ts, Θs)
    import pdb;pdb.set_trace()
    ΘT = Θs[-1] 
    sde.last_samples = ΘT
    lng = log_g(ΘT, ln_prior, ln_like_partial, γ_t[-1,0,0], debug=debug)
    girsanov_factor_dt = 0.5  * (( (μs_detached / γ_t)**2).sum(axis=-1)).sum(axis=0) * Δt
    
#     import pdb; pdb.set_trace()
    if dw:
        dW = torch.normal(mean=0.0, std=math.sqrt(Δt), size=μs_detached.shape).to(device)
        girsanov_factor_dW =  (torch.einsum("ijk,ijk->ij", μs_detached / γ_t, dW)).sum(axis=0).mean()
    else:
        girsanov_factor_dW = 0

    girsanov_factor = girsanov_factor_dt + girsanov_factor_dW
    
    return (girsanov_factor  - lng).mean()  / X.shape[0]

