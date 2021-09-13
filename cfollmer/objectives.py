import torch
import torch.nn.functional as F

import torchsde
import math
import matplotlib.pyplot as plt

import numpy as np



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



def relative_entropy_control_cost(sde, Θ_0, X, y, ln_prior, ln_like, Δt=0.05, γ=1.0, device="cpu"):
    """
    Objective for the Hamilton-Bellman-Jacobi Follmer Sampler
    """
    n = int(1.0 / Δt)
    ts = torch.linspace(0, 1, n).to(device)
    
    ln_like_partial = lambda Θ: ln_like(Θ, X, y)
    
    Θs =  torchsde.sdeint(sde, Θ_0, ts, method="euler",dt=Δt)
    μs = sde.f(ts, Θs)
    ΘT = Θs[-1] 
    lng = log_g(ΘT, ln_prior, ln_like_partial, γ)
    girsanov_factor = (0.5 / γ) * ((μs**2).sum(axis=-1)).sum(axis=0)
    
    return (girsanov_factor - lng).mean()


def relative_entropy_control_cost_direct(sde, Θ_0, ln_prior, Δt=0.05, γ=1.0, device="cpu"):
    """
    Objective for the Hamilton-Bellman-Jacobi Follmer Sampler
    """
    n = int(1.0 / Δt)
    ts = torch.linspace(0, 1, n).to(device)
    
    ln_like_partial = lambda Θ: ln_like(Θ, X, y)
    
    Θs =  torchsde.sdeint(sde, Θ_0, ts, method="euler",dt=Δt)
    μs = sde.f(ts, Θs)
    ΘT = Θs[-1] 
    lng = log_g_direct(ΘT, ln_prior, γ)
    girsanov_factor = (0.5 / γ) * ((μs**2).sum(axis=-1)).sum(axis=0)
    
    return (girsanov_factor - lng).mean()