from typing import Callable, Optional

import torch
import torch.nn.functional as F

import torchsde
import math
import matplotlib.pyplot as plt

import numpy as np

from functorch import vmap


def relative_entropy_control_cost(
        sde: torch.nn.Module,
        log_p: Callable,
        param_batch_size: Optional[int] = 32,
        dt: Optional[float] = 0.05,
        device: Optional[torch.device] = None,
    ):
    """
    Objective for the Hamilton-Bellman-Jacobi Follmer Sampler
    """

    param_trajectory, ts = sde.sample_trajectory(param_batch_size, dt=dt, device=device)
    param_T = param_trajectory[-1]

    # Has shape [T, batch_size, dim]
    # Note: this is essentially evaluated twice (once before in sdeint call).
    # Could adjust torchsde to return the drift as an extra parameter (seems that there's
    # no way currently)
    us = vmap(sde.f)(ts, param_trajectory)

    # Not sure if this terminology is correct so feel free to correct
    energy_cost = torch.sum(us**2, dim=[0, 2]) * dt  / (2 * sde.gamma)
    terminal_cost = - torch.sum(param_T**2, dim=1) / (2 * sde.gamma) - log_p(param_T)

    return torch.mean(energy_cost + terminal_cost)


def stl_control_cost(
        sde: torch.nn.Module,
        log_p: Callable,
        param_batch_size: Optional[int] = 32,
        dt: Optional[float] = 0.05,
        device: Optional[torch.device] = None,
    ):
    """
    TODO : seems to give reasonable results, incredibly slow
    
    Objective for the STL (Xu et al. 2021) Hamilton-Bellman-Jacobi Follmer Sampler
    """

    param_trajectory, ts = sde.sample_trajectory(param_batch_size, dt=dt, device=device)
    param_T = param_trajectory[-1]

    # Has shape [T, batch_size, dim]
    # Note: this is essentially evaluated twice (once before in sdeint call).
    # Could adjust torchsde to return the drift as an extra parameter (seems that there's
    # no way currently)
    us = vmap(sde.f)(ts, param_trajectory)
    f_detached = lambda ts, xs: sde.f(ts, xs, detach=True)
    us_detached = vmap(f_detached)(ts, param_trajectory)
    
    # dW samples for Ito integral
    dW = torch.normal(mean=0.0, std=math.sqrt(dt), size=us_detached.shape).to(device)

    # Costs
    energy_cost = torch.sum(us**2, dim=[0, 2]) * dt  / (2 * sde.gamma)
    ito_cost = (torch.einsum("ijk,ijk->ij", us_detached / sde.gamma, dW)).sum(axis=0)#.mean() # batched dot products
    terminal_cost = - torch.sum(param_T**2, dim=1) / (2 * sde.gamma) - log_p(param_T)

    return torch.mean(energy_cost + ito_cost + terminal_cost)


def vargrad_control_cost(
        sde: torch.nn.Module,
        log_p: Callable,
        param_batch_size: Optional[int] = 32,
        dt: Optional[float] = 0.05,
        device: Optional[torch.device] = None,
    ):
    """
    TODO : Test and double check for errors
    
    Note this is implemented in the most naive way following Eq  in Nüsken and Ritchter
    a more efficient implementation would augment the state-space as we disccused
    just currently racing through this. 
    
    Objective for the VarGrad (Nüsken et al. 2020) Hamilton-Bellman-Jacobi Follmer Sampler
    """

    param_trajectory, ts = sde.sample_trajectory(param_batch_size, dt=dt, device=device)
    
    param_trajectory = param_trajectory.detach()
    param_T = param_trajectory[-1]

    # Has shape [T, batch_size, dim]
    # Note: this is essentially evaluated twice (once before in sdeint call).
    # Could adjust torchsde to return the drift as an extra parameter (seems that there's
    # no way currently)
    us = vmap(sde.f)(ts, param_trajectory)
    f_detached = lambda ts, xs: sde.f(ts, xs, detach=True)
    us_detached = vmap(f_detached)(ts, param_trajectory)
    
    # dW samples for Ito integral
    dW = torch.normal(mean=0.0, std=math.sqrt(dt), size=us_detached.shape).to(device)

    # Costs
    energy_cost = torch.sum(us**2, dim=[0, 2]) * dt  / (2 * sde.gamma)
    ito_cost = (torch.einsum("ijk,ijk->ij", us / sde.gamma, dW)).sum(axis=0) # .mean() # batched dot products
    cross_term =  (torch.einsum("ijk,ijk->ij", us / sde.gamma, us_detached)).sum(axis=0) # .mean()
    terminal_cost = - torch.sum(param_T**2, dim=1) / (2 * sde.gamma) - log_p(param_T)
    
    Y_T = energy_cost - ito_cost - cross_term

    return torch.var(Y_T - terminal_cost, unbiased=True)
