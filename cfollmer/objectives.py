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
