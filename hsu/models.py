import torch
import torch.distributions as D
import torch.nn.functional as F

import numpy as np

from . import utils

from .constraints import Simplex, Hypercube, Composite

constraints_LMM = Composite([Simplex(), Hypercube()])
constraints_NCM = Composite([Simplex(), Hypercube()])

def likelihood_LMM(Y, M, A, sigma2_noise):
    A = utils.extend_simplex(A)

    L = utils.log_gaussian(Y, A @ M, sigma2_noise)

    return L

def likelihood_NCM(Y, M, A, sigma2_endm):
    A = utils.extend_simplex(A)
    norms2 = torch.sum(A**2, dim=-1, keepdim=True)

    L = utils.log_gaussian(Y, A @ M, norms2 * sigma2_endm)

    return L
