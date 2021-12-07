from typing import Callable

import torch
import torch.nn.functional as F

import scipy.io as spio
import pandas as pd
import numpy as np
import glob
import os

def log_gaussian(x, mu, sigma2):
    diff = x - mu

    return torch.sum(- 0.5 * (np.log(2 * np.pi) + torch.log(sigma2) + diff**2 / sigma2))

def extend(x, shape):
    assert(x.shape == shape[:len(x.shape)])

    view = [1] * len(shape)
    for i in range(len(x.shape)):
        view[i] = x.shape[i]

    return x.view(view)

def match(x, shape):
    assert(x.shape == shape[:len(x.shape)])

    repetitions = list(shape)
    repetitions[:len(x.shape)] = [1] * len(x.shape)
        
    return extend(x, shape).repeat(repetitions)

def reduce_sum(x: torch.Tensor) -> torch.Tensor:
    dims = list(range(1, len(x.shape)))
    return torch.sum(x, dim=dims) if dims != [] else x

def repeat(x: torch.Tensor, n: int) -> torch.Tensor:
    sizes = [1 for _ in range(len(x.shape) + 1)]
    sizes[0] = n
    return x.repeat(sizes)

def extend_simplex(x: torch.Tensor) -> torch.Tensor:
    return torch.cat([x, 1 - x.sum(dim=-1, keepdim=True)], dim=-1)

def extend_zeros(x: torch.Tensor) -> torch.Tensor:
    return F.pad(x, (0, 1))

def reduce(x: torch.Tensor) -> torch.Tensor:
    return x[:, :-1]

def make_grad_func(U: Callable):

    @torch.enable_grad()
    def grad_U(*xs, create_graph=False, mask=None):
        if mask is None:
            mask = [True] * len(xs)

        xs = [x.clone().requires_grad_(m) for x, m in zip(xs, mask)]
        xs_diff = [x for x, m in zip(xs, mask) if m == True]
        grads = torch.autograd.grad(U(*xs).sum(), xs_diff, create_graph=create_graph)
        grads = [torch.nan_to_num(grad) for grad in grads]
        return grads

    return grad_U

def read_usgs(material, spectra_fn, wavelengths_fn):
    spectra_wavelengths = pd.read_csv(wavelengths_fn, names=[material], skiprows=1, squeeze=True)
    spectra = pd.read_csv(spectra_fn, names=[material], skiprows=1, squeeze=True)

    mask = spectra != -1.23e34
    spectra_wavelengths = spectra_wavelengths[mask]
    spectra = spectra[mask]

    return spectra, spectra_wavelengths

def _todict(matobj):
    '''
    A recursive function which constructs from matobjects nested dictionaries
    '''
    dict = {}
    for strg in matobj._fieldnames:
        elem = matobj.__dict__[strg]
        if isinstance(elem, spio.matlab.mio5_params.mat_struct):
            dict[strg] = _todict(elem)
        else:
            dict[strg] = elem
    return dict

def _check_keys(dict):
    '''
    checks if entries in dictionary are mat-objects. If yes
    todict is called to change them to nested dictionaries
    '''
    for key in dict:
        if isinstance(dict[key], spio.matlab.mio5_params.mat_struct):
            dict[key] = _todict(dict[key])
    return dict

def loadmat(filename):
    '''
    this function should be called instead of direct spio.loadmat
    as it cures the problem of not properly recovering python dictionaries
    from mat files. It calls the function check keys to cure all entries
    which are still mat-objects
    '''
    data = spio.loadmat(filename, struct_as_record=False, squeeze_me=True)
    return _check_keys(data)

materials = ["alunite", "concrete", "green_grass", "kaolinite", "limestone",
             "muscovite", "melting_snow", "oak_leaf", "sheet_metal", "plywood"]

def read_spectra(material, directory):
    spectra_fn = glob.glob(directory + material + "/splib07a*AREF.txt")[0]
    wavelengths_fn = glob.glob(directory + material + "/*Wavelengths*.txt")[0]

    spectra, wavelengths = read_usgs(material, spectra_fn, wavelengths_fn)

    wavelengths_ = np.linspace(0.5, 2, 100)
    idx = [wavelengths.sub(wl).abs().idxmin() for wl in wavelengths_]

    spectra = spectra[idx]

    return torch.tensor(spectra.to_numpy(), dtype=torch.float)

def get_spectra(R, directory="data/"):
    return torch.stack([read_spectra(material, directory=directory) for material in materials[:R]])
