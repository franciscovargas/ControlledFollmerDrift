from typing import Callable
from typing import Optional
from typing import Union
from typing import Tuple
from typing import List

from tqdm import tqdm

import torch
from . import utils

def ula(
        x_init: Union[torch.Tensor, Tuple[torch.Tensor]],
        grad_U: Callable,
        step_sizes: Union[torch.Tensor, Callable],
        n_steps: Optional[int] = 1,
        device: Optional[torch.device] = None,
        verbose: Optional[bool] = False,
        ):

    if type(x_init) == torch.Tensor:
        x_init = (x_init,)

    n_vars = len(x_init)

    xs = []
    for i in range(n_vars):
        empty = torch.empty_like(x_init[i], device=device, dtype=torch.float)
        xs.append(utils.repeat(empty, n_steps + 1))
        xs[i][0] = x_init[i]

    curr_state = x_init

    iterator = range(n_steps)
    if verbose:
        iterator = tqdm(iterator)

    for i in iterator:
        grads = grad_U(*curr_state)

        for j in range(n_vars):

            if type(step_sizes) == torch.Tensor:
                s = step_sizes[j]
            else:
                s = step_sizes(i, j)
            curr_state[j] = curr_state[j] + 0.5 * s * grads[j] + s ** 0.5 * torch.randn_like(curr_state[j])

        for j in range(n_vars):
            xs[j][i + 1] = curr_state[j]

    return xs
