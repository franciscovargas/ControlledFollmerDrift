from typing import Callable
from typing import Optional
from typing import Union
from typing import Tuple
from typing import List

import torch

from . import utils

class Constraint:
    def h(self, x) -> torch.Tensor:
        raise NotImplementedError

    def h_dual(self, y) -> torch.Tensor:
        raise NotImplementedError

    def h_grad(self, x):
        raise NotImplementedError

    def h_dual_grad(self, y):
        raise NotImplementedError

    def logdethess(self, y):
        raise NotImplementedError

class Simplex(Constraint):

    def h(self, x : torch.Tensor) -> torch.Tensor:
        dims = list(range(1, len(x.shape) + 1))

        s = torch.sum(x, dim=-1)

        x1 = torch.unsqueeze(x, dim=-2)
        x2 = torch.unsqueeze(x, dim=-1)

        return torch.sum(x1 @ torch.log(x2) + (1 - s) * torch.log(1 - s), dim=dims)

    def h_dual(self, y : torch.Tensor) -> torch.Tensor:
        lse = torch.logsumexp(utils.extend_zeros(y), dim=-1)
        return torch.sum(lse)

    def h_grad(self, x : torch.Tensor) -> torch.Tensor:
        return torch.log(x) - torch.log(1 - torch.sum(x, dim=-1, keepdim=True))

    def h_dual_grad(self, y : torch.Tensor) -> torch.Tensor:
        return torch.softmax(utils.extend_zeros(y), dim=-1)[..., :-1]

    def logdethess(self, y : torch.Tensor) -> torch.Tensor:
        return torch.sum(y) - (1 + y.shape[-1]) * self.h_dual(y)

class Hypercube(Constraint):

    simplex = Simplex()

    def h(self, x : torch.Tensor) -> torch.Tensor:
        return self.simplex.h(x[..., None])

    def h_dual(self, y : torch.Tensor) -> torch.Tensor:
        return self.simplex.h_dual(y[..., None])

    def h_grad(self, x : torch.Tensor) -> torch.Tensor:
        return torch.squeeze(self.simplex.h_grad(x[..., None]), dim=-1)

    def h_dual_grad(self, y : torch.Tensor) -> torch.Tensor:
        return torch.squeeze(self.simplex.h_dual_grad(y[..., None]), dim=-1)

    def logdethess(self, y : torch.Tensor) -> torch.Tensor:
        return self.simplex.logdethess(y[..., None])

class Composite(Constraint):

    def __init__(self, clist):
        self.clist = clist

    def h(self, *xs : Tuple[torch.Tensor]) -> torch.Tensor:
        return torch.stack([c.h(x) for c, x in zip(self.clist, xs)]).sum()

    def h_dual(self, *ys : Tuple[torch.Tensor]) -> torch.Tensor:
        return torch.stack([c.h_dual(y) for c, y in zip(self.clist, ys)]).sum()

    def h_grad(self, *xs : Tuple[torch.Tensor]) -> List[torch.Tensor]:
        return [c.h_grad(x) for c, x in zip(self.clist, xs)]

    def h_dual_grad(self, *ys : Tuple[torch.Tensor]) -> List[torch.Tensor]:
        return [c.h_dual_grad(y) for c, y in zip(self.clist, ys)]

    def logdethess(self, *ys : Tuple[torch.Tensor]) -> List[torch.Tensor]:
        return torch.stack([c.logdethess(y) for c, y in zip(self.clist, ys)]).sum()

def apply_constraints(U: Callable, c: Constraint) -> Callable:

    def W(*ys):
        return U(*c.h_dual_grad(*ys)) + c.logdethess(*ys)

    return W
