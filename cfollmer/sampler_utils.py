import torch
from collections import OrderedDict
from layers import ResBlock, get_timestep_embedding


def detach_state_dict(state_dict):
    d = OrderedDict()

    for k, v in state_dict.items():
        d[k] = v.detach()
    return v


class SimpleForwardNet(torch.nn.Module):

    def __init__(self, input_dim=1):
        super(SimpleForwardNet, self).__init__()

        width = 20
        self.nn = torch.nn.Sequential(
            torch.nn.Linear(input_dim + 1, width), torch.nn.ReLU(),
            torch.nn.Linear(width, width), torch.nn.ReLU(),
            torch.nn.Linear(width, width), torch.nn.ReLU(),
            torch.nn.Linear(width, width), torch.nn.ReLU(),
            torch.nn.Linear(width, input_dim)
        )

        self.nn[-1].weight.data.fill_(0.0)

    def forward(self, x):
        return self.nn(x)


class SimpleForwardNetBN(torch.nn.Module):

    def __init__(self, input_dim=1):
        super(SimpleForwardNetBN, self).__init__()

        width = 20
        self.nn = torch.nn.Sequential(
            torch.nn.Linear(input_dim + 1, width), torch.nn.BatchNorm1d(width, affine=False), torch.nn.ReLU(),
            torch.nn.Linear(width, width), torch.nn.BatchNorm1d(width, affine=False), torch.nn.ReLU(),
            torch.nn.Linear(width, width), torch.nn.BatchNorm1d(width, affine=False), torch.nn.ReLU(),
            torch.nn.Linear(width, width), torch.nn.BatchNorm1d(width, affine=False), torch.nn.ReLU(),
            torch.nn.Linear(width, input_dim)
        )

        self.nn[-1].weight.data.fill_(0.0)

    def forward(self, x):
        return self.nn(x)


class ResNetScoreNetwork(torch.nn.Module):

    def __init__(self,
                 input_dim: int = 1,
                 pos_dim: int = 16,
                 res_block_initial_widths=None,
                 res_block_final_widths=None,
                 res_block_inner_layers=None,
                 activation=torch.nn.SiLU()):
        super().__init__()
        if res_block_initial_widths is None:
            res_block_initial_widths = [pos_dim, pos_dim, pos_dim]
        if res_block_final_widths is None:
            res_block_final_widths = [pos_dim, pos_dim, pos_dim]
        if res_block_inner_layers is None:
            res_block_inner_layers = [128, 128]

        self.temb_dim = pos_dim

        # ResBlock Sequence
        res_layers = []
        initial_dim = input_dim
        for initial, final in zip(res_block_initial_widths, res_block_final_widths):
            res_layers.append(ResBlock(initial_dim, initial, final, res_block_inner_layers, activation))
            initial_dim = initial + final
        self.res_sequence = torch.nn.Sequential(*res_layers)

        # Time FCBlock
        self.time_block = torch.nn.Sequential(torch.nn.Linear(self.temb_dim, self.temb_dim * 2), activation)

        # Final_block
        self.final_block = torch.nn.Sequential(torch.nn.Linear(self.temb_dim * 2 + initial_dim, input_dim))

    def forward(self, x, t):
        # t needs the same shape as x (except for the final dim, which is 1)
        t_emb = get_timestep_embedding(t, self.temb_dim)
        t_emb = self.time_block(t_emb)
        x_emb = self.res_sequence(x)
        h = torch.cat([x_emb, t_emb], -1)
        return self.final_block(h)


class LeNet5MNIST(torch.nn.Module):
    def __init__(self, channels: int = 1, out_dims: int = 10):
        super().__init__()

        # make_functional does not work with LazyLinear
        self.net = torch.nn.Sequential(
            torch.nn.Conv2d(1, 6, (5, 5), padding=2), torch.nn.Sigmoid(),
            torch.nn.AvgPool2d((2, 2), 2),
            torch.nn.Conv2d(6, 16, (5, 5)), torch.nn.Sigmoid(),
            torch.nn.AvgPool2d((2, 2), 2),
            torch.nn.Flatten(),
            torch.nn.Linear(400, 120), torch.nn.Sigmoid(),
            torch.nn.Linear(120, 84), torch.nn.Sigmoid(),
            torch.nn.Linear(84, 10)
        )

    def forward(self, x):
        return self.net(x)


class FollmerSDE(torch.nn.Module):

    def __init__(self, state_size=1, brownian_size=1,
                 batch_size=10, γ=1.0, drift=SimpleForwardNet,
                 device="cpu"):
        super().__init__()
        self.noise_type = 'diagonal'
        self.sde_type = 'ito'

        drift = SimpleForwardNetBN if drift is None else drift

        self.state_size = state_size
        self.brownian_size = brownian_size
        self.batch_size = batch_size

        self.device = device

        self.γ = γ
        self.μ = drift(input_dim=state_size).to(device)

        self.μ_detached = drift(input_dim=state_size).to(device)

    # Drift
    def f(self, t, y):
        # This is ugly should be cleaned up made more clear
        d = y.shape[0] if len(y.shape) == 2 else y.shape[1]
        t_ = t.to(self.device) * torch.ones(d, 1).to(self.device)
        t_ = t_ if len(y.shape) == 2 else t_.T[..., None]
        y = torch.cat((y, t_), dim=-1)

        return self.μ(y)  # shape (batch_size, state_size)

    def f_detached(self, t, y):
        # For STL estimator

        d = y.shape[0] if len(y.shape) == 2 else y.shape[1]
        t_ = t.to(self.device) * torch.ones(d, 1).to(self.device)
        t_ = t_ if len(y.shape) == 2 else t_.T[..., None]
        y = torch.cat((y, t_), dim=-1)

        return self.μ_detached(y)

        # Diffusion

    def g(self, t, y):
        diffusion = (torch.ones_like(y).to(self.device) * self.γ)
        return diffusion
