import torch

from .layers import ResBlock, get_timestep_embedding


class AbstractDrift(torch.nn.Module):

    def __init__(self):
        super(AbstractDrift, self).__init__()

    def forward(self, x, t):
        x = torch.cat((x, t), dim=-1)
        return self.nn(x)


class SimpleForwardNet(AbstractDrift):

    def __init__(self, input_dim=1, width=20):
        super(SimpleForwardNet, self).__init__()

        self.input_dim = input_dim
        
        self.nn = torch.nn.Sequential(
            torch.nn.Linear(input_dim + 1, width), torch.nn.ReLU(),
            torch.nn.Linear(width, width), torch.nn.ReLU(),
            torch.nn.Linear(width, width), torch.nn.ReLU(),
            torch.nn.Linear(width, width), torch.nn.ReLU(),
            torch.nn.Linear(width, input_dim),
        )
        
        self.nn[-1].weight.data.fill_(0.0)

    
class SimpleForwardNetBN(AbstractDrift):

    def __init__(self, input_dim=1, width=20):
        super(SimpleForwardNetBN, self).__init__()

        self.input_dim = input_dim
        
        self.nn = torch.nn.Sequential(
            torch.nn.Linear(input_dim + 1, width), torch.nn.BatchNorm1d(width, affine=False), torch.nn.Softplus(),
            torch.nn.Linear(width, width), torch.nn.BatchNorm1d(width, affine=False), torch.nn.Softplus(),
            torch.nn.Linear(width, width), torch.nn.BatchNorm1d(width, affine=False), torch.nn.Softplus(),
            torch.nn.Linear(width, width), torch.nn.BatchNorm1d(width, affine=False), torch.nn.Softplus(),
            torch.nn.Linear(width, input_dim)
        )
        
        self.nn[-1].weight.data.fill_(0.0)
        self.nn[-1].bias.data.fill_(0.0)


class ResNetScoreNetwork(AbstractDrift):

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

        self.input_dim = input_dim

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
