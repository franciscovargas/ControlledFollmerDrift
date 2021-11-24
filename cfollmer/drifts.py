import torch
import torch.nn.functional as F

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

    # def __init__(self,
    #              input_dim: int = 1,
    #              pos_dim: int = 16,
    #              res_block_initial_widths=None,
    #              res_block_final_widths=None,
    #              res_block_inner_layers=None,
    #              activation=torch.nn.SiLU()):
    def __init__(self, input_dim: int):
        super().__init__()
        res_block_initial_widths = [300, 300, 300]
        res_block_final_widths = [300, 300, 300]
        res_block_inner_layers = [300, 300, 300]

        self.input_dim = input_dim

        self.temb_dim = 128

        # ResBlock Sequence
        res_layers = []
        initial_dim = input_dim
        for initial, final in zip(res_block_initial_widths, res_block_final_widths):
            res_layers.append(ResBlock(initial_dim, initial, final, res_block_inner_layers, torch.nn.Softplus()))
            initial_dim = initial + final
        self.res_sequence = torch.nn.Sequential(*res_layers)

        # Time FCBlock
        self.time_block = torch.nn.Sequential(torch.nn.Linear(self.temb_dim, self.temb_dim * 2), torch.nn.Softplus())

        # Final_block
        self.final_block = torch.nn.Sequential(torch.nn.Linear(self.temb_dim * 2 + initial_dim, input_dim))

    def forward(self, x, t):
        # t needs the same shape as x (except for the final dim, which is 1)
        t_emb = get_timestep_embedding(t, self.temb_dim)
        t_emb = self.time_block(t_emb)
        x_emb = self.res_sequence(x)
        h = torch.cat([x_emb, t_emb], -1)
        return self.final_block(h)


class ResNetScoreNetworkLarge(AbstractDrift):

    # def __init__(self,
    #              input_dim: int = 1,
    #              pos_dim: int = 16,
    #              res_block_initial_widths=None,
    #              res_block_final_widths=None,
    #              res_block_inner_layers=None,
    #              activation=torch.nn.SiLU()):
    def __init__(self, input_dim: int):
        super().__init__()
        res_block_initial_widths = [300, 300, 300]
        res_block_final_widths = [300, 300, 300]
        res_block_inner_layers = [300, 300, 300]

        self.input_dim = input_dim

        self.temb_dim = 128

        # ResBlock Sequence
        res_layers = []
        initial_dim = input_dim
        for initial, final in zip(res_block_initial_widths, res_block_final_widths):
            res_layers.append(ResBlock(initial_dim, initial, final, res_block_inner_layers, torch.nn.Softplus()))
            initial_dim = initial + final
        self.res_sequence = torch.nn.Sequential(*res_layers)

        # Time FCBlock
        self.time_block = torch.nn.Sequential(torch.nn.Linear(self.temb_dim, self.temb_dim * 2), torch.nn.Softplus())

        # Final_block
        self.final_block = torch.nn.Sequential(torch.nn.Linear(self.temb_dim * 2 + initial_dim, input_dim))

        self.final_block[-1].weight.data.fill_(0.0)
        self.final_block[-1].bias.data.fill_(0.0)

    def forward(self, x, t):
        # t needs the same shape as x (except for the final dim, which is 1)
        t_emb = get_timestep_embedding(t, self.temb_dim)
        t_emb = self.time_block(t_emb)
        x_emb = self.res_sequence(x)
        h = torch.cat([x_emb, t_emb], -1)
        return self.final_block(h)


class ScoreNetwork(torch.nn.Module):

    # def __init__(self, encoder_layers=None, pos_dim=16, decoder_layers=None, x_dim=2):
    def __init__(self, input_dim: int):
        super().__init__()
        self.input_dim = input_dim
        decoder_layers = [300, 300]
        encoder_layers = [300]
        pos_dim = 150
        self.temb_dim = pos_dim
        t_enc_dim = pos_dim * 2
        self.locals = [encoder_layers, pos_dim, decoder_layers, input_dim]

        self.net = MLP(2 * t_enc_dim,
                       layer_widths=decoder_layers + [input_dim],
                       activate_final=False,
                       activation_fn=torch.nn.Softplus(),
                       final_zero=True)

        self.t_encoder = MLP(pos_dim,
                             layer_widths=encoder_layers + [t_enc_dim],
                             activate_final=True,
                             activation_fn=torch.nn.Softplus())

        self.x_encoder = MLP(input_dim,
                             layer_widths=encoder_layers + [t_enc_dim],
                             activate_final=True,
                             activation_fn=torch.nn.Softplus())

    def forward(self, x, t):
        if len(x.shape) == 1:
            x = x.unsqueeze(0)
        if len(t.shape) != len(x.shape):
            t = (torch.ones_like(x, device=x.device) * t)[..., 0:1]

        temb = get_timestep_embedding(t, self.temb_dim)
        temb = self.t_encoder(temb)
        xemb = self.x_encoder(x)
        h = torch.cat([xemb, temb], -1)
        out = self.net(h)
        return out


class MLP(torch.nn.Module):
    def __init__(self, input_dim, layer_widths, activate_final=False, activation_fn=F.relu, final_zero=False):
        super(MLP, self).__init__()
        layers = []
        prev_width = input_dim
        for layer_width in layer_widths:
            layers.append(torch.nn.Linear(prev_width, layer_width))
            # # same init for everyone
            # torch.nn.init.constant_(layers[-1].weight, 0)
            prev_width = layer_width
        self.input_dim = input_dim
        self.layer_widths = layer_widths
        if final_zero:
            layers[-1].weight.data.fill_(0.0)
            layers[-1].bias.data.fill_(0.0)
        self.layers = torch.nn.ModuleList(layers)
        self.activate_final = activate_final
        self.activation_fn = activation_fn

    def forward(self, x):
        for i, layer in enumerate(self.layers[:-1]):
            x = self.activation_fn(layer(x))
        x = self.layers[-1](x)
        if self.activate_final:
            x = self.activation_fn(x)
        return x
