from typing import Optional

import torch
import torch.nn.functional as F

from .layers import ResBlock, get_timestep_embedding


class AbstractDrift(torch.nn.Module):

    def __init__(self):
        super(AbstractDrift, self).__init__()

    def forward(self, x, t):
        batch_size = x.shape[0]

        t_ = t * torch.ones((batch_size, 1), device=x.device)

        x = torch.cat((x, t_), dim=-1)

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


class DecoupledDrift(AbstractDrift):

    def __init__(self, global_dim=1, local_dim=1, data_dim=1, width=20):
        super(DecoupledDrift, self).__init__()

        self.global_dim = global_dim
        self.local_dim = local_dim
        self.data_dim = data_dim
        
        self.nn = torch.nn.Sequential(
            torch.nn.Linear(global_dim + local_dim + data_dim + 1, width), torch.nn.BatchNorm1d(width, affine=False), torch.nn.Softplus(),
            torch.nn.Linear(width, width), torch.nn.BatchNorm1d(width, affine=False), torch.nn.Softplus(),
            torch.nn.Linear(width, width), torch.nn.BatchNorm1d(width, affine=False), torch.nn.Softplus(),
            torch.nn.Linear(width, width), torch.nn.BatchNorm1d(width, affine=False), torch.nn.Softplus(),
            torch.nn.Linear(width, local_dim)
        )
        
        self.nn[-1].weight.data.fill_(0.0)
        self.nn[-1].bias.data.fill_(0.0)


class ResNetScoreNetwork(AbstractDrift):

    def __init__(
            self,
            input_dim: int,
            width: Optional[int] = 300,
            ):
        super().__init__()
        res_block_initial_widths = 3 * [width]
        res_block_final_widths = 3 * [width]
        res_block_inner_layers = 3 * [width]

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
        batch_size = x.shape[0]

        t = t * torch.ones((batch_size, 1), device=x.device)

        # t needs the same shape as x (except for the final dim, which is 1)
        t_emb = get_timestep_embedding(t, self.temb_dim)
        t_emb = self.time_block(t_emb)
        x_emb = self.res_sequence(x)
        h = torch.cat([x_emb, t_emb], -1)
        return self.final_block(h)


class ScoreNetwork(torch.nn.Module):

    def __init__(self, input_dim: int):
        super().__init__()
        self.input_dim = input_dim
        decoder_layers = [300, 300]
        encoder_layers = [300, 300]
        pos_dim = 100
        self.temb_dim = pos_dim
        t_enc_dim = pos_dim * 2
        self.locals = [encoder_layers, pos_dim, decoder_layers, input_dim]

        self.net = MLP(2 * t_enc_dim,
                       layer_widths=decoder_layers + [input_dim],
                       activate_final=False,
                       activation_fn=torch.nn.Softplus())

        self.t_encoder = MLP(pos_dim,
                             layer_widths=encoder_layers + [t_enc_dim],
                             activate_final=False,
                             activation_fn=torch.nn.Softplus())

        self.x_encoder = MLP(input_dim,
                             layer_widths=encoder_layers + [t_enc_dim],
                             activate_final=False,
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
    def __init__(self, input_dim, layer_widths, activate_final=False, activation_fn=F.relu):
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

## Below are failed experiments

class HSUDriftAutoEncoder(torch.nn.Module):
    def __init__(self, width, height, channels):
        super(HSUDrift, self).__init__()

        self.width, self.height, self.channels = width, height, channels

        self.input_dim = width * height * (channels - 2) + 1

        self.encoder = torch.nn.Sequential(
                torch.nn.Conv2d(channels, 16, 7, stride=2),
                torch.nn.BatchNorm2d(16),
                torch.nn.ReLU(),
                torch.nn.Conv2d(16, 32, 5, stride=2),
                torch.nn.BatchNorm2d(32),
                torch.nn.ReLU(),
                torch.nn.Conv2d(32, 16, 3, stride=2),
                torch.nn.BatchNorm2d(16),
                torch.nn.ReLU(),
                torch.nn.AvgPool2d(3),
        )

        self.bottleneck = torch.nn.Sequential(
                torch.nn.Linear(400, 300),
                torch.nn.ReLU(),
                torch.nn.Linear(300, 400),
                torch.nn.ReLU(),
        )

        self.decoder = torch.nn.Sequential(
                torch.nn.ConvTranspose2d(16, 16, 3, stride=2),
                torch.nn.BatchNorm2d(16),
                torch.nn.ReLU(),
                torch.nn.ConvTranspose2d(16, 32, 4, stride=3),
                torch.nn.BatchNorm2d(32),
                torch.nn.ReLU(),
                torch.nn.ConvTranspose2d(32, 16, 5, stride=2),
                torch.nn.BatchNorm2d(16),
                torch.nn.ReLU(),
                torch.nn.ConvTranspose2d(16, channels - 1, 5, stride=2),
        )

    def forward(self, params, t):
        A = params[:, :-1]
        A = A.view(-1, self.width, self.height, self.channels - 2)
        sigma2_noise = params[:, -1]
        sigma2_noise = sigma2_noise[:, None, None]
        t = t[:, None]

        stacked = F.pad(A, (0, 2))
        stacked[..., -2] += sigma2_noise
        stacked[..., -1] += t

        stacked = torch.transpose(stacked, 1, 3)

        enc = self.encoder(stacked)
        enc = torch.flatten(enc, start_dim=1)
        enc = self.bottleneck(enc)
        enc = enc.view(-1, 16, 5, 5)

        out = self.decoder(enc)
        out = torch.transpose(out, 1, 3)

        sigma2_noise = torch.mean(out[:, :, :, -1], dim=[1, 2])[:, None]
        A = out[:, :, :, :-1].reshape(-1, self.width * self.height * (self.channels - 2))

        out = torch.cat([A, sigma2_noise], dim=1)

        return out

class HSUDrift(torch.nn.Module):
    def __init__(self, width, height, channels):
        super(HSUDrift, self).__init__()

        self.width, self.height, self.channels = width, height, channels

        self.input_dim = width * height * (channels - 2) + 1

        self.nn = torch.nn.Sequential(
                torch.nn.Conv2d(channels, 16, 5, padding=2, stride=4),
                torch.nn.BatchNorm2d(16),
                torch.nn.ReLU(),
                torch.nn.Conv2d(16, 16, 5, padding=2, stride=2),
                torch.nn.BatchNorm2d(16),
                torch.nn.ReLU(),
                torch.nn.ConvTranspose2d(16, 16, 5, padding=2, stride=2),
                torch.nn.BatchNorm2d(16),
                torch.nn.ReLU(),
                torch.nn.ConvTranspose2d(16, channels - 1, 5, padding=2, stride=4)
        )

        self.nn[-1].weight.data.fill_(0.0)
        self.nn[-1].bias.data.fill_(0.0)

    def forward(self, params, t):
        A = params[:, :-1]
        A = A.view(-1, self.width, self.height, self.channels - 2)
        sigma2_noise = params[:, -1]
        sigma2_noise = sigma2_noise[:, None, None]
        t = t[:, None]

        stacked = F.pad(A, (0, 2))
        stacked[..., -2] += sigma2_noise
        stacked[..., -1] += t

        stacked = torch.transpose(stacked, 1, 3)

        out = self.nn(stacked)
        out = torch.transpose(out, 1, 3)

        sigma2_noise = torch.mean(out[:, :, :, -1], dim=[1, 2])[:, None]
        A = out[:, :, :, :-1].reshape(-1, self.width * self.height * (self.channels - 2))

        out = torch.cat([A, sigma2_noise], dim=1)

        return out
