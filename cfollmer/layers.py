import math

import torch
import torch.nn.functional as F


class ResBlock(torch.nn.Module):
    def __init__(self,
                 input_dim: int,
                 initial_layer_width: int = 10,
                 final_layer_width: int = 10,
                 layer_widths=None,
                 activation=torch.nn.SiLU()
                 ):
        super().__init__()

        if layer_widths is None:
            layer_widths = [128, 128]

        self.first_layer = torch.nn.Sequential(torch.nn.Linear(input_dim, initial_layer_width), activation)

        layers = []
        prev_width = initial_layer_width

        for i in range(0, len(layer_widths)):
            layers.append(torch.nn.Linear(prev_width, layer_widths[i]))
            layers.append(activation)
            prev_width = layer_widths[i]

        layers.append(torch.nn.Linear(prev_width, final_layer_width))
        layers.append(activation)
        self.net = torch.nn.Sequential(*layers)

    def forward(self, x):
        out_first = self.first_layer(x)
        out_sequence = self.net(out_first)
        return torch.cat((out_first, out_sequence), -1)


def get_timestep_embedding(timesteps, embedding_dim=128):
    half_dim = embedding_dim // 2
    emb = math.log(10000) / (half_dim - 1)
    emb = torch.exp(torch.arange(half_dim, dtype=torch.float, device=timesteps.device) * -emb)

    emb = timesteps.float() * emb.unsqueeze(0)
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
    if embedding_dim % 2 == 1:  # zero pad
        emb = F.pad(emb, [0, 1])

    return emb
