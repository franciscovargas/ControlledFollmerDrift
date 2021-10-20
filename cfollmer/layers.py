from typing import List
import torch


class ResBlock(torch.nn.Module):
    def __init__(self, input_dim: int,
                 layer_widths: List[int] = [128, 64, 32, 16],
                 activation = torch.nn.SiLU()
                 ):
        super().__init__()

        self.first_layer = torch.nn.Sequential(torch.nn.Linear(input_dim, layer_widths[0]), activation)

        layers = []
        prev_width = layer_widths[0]

        for i in range(1, len(layer_widths)):
            layers.append(torch.nn.Linear(prev_width, layer_widths[i]))
            layers.append(activation)
            prev_width = layer_widths[i]

        self.net = torch.nn.Sequential(*layers)

    def forward(self, x):
        out_first = self.first_layer(x)
        out_sequence = self.net(out_first)
        return torch.cat((out_first, out_sequence), -1)


