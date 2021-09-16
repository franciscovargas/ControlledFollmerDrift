import torch



class SimpleForwardNet(torch.nn.Module):

    def __init__(self, input_dim=1):
        super(SimpleForwardNet, self).__init__()
        
        width = 20
        self.nn = torch.nn.Sequential(
            torch.nn.Linear(input_dim + 1, width), torch.nn.ReLU(),
            torch.nn.Linear(width, width), torch.nn.ReLU(),
            torch.nn.Linear(width, width), torch.nn.ReLU(),
            torch.nn.Linear(width, input_dim )
        )
        
        self.nn[-1].weight.data.fill_(0.0)
        

    def forward(self, x):
        return self.nn(x)


class FollmerSDE(torch.nn.Module):

    def __init__(self, state_size=1, brownian_size=1, 
                 batch_size=10, γ=1.0, drift=SimpleForwardNet,
                device="cpu"):
        super().__init__()
        self.noise_type = 'diagonal'
        self.sde_type = 'ito'
        
        self.state_size = state_size
        self.brownian_size = brownian_size
        self.batch_size = batch_size
        
        self.device = device

        self.γ = γ
        self.μ = SimpleForwardNet(input_dim=state_size).to(device)

    # Drift
    def f(self, t, y):
        
        # This is ugly should be cleaned up made more clear
        d = y.shape[0] if len(y.shape) == 2 else y.shape[1]
        t_ = t.to(self.device) * torch.ones(d,1).to(self.device)
        t_ = t_ if len(y.shape) == 2 else t_.T[...,None]
        y = torch.cat((y, t_), dim=-1)

        return self.μ(y)   # shape (batch_size, state_size)

    # Diffusion
    def g(self, t, y):
        diffusion = (torch.ones_like(y).to(self.device) * self.γ)  
        return diffusion