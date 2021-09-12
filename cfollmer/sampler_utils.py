import torch



class SimpleForwardNet(torch.nn.Module):

    def __init__(self, input_dim=1):
        super(SimpleForwardNet, self).__init__()

        self.nn = torch.nn.Sequential(
            torch.nn.Linear(input_dim, 20), torch.nn.ReLU(),
            torch.nn.Linear(20, 20), torch.nn.ReLU(),
            torch.nn.Linear(20, input_dim)
        )
        

    def forward(self, x):
        return self.nn(x)


class FollmerSDE(torch.nn.Module):

    def __init__(self, state_size=1, brownian_size=1, 
                 batch_size=10, γ=1.0, drift=SimpleForwardNet,
                device="cpu"):
        super().__init__()
        self.noise_type = 'scalar'
        self.sde_type = 'ito'
        
        self.state_size = state_size
        self.brownian_size = brownian_size
        self.batch_size = batch_size
        
        self.device = device

        self.γ = γ
        self.μ = SimpleForwardNet(input_dim=state_size).to(device)

    # Drift
    def f(self, t, y):
        return self.μ(y)  # shape (batch_size, state_size)

    # Diffusion
    def g(self, t, y):
        return (torch.ones_like(y).to(self.device) * self.γ)[:, :, None]