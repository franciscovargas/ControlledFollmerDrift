import torch
import torchsde

class FollmerSDE(torch.nn.Module):

    def __init__(self, gamma, drift_network):
        super().__init__()

        self.noise_type = 'diagonal'
        self.sde_type = 'ito'
        self.gamma = gamma
        self.drift_network = drift_network
        self.dim = drift_network.input_dim
        
    def f(self, t, y):
        t_ = t * torch.ones((y.shape[0], 1), device=y.device)
        return self.drift_network(t_, y)
        
    def g(self, t, y):
        return torch.sqrt(self.gamma * torch.ones_like(y))

    def sample_trajectory(self, batch_size, dt=0.05, device=None):
        param_init = torch.zeros((batch_size, self.dim), device=device)

        n_steps = int(1.0 / dt)

        ts = torch.linspace(0, 1, n_steps, device=device)

        param_trajectory = torchsde.sdeint(self, param_init, ts, method="euler", dt=dt)

        return param_trajectory, ts

    def sample(self, batch_size, dt=0.05, device=None):
        return self.sample_trajectory(batch_size, dt=dt, device=device)[0][-1]
