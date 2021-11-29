import torch
import torchsde
import copy
import math


class FollmerSDE(torch.nn.Module):

    def __init__(self, gamma, drift_network):
        super().__init__()

        self.noise_type = 'diagonal'
        self.sde_type = 'ito'
        self.gamma = gamma
        self.drift_network = drift_network
        self.drift_network_detatched = copy.deepcopy(drift_network)
        self.dim = drift_network.input_dim
        
    def f(self, t, y, detach=False):
        t_ = t * torch.ones((y.shape[0], 1), device=y.device)
        if detach:
            return self.drift_network_detatched(y, t_)
        return self.drift_network(y, t_)
        
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
    
    
class FollmerSDE_STL(torch.nn.Module):

    def __init__(self, gamma, drift_network):
        super().__init__()

        self.noise_type = 'diagonal'
        self.sde_type = 'ito'
        self.gamma = gamma
        self.drift_network = drift_network
        self.drift_network_detatched = copy.deepcopy(drift_network)
        self.dim = drift_network.input_dim

    def f(self, t, y, detach=False):
        t_ = t * torch.ones((y.shape[0], 1), device=y.device)
        
        batch_size, dim = y.shape 
        
        # [batch_size, dim]
        f = self.drift_network(y[:,:-1], t_)
        
        # [batch_size, 1]
        kl_row = (f**2).sum(dim=-1) / (2 * self.gamma)
        
        try:
            return torch.cat((f, kl_row[...,None]), dim=1)
        except:
            import pdb;pdb.set_trace()

    def g(self, t, y):
        batch_size, dim = y.shape 
        ones = torch.ones_like(y)
        
        gammas = torch.sqrt(self.gamma * torch.ones_like(y)[:,:-1])
        zeros =  torch.ones_like(y)[:,1] * 0

        return  torch.cat((gammas, zeros[..., None]), dim=(1))

    def sample_trajectory(self, batch_size, dt=0.05, device=None):
        param_init = torch.zeros((batch_size, self.dim + 1), device=device)

        n_steps = int(1.0 / dt)

        ts = torch.linspace(0, 1, n_steps, device=device)

        param_trajectory = torchsde.sdeint(self, param_init, ts, method="euler", dt=dt)

        return param_trajectory, ts

    def sample(self, batch_size, dt=0.05, device=None):
        return self.sample_trajectory(batch_size, dt=dt, device=device)[0][-1][:-1]
