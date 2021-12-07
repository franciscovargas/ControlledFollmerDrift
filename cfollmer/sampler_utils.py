import torch
import torchsde
import copy


class FollmerSDE(torch.nn.Module):

    def __init__(self, gamma, drift):
        super().__init__()

        self.noise_type = 'diagonal'
        self.sde_type = 'ito'
        self.gamma = gamma
        self.drift = drift
        self.dim = drift.input_dim
        
    def f(self, t, y):
        return self.drift(y, t)
        
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

        return torch.cat((f, kl_row[...,None]), dim=1)

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
        return self.sample_trajectory(batch_size, dt=dt, device=device)[0][-1][:, :-1]


class DecoupledSDE(torch.nn.Module):

    def __init__(
            self, 
            gamma: float,
            global_drift: torch.nn.Module,
            local_drift: torch.nn.Module,
            global_dim: int,
            local_dim: int,
            data_dim: int,
            ):

        super().__init__()

        self.noise_type = 'diagonal'
        self.sde_type = 'ito'

        self.gamma = gamma

        self.global_drift = global_drift
        self.local_drift = local_drift

        self.global_dim = global_dim
        self.local_dim = local_dim
        self.data_dim = data_dim

    def f(self, t, params):

        # Assuming that self.data exists and has dim
        # [data_batch_size, data_dim]

        # Assuming that y has shape
        # [param_batch_size, global_dim + data_batch_size * local_dim)]

        global_dim = self.global_dim
        local_dim = self.local_dim
        data_dim = self.data_dim

        param_batch_size = params.shape[0]
        data_batch_size = (params.shape[1] - global_dim) // local_dim

        global_params = params[:, :global_dim]
        local_params = params[:, global_dim:]

        # First compute the drift for the global parameters

        global_f = self.global_drift(global_params, t)

        # Now compute the drift for the local parameters

        # batch size for the drift function
        batch_size = param_batch_size * data_batch_size

        # Force local params have shape
        # [batch_size, local_dim]
        local_params = local_params.reshape(batch_size, -1)

        # Interleave the global parameter to match
        # [batch_size, global_dim]
        global_params_batched = torch.repeat_interleave(global_params, data_batch_size, dim=0)

        # Interleave the data to match
        # [batch_size, data_dim]
        data_batched = self.data.repeat(param_batch_size, 1)

        local_drift_input = [global_params_batched, local_params, data_batched]
        local_drift_input = torch.cat(local_drift_input, dim=-1)

        local_f = self.local_drift(local_drift_input, t)
        local_f = local_f.view(param_batch_size, -1)

        f = torch.cat([global_f, local_f], dim=-1)

        return f

    def g(self, t, params):
        return torch.sqrt(self.gamma * torch.ones_like(params))

    def sample_trajectory(self, param_batch_size, dt=0.05, device=None):
        data_batch_size = self.data.shape[0]
        
        param_init = torch.zeros((param_batch_size, self.global_dim + self.local_dim * data_batch_size), device=device)

        n_steps = int(1.0 / dt)

        ts = torch.linspace(0, 1, n_steps, device=device)

        param_trajectory = torchsde.sdeint(self, param_init, ts, method="euler", dt=dt)

        return param_trajectory, ts

    def sample(self, batch_size, dt=0.05, device=None):
        return self.sample_trajectory(batch_size, dt=dt, device=device)[0][-1]
