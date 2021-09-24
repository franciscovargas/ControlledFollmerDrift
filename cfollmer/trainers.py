import torch
import math
from cfollmer.objectives import relative_entropy_control_cost, stl_relative_entropy_control_cost
from cfollmer.sampler_utils import FollmerSDE
from tqdm.notebook import tqdm



def basic_batched_trainer(
        γ, Δt, ln_prior, log_likelihood_vmap, dim, X_train, y_train,
        method="euler", stl=True, adjoint=False, optimizer=None,
        num_steps=200, batch_size_data=None, batch_size_Θ=200, lr=0.001,
        batchnorm=True, device="cpu"
    ):

    t_size = int(math.ceil(1.0/Δt))
    ts = torch.linspace(0, 1, t_size).to(device)
    
    Θ_0 = torch.zeros((batch_size_Θ, dim)).to(device) 
    
    sde = FollmerSDE(dim, dim, batch_size_Θ  , γ=γ, device=device).to(device)
    optimizer = torch.optim.Adam(sde.μ.parameters(), lr=lr, weight_decay =0.5)
    #     optimizer = torch.optim.LBFGS(gpr.parameters(), lr=0.01)
    losses = []
    # with torch.autograd.set_detect_anomaly(True):
    loss_ = stl_relative_entropy_control_cost if stl else relative_entropy_control_cost

    for i in tqdm(range(num_steps)):
        optimizer.zero_grad()
        
        loss_call = lambda: loss_(
            sde, Θ_0.float(),
            X_train.float(), y_train.float(),
            ln_prior, log_likelihood_vmap, γ=γ,
            batchnorm=True, device=device
        )

        if isinstance(optimizer, torch.optim.LBFGS):
            def closure():
                loss = loss_call()
                optimizer.zero_grad()
                loss.backward()
                return loss

            optimizer.step(closure)
            losses.append(closure().item())
        else:
            loss = loss_call()
            optimizer.zero_grad()
            loss.backward()

            optimizer.step()       
            losses.append(loss.item())
        if stl:
            sde.μ_detached.load_state_dict((sde.μ.state_dict()))
    return sde, losses