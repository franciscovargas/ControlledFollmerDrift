import torch
import math
from cfollmer.objectives import relative_entropy_control_cost, stl_relative_entropy_control_cost
from cfollmer.sampler_utils import FollmerSDE
from tqdm.notebook import tqdm



def basic_batched_trainer(
    γ, Δt, ln_prior, log_likelihood, state_dim, X_train, y_train,
    method="euler", stl=True, adjoint=False, optimizer=None,
    num_steps=400, batch_size_data=None, batch_size_Θ=50,
    batchnorm=False, device="cpu"
):

    t_size = int(math.ceil(1.0/Δt))
    ts = torch.linspace(0, 1, t_size).to(device)
    
    Θ_0 = torch.zeros((batch_size_Θ, state_dim)).to(device)

    sde = FollmerSDE(
        state_dim, state_dim, batch_size_Θ, γ=γ, device=device
    ).to(device)
    
    optimizer = torch.optim.Adam(
        sde.μ.parameters(),
        lr=0.01,
        weight_decay=0.5
    ) if optimizer is None else optimizer
    #     optimizer = torch.optim.LBFGS(gpr.parameters(), lr=0.01)
    losses = []

    loss_ = stl_relative_entropy_control_cost if stl else relative_entropy_control_cost
    for i in tqdm(range(num_steps)):
        optimizer.zero_grad()
        
#         loss_call = lambda: loss_(
#             sde, Θ_0,
#             X_train, y_train,
#             ln_prior, log_likelihood, γ=γ,
#             batchnorm=batchnorm, device=device,
#             method=method, adjoint=adjoint
#         )

        if isinstance(optimizer, torch.optim.LBFGS):
            def closure():
                loss = loss_(
                    sde, Θ_0,
                    X_train, y_train,
                    ln_prior, log_likelihood, γ=γ,
                    batchnorm=batchnorm, device=device,
                    method=method, adjoint=adjoint
                )
                optimizer.zero_grad()
                loss.backward()
                return loss

            optimizer.step(closure)
            losses.append(closure().item())
        else:
            loss = loss_(
                sde, Θ_0,
                X_train, y_train,
                ln_prior, log_likelihood, γ=γ,
                batchnorm=batchnorm, device=device,
                method=method, adjoint=adjoint
            )
            optimizer.zero_grad()
            loss.backward()

            optimizer.step()
            losses.append(loss.item())
        if stl:
            sde.μ_detached.load_state_dict((sde.μ.state_dict()))
    return sde, losses