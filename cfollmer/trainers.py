import torch
import math
from cfollmer.objectives import relative_entropy_control_cost, stl_relative_entropy_control_cost, simplified
from cfollmer.sampler_utils import FollmerSDE
from tqdm.notebook import tqdm
import gc


def basic_batched_trainer(
        γ, Δt, ln_prior, log_likelihood_vmap, dim, X_train, y_train,net=None,
        method="euler", stl=True, adjoint=False, optimizer=None,
        num_steps=200, batch_size_data=None, batch_size_Θ=200, lr=0.001,
        batchnorm=True, device="cpu", drift=None, debug=False, simple=False
    ):

    t_size = int(math.ceil(1.0/Δt))
    ts = torch.linspace(0, 1, t_size).to(device)
    
    Θ_0 = torch.zeros((batch_size_Θ, dim)).to(device) 
    
    sde = FollmerSDE(dim, dim, batch_size_Θ  , γ=γ, device=device, drift=drift).to(device)
    optimizer = torch.optim.Adam(sde.μ.parameters(), lr=lr, weight_decay=0.5)
    #     optimizer = torch.optim.LBFGS(gpr.parameters(), lr=0.01)
    losses = []

    if net is not None:
        log_likelihood_vmap_c = log_likelihood_vmap
        def log_likelihood_vmap(Θ, X, y):
            return log_likelihood_vmap_c(Θ, X, y, net=net)
    
    avg_loss_list = []
    batch_size = len(X_train) if batch_size_data is None else batch_size_data
    n_batches = int(len(X_train) / batch_size_data) 
    
    
    loss_ = stl_relative_entropy_control_cost if stl else relative_entropy_control_cost#
    if simple:
        loss_ = simplified

    for i in tqdm(range(num_steps)):
        
        # shuffle train (refresh):
        perm = torch.randperm(len(X_train))

        # stochastic minibatch GD (MC estimate of gradient via subsample)
        for batch in tqdm(range(n_batches)): # Make sure to go through whole dtaset
            if batch > 0 or  i > 0 and net is not None:
                thetas = sde.last_samples 
                cls = net.predict(X_train[:2000], thetas)
                print( "ACCURACY", (cls == y_train[:2000]).float().mean())
            batch_X = X_train[perm,...][batch*batch_size_data:(batch+1)*batch_size_data,]
            batch_y = y_train[perm,...][batch*batch_size_data:(batch+1)*batch_size_data,]
            
            optimizer.zero_grad()

            loss_call = lambda: loss_(
                sde, Θ_0.float(),
                batch_X.float(), batch_y.float(),
                ln_prior, log_likelihood_vmap, γ=γ,
                batchnorm=batchnorm, device=device,adjoint=adjoint,
                debug=debug
            )

            if isinstance(optimizer, torch.optim.LBFGS):
                def closure():
                    loss = loss_call()
                    optimizer.zero_grad()
                    loss.backward()
                    return loss

                optimizer.step(closure)
                avg_loss_list.append(closure().item())
            else:
                loss = loss_call()

                print(loss.item())
                optimizer.zero_grad()
                loss.backward()

                optimizer.step()       
                avg_loss_list.append(loss.item())
            if stl:
                sde.μ_detached.load_state_dict((sde.μ.state_dict()))
            # Clear up memory leaks
            gc.collect()
            torch.cuda.empty_cache()
        losses.append(torch.mean(torch.tensor(avg_loss_list)))
        avg_loss_list = []
    return sde, losses
