import gc
import os

import functorch
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.transforms import ToTensor
from tqdm import tqdm

import cfollmer.functional as functional
from cfollmer.drifts import SimpleForwardNetBN, ResNetScoreNetwork
from cfollmer.evaluation_utils import ECE
from cfollmer.objectives import relative_entropy_control_cost, stl_control_cost, stl_control_cost_aug
from cfollmer.sampler_utils import FollmerSDE, FollmerSDE_STL

device = "cuda" if torch.cuda.is_available() else "cpu"


class LeNet5Cifar10(torch.nn.Module):

    def __init__(self, n_classes):
        super(LeNet5Cifar10, self).__init__()

        self.feature_extractor = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2),
            torch.nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(kernel_size=2),
        )

        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(in_features=16 * 5 * 5, out_features=120),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=120, out_features=84),
            torch.nn.ReLU(),
            torch.nn.Linear(in_features=84, out_features=n_classes),
        )

    def forward(self, x):
        x = self.feature_extractor(x)
        x = torch.flatten(x, 1)
        logits = self.classifier(x)
        return logits


transform = transforms.Compose(
    [ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
     ])

CIFAR10_train = datasets.CIFAR10("data/cifar10/", download=True, transform=transform, train=True)
CIFAR10_test = datasets.CIFAR10("data/cifar10/", download=True, transform=transform, train=False)

N_train = len(CIFAR10_train)
N_test = len(CIFAR10_test)

model = LeNet5Cifar10(10).to(device)
func_model, params = functorch.make_functional(model)
size_list = functional.params_to_size_tuples(params)
dim = functional.get_number_of_params(size_list)

sigma2 = 1


def log_prior(params):
    return -torch.sum(params ** 2) / (2 * sigma2)


def log_likelihood(x, y, params):
    preds = func_model(functional.get_params_from_array(params, size_list), x)
    return -F.cross_entropy(preds, y, reduction="sum")


def log_likelihood_batch(x, y, params_batch):
    func = lambda params: log_likelihood(x, y, params)
    func = functorch.vmap(func)
    return func(params_batch)


def log_posterior(x, y, params):
    return log_prior(params) + (N_train / x.shape[0]) * log_likelihood(x, y, params)


def log_posterior_batch(x, y, params_batch):
    func = lambda params: log_posterior(x, y, params)
    func = functorch.vmap(func)
    return func(params_batch)


def train(gamma, n_epochs, data_batch_size, param_batch_size, dt=0.05, stl=False):
    if stl:
        # sde = FollmerSDE_STL(gamma, SimpleForwardNetBN(input_dim=dim, width=200)).to(device)
        sde = FollmerSDE_STL(gamma, ResNetScoreNetwork(input_dim=dim, final_zero=True)).to(device)
    else:
        sde = FollmerSDE(gamma, ResNetScoreNetwork(input_dim=dim, final_zero=True)).to(device)
    optimizer = torch.optim.Adam(sde.parameters(), lr=1e-5)

    dataloader_train = DataLoader(CIFAR10_train, shuffle=True, batch_size=data_batch_size, num_workers=2)

    losses = []

    for _ in range(n_epochs):
        epoch_losses = []
        for x, y in tqdm(iter(dataloader_train)):
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()

            partial_log_p = lambda params_batch: log_posterior_batch(x, y, params_batch)

            if stl:
                loss = stl_control_cost_aug(sde, partial_log_p, param_batch_size=param_batch_size, dt=dt, device=device)
            else:
                loss = relative_entropy_control_cost(sde, partial_log_p, param_batch_size=param_batch_size, dt=dt,
                                                     device=device)
            loss.backward()

            epoch_losses.append(loss.detach().cpu().numpy())
            optimizer.step()

            if stl:  # double check theres no references left
                sde.drift_network_detatched.load_state_dict((sde.drift_network.state_dict()))

        #  Memory leaks somewhere with sdeint / param_T = param_trajectory[-1]
        gc.collect()

        losses.append(epoch_losses)

    losses = np.array(losses)

    return sde, losses


def evaluate(param_samples):
    dataloader_test = DataLoader(CIFAR10_test, shuffle=False, batch_size=data_batch_size, num_workers=2)
    # dataloader_test = DataLoader(CIFAR10_train, shuffle=False, batch_size=data_batch_size, num_workers=2)

    all_predictions = []
    all_confidences = []
    all_logps = []

    for x, y in tqdm(iter(dataloader_test)):
        with torch.no_grad():
            x = x.to(device)
            y = y.to(device)

            predict_func = lambda params: func_model(functional.get_params_from_array(params, size_list), x)
            predict_func = functorch.vmap(predict_func)

            out = F.softmax(predict_func(param_samples), dim=-1)
            out = torch.mean(out, dim=0)

            confidences, predictions = torch.max(out, dim=1)

            all_predictions.append(predictions)
            all_confidences.append(confidences)

            all_logps.append(torch.mean(log_likelihood_batch(x, y, param_samples)))

    all_predictions = torch.hstack(all_predictions).cpu().numpy()
    all_confidences = torch.hstack(all_confidences).cpu().numpy()
    true_labels = np.array(CIFAR10_test.targets)
    # true_labels = np.array(CIFAR10_train.targets)

    accuracy = np.mean(all_predictions == true_labels)
    ece = ECE(all_confidences, all_predictions, true_labels)

    logp = torch.sum(torch.stack(all_logps)) / N_test
    logp = logp.cpu().numpy()
    return accuracy, ece, logp


# gamma_sqrt = [1, 0.5, 0.1, 0.08, *0.05*, 0.01]
gamma_sqrt = 0.05
gamma = gamma_sqrt ** 2
n_epochs = 10
data_batch_size = 32
param_batch_size = 32

num_exp = 5

weights_path = "weights/bnn_cifar10/resnet/"
results_path = "results/bnn_cifar10/resnet/"
if not os.path.exists(weights_path):
    os.makedirs(weights_path)
if not os.path.exists(results_path):
    os.makedirs(results_path)

for i in range(num_exp):
    sde, losses = train(gamma, n_epochs, data_batch_size, param_batch_size, dt=0.05, stl=True)
    torch.save(sde.state_dict(), f"{weights_path}weights-vargrad-sigma1-{i}.pt")
    #     plt.figure(figsize=(15, 15))
    #     plt.plot(losses[0])
    #     plt.savefig(f"experiments/losses_{gamma_sqrt}_{i}.png")
    #     plt.close()
    del sde
    gc.collect()
    torch.cuda.empty_cache()


accuracies, eces, logps = [], [], []

for i in range(num_exp):
    # sde = FollmerSDE(gamma, SimpleForwardNetBN(input_dim=dim, width=200)).to(device)
    sde = FollmerSDE(gamma, ResNetScoreNetwork(dim)).to(device)
    sde.load_state_dict(torch.load(f"{weights_path}weights-vargrad-sigma1-{i}.pt"))

    with torch.no_grad():
        param_samples = sde.sample(100, dt=0.005, device=device)

    accuracy, ece, logp = evaluate(param_samples)

    accuracies.append(accuracy)
    eces.append(ece)
    logps.append(logp)

accuracies = np.array(accuracies)
eces = np.array(eces)
logps = np.array(logps)

SBP_df = pd.DataFrame({"Accuracy": accuracies, "ECE": eces, "log predictive": logps})

SBP_df.describe()
with open(f"{results_path}/cifar10_cfollmer_vargrad.txt", 'w') as f:
    f.write(str(SBP_df.describe()))
print("FINISHED CFOLLMER")

# SGLD
"""
@torch.enable_grad()
def gradient(x, y, params):
    params_ = params.clone().requires_grad_(True)
    loss = log_posterior(x, y, params_)
    grad, = torch.autograd.grad(loss, params_)
    return loss.detach().cpu().numpy(), grad


def step_size(n):
    return 1e-4 / (1 + n) ** 0.55


def sgld(n_epochs, data_batch_size):
    dataloader_train = DataLoader(CIFAR10_train, shuffle=True, batch_size=data_batch_size, num_workers=2)
    params = torch.cat([param.flatten() for param in model.parameters()]).detach()
    losses = []
    step = 0
    for _ in range(n_epochs):
        epoch_losses = []
        for x, y in tqdm(iter(dataloader_train)):
            x = x.to(device)
            y = y.to(device)

            eps = step_size(step)
            loss, grad = gradient(x, y, params)
            params = params + 0.5 * eps * grad  # + np.sqrt(eps) * torch.randn_like(params)
            step += 1
            epoch_losses.append(loss)

        losses.append(epoch_losses)

    param_samples = []

    iterator = iter(dataloader_train)
    for _ in range(100):
        x = x.to(device)
        y = y.to(device)

        eps = step_size(step)
        loss, grad = gradient(x, y, params)
        params = params + 0.5 * eps * grad + np.sqrt(eps) * torch.randn_like(params)
        param_samples.append(params)
        step += 1

    param_samples = torch.stack(param_samples)
    losses = np.array(losses)

    return param_samples, losses


accuracies, eces, logps = [], [], []

for i in range(num_exp):
    param_samples, losses = sgld(n_epochs, data_batch_size)

    accuracy, ece, logp = evaluate(param_samples)

    accuracies.append(accuracy)
    eces.append(ece)
    logps.append(logp)

accuracies = np.array(accuracies)
eces = np.array(eces)
logps = np.array(logps)

SGLD_df = pd.DataFrame({"Accuracy": accuracies, "ECE": eces, "log predictive": logps})
with open("experiments/bayesian_nn_cifar10_sgld.txt", 'w') as f:
    f.write(str(SGLD_df.describe()))
print("FINISHED SGLD")


# SGD
def sgd(n_epochs, data_batch_size):
    dataloader_train = DataLoader(CIFAR10_train, shuffle=True, batch_size=data_batch_size, num_workers=2)
    model = LeNet5Cifar10(10).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)
    losses = []

    for i in range(n_epochs):
        for x, y in tqdm(iter(dataloader_train)):
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()

            out = model(x)

            l = F.cross_entropy(out, y, reduction="mean")

            l.backward()

            losses.append(l.detach().cpu().numpy())

            optimizer.step()

    losses = np.array(losses)

    return model, losses


accuracies, eces, logps = [], [], []

for i in range(num_exp):
    model, losses = sgd(n_epochs, data_batch_size)
    params = model.parameters()
    params = torch.cat([param.flatten() for param in params]).detach()
    params = params.view(1, -1)
    accuracy, ece, logp = evaluate(params)

    accuracies.append(accuracy)
    eces.append(ece)
    logps.append(logp)

accuracies = np.array(accuracies)
eces = np.array(eces)
logps = np.array(logps)

SGD_df = pd.DataFrame({"Accuracy": accuracies, "ECE": eces, "log predictive": logps})
with open("experiments/bayesian_nn_cifar10_sgd.txt", 'w') as f:
    f.write(str(SGD_df.describe()))
print("FINISHED SGD")

"""
