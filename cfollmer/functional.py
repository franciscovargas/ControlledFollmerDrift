import torch
import functorch
import math
from sampler_utils import LeNet5MNIST


def params_to_size_tuples(params):
    return [p.shape for p in params]


def get_number_of_params(size_list):
    return sum([math.prod(s) for s in size_list])


def get_params_from_array(array, size_list):
    cur_index = 0
    param_list = []
    for s in size_list:
        step_number = math.prod(s)
        param_list.append(array[cur_index:cur_index+step_number].reshape(s))
        cur_index += step_number
    return param_list


def example():
    model = LeNet5MNIST()
    x = torch.randn((50, 1, 28, 28))

    func_model, params = functorch.make_functional(model)
    s_list = params_to_size_tuples(params)
    n_params = get_number_of_params(s_list)

    theta = torch.randn((n_params,))

    new_params = get_params_from_array(theta, s_list)

    print(func_model(params, x).shape)
    print(func_model(new_params, x).shape)
