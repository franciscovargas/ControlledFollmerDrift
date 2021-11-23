import torch
import functorch
import math


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
