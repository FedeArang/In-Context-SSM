import torch
import numpy as np
from plot import plot
import pandas as pd


def logistic_map(r, x):
    return r*x*(1-x)

def unroll_logistic_map(r, length, initial_state):

    data = np.zeros(length)
    data[0]=initial_state

    for i in range(length-1):
        data[i+1]=logistic_map(r, data[i])

    return torch.Tensor(data)


if __name__ == '__main__':

    N=256
    length = 1000
    r=3.9
    np.random.seed(seed=10)
    initial_state = np.random.uniform(low=0, high=1)
    test_data = unroll_logistic_map(r, length, initial_state)
    batch_size=1
    filename = 'experiments/logistic_map.pdf'
    experiment = 'logistic_map'
    plot(test_data, filename, length, N, batch_size, experiment)


