import torch
import numpy as np
from plot import plot
import pandas as pd


def brownian_update(mu, sigma, dt, x, seed):
    
    np.random.seed(seed=seed)
    return np.random.normal(x+mu*dt, sigma^2*dt)

def unroll_brownian_process(config):

    data = np.zeros(config['length'])
    data[0]=config['initial_state']

    for i in range(config['length']-1):
        data[i+1]=brownian_update(config['mu'], config['sigma'], config['dt'], data[i], config['seed'])

    return torch.Tensor(data)


if __name__ == '__main__':

    experiment_config ={'length':1000, 
                        'dt': 0.01,
                        'mu':1, 
                        'sigma': 0.1,
                        'seed': np.random.seed(seed=10),
                        'initial_state': 0}
    
    plot_config = {'N':256, 
                   'length': experiment_config['length'], 
                   'filename': 'experiments/brownian_motion.pdf',
                   'experiment': 'brownian_motion'}
    
    test_data = unroll_brownian_process(experiment_config)
    plot(test_data, plot_config)
