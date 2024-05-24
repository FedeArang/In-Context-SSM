import torch
import numpy as np
from plot import plot
import pandas as pd


def geometric_brownian_update(mu, sigma, dt, x):
    
    return np.random.normal(x*(1+mu*dt), dt*(sigma*x)**2)

def unroll_brownian_process(config):

    data = np.zeros(config['length'])
    data[0]=config['initial_state']

    for i in range(config['length']-1):
        data[i+1]=geometric_brownian_update(config['mu'], config['sigma'], config['dt'], data[i])

    return torch.Tensor(data)


if __name__ == '__main__':

    np.random.seed(seed=10)

    experiment_config ={'length':500, 
                        'dt': 0.01,
                        'mu':1, 
                        'sigma': 0.2,
                        'initial_state': np.random.uniform(low=-1, high=1)}
    
    plot_config = {'N': 256, 
                   'T': experiment_config['length'], 
                   'filename': 'experiments/geometric_brownian_motion.pdf',
                   'experiment': 'geometric_brownian_motion'}
    
    test_data = unroll_brownian_process(experiment_config)
    plot(test_data, plot_config)
