import torch
import numpy as np
import torch.utils.data as data
import nengo
from .plot import plot

class FunctionApprox(data.TensorDataset):

    def __init__(self, config):
        
        length = config['T']
        nbatches = config['nbatches']
        dt = config['dt']

        rng = np.random.RandomState(seed=config['seed'])
        process = nengo.processes.WhiteSignal(length * dt, high=config['freq'], y0=0)
        X = np.empty((nbatches, length, 1))
        for i in range(nbatches):
            X[i, :] = process.run_steps(length, dt=dt, rng=rng)
            # X[i, :] /= np.max(np.abs(X[i, :]))
        X = torch.Tensor(X)
        super().__init__(X, X)

if  __name__ == '__main__':

    plot_config = {'T': 10000,
                   'N': 256,
                   'batch_size': 1,
                   'experiment' : 'function_approximation',
                   'filename' : 'experiments/function_approx_whitenoise.pdf'}
    
    experiments_config = {'dt' : 1e-3,
                          'T': plot_config['T'],
                          'nbatches': 10,
                          'freq' : 1.0,
                          'seed': 1}
    dt = 1e-3
    nbatches = 10
    #train = FunctionApprox(T, dt, nbatches, freq=1.0, seed=0)
    test_data = FunctionApprox(experiments_config)
    
    plot(test_data, plot_config)