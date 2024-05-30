import torch
import numpy as np
import torch.utils.data as data
import nengo
from plot import plot

class FunctionApprox(data.TensorDataset):

    def __init__(self, config):
        
        length = config['T']
        nbatches = config['nbatches']
        dt = config['dt']

        rng = np.random.RandomState(seed=config['seed'])
        if config['signal']=='WhiteSignal':
            process = nengo.processes.WhiteSignal(length * dt, high=config['freq'], y0=0)
        elif config['signal']=='FilteredNoise':
            process =  nengo.processes.FilteredNoise(synapse=nengo.synapses.Alpha(config['alpha']), seed = config['seed'])
        else:
            raise ValueError('Signal type not supported')
        X = np.empty((nbatches, length, 1))
        for i in range(nbatches):
            X[i, :] = process.run_steps(length, dt=dt, rng=rng)
            # X[i, :] /= np.max(np.abs(X[i, :]))
        X = torch.Tensor(X)
        super().__init__(X, X)

if  __name__ == '__main__':

    N_RUNS = 100
    legt_loss=np.zeros(N_RUNS)
    legs_loss=np.zeros(N_RUNS)
    fout_loss=np.zeros(N_RUNS)
    
    for i in range(N_RUNS):
        
        plot_config = {'T': 10000,
                    'N': 65,
                    'batch_size': 1,
                    'experiment' : 'function_approximation',
                    'filename' : f'experiments/function_approx_filterednoise_{i}.jpg'}
        
        experiments_config = {'dt' : 1e-3,
                            'T': plot_config['T'],
                            'nbatches': 10,
                            'freq' : 1.0,
                            'alpha': 0.1, 
                            'signal': 'FilteredNoise',
                            'seed': i}
        dt = 1e-3
        nbatches = 10
        #train = FunctionApprox(T, dt, nbatches, freq=1.0, seed=0)
        test_data = FunctionApprox(experiments_config)
        
        legt_loss[i], legs_loss[i], fout_loss[i] = plot(test_data, plot_config, return_losses=True)
    
    #print(legt_loss, legs_loss, fout_loss)
    print(f'loss means: {np.mean(legt_loss), np.mean(legs_loss), np.mean(fout_loss)}')
    print(f'loss std: {np.std(legt_loss), np.std(legs_loss), np.std(fout_loss)}')