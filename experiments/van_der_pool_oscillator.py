import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from plot import plot
import torch

# Define the differential equation
def van_der_pol_oscillator(t, x, mu):
    return mu * (1 - x**2) * np.sin(t)

if __name__=='__main__':
    # Parameters
    n_points = 10000
    t_span = (0,20)

    experiment_config = {'mu':7.0,  # Nonlinearity parameter
                         'x0': [0.5],  # Initial condition
                         't_span' :  t_span,  # Time interval for the solution
                         'N_points' : n_points,
                         't_eval' : np.linspace(t_span[0], t_span[1], n_points)  # Time points where the solution is evaluated
    }

    plot_config = {'T': 10000,
                    'N': 17,
                    'batch_size': 1,
                    'experiment' : 'van_der_pool',
                    'filename' : f'experiments/van_der_pool.jpg'}

    # Solve the differential equation
    solution = solve_ivp(van_der_pol_oscillator, experiment_config['t_span'], experiment_config['x0'], args=(experiment_config['mu'],), 
                         t_eval=experiment_config['t_eval'], method='RK45')
    test_data = torch.Tensor((solution['y'])[0])

    print(plot(test_data, plot_config, return_losses=True))



