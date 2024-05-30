import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import torch
from plot import plot

def solve_bernoulli_numerical(p_func, q_func, n, x0, xf, y0):
    """
    Solves the Bernoulli differential equation numerically using scipy's solve_ivp:
        y' + p(x) * y = q(x) * y**n

    Args:
    p_func (callable): The function p(x).
    q_func (callable): The function q(x).
    n (float): The exponent n in the equation.
    x0 (float): The starting point of the interval for x.
    xf (float): The end point of the interval for x.
    y0 (float): Initial condition y(x0) = y0.

    Returns:
    T (array): Array of x values.
    Y (array): Array of y values corresponding to T.
    """
    def bernoulli_ode(x, y):
        return -p_func(x) * y + q_func(x) * y**n
    
    # Solve the ODE
    sol = solve_ivp(bernoulli_ode, [x0, xf], [y0], t_eval=np.linspace(x0, xf, 10000))

    return sol.t, sol.y[0]

# Example functions and parameters
def p_func(x):
    return np.cos(5*x)  # Example p(x) = 1

def q_func(x):
    return np.sin(x)  # Example q(x) = x

n = 0.5  # Exponent n
x0, xf = 0, 10  # Interval from x0 to xf
y0 = 0.1  # Initial condition y(x0) = 0.1

# Solve the Bernoulli equation
T, Y = solve_bernoulli_numerical(p_func, q_func, n, x0, xf, y0)

test_data = torch.Tensor(Y)

plot_config = {'T': 10000,
                    'N': 17,
                    'batch_size': 1,
                    'experiment' : 'bernoulli',
                    'filename' : f'experiments/bernoulli.jpg'}

print(plot(test_data, plot_config, return_losses=True))