from torch.utils.data import Dataset
import torch
import numpy as np
import nengo
import random


def get_datasets(config: dict, test: bool):
    if test:
        if config["test"]["data"]["dataset"] == "PolyDataset":
            return PolyDataset(degree=config["test"]["data"]["degree"], num_points=config["test"]["data"]["num_points"], num_functions=config["test"]["data"]["num_functions"], device=config["device"], test=True)
        elif config["test"]["data"]["dataset"] == "WhiteSignalDataset":
            return WhiteSignalDataset(num_points=config["test"]["data"]["num_points"], num_functions=config["test"]["data"]["num_functions"], device=config["device"], test=True)
        elif config["test"]["data"]["dataset"] == "BrownianMotionDataset":
            return BrownianMotionDataset(num_points=config["test"]["data"]["num_points"], num_functions=config["test"]["data"]["num_functions"], mu=config["test"]["data"]["mu"], sigma=config["test"]["data"]["sigma"], dt=config["test"]["data"]["dt"], device=config["device"], test=True)
        else:
            raise ValueError("Unknown dataset")
    else:
        if config["train"]["data"]["dataset"] == "PolyDataset":
            return PolyDataset(degree=config["train"]["data"]["degree"], num_points=config["train"]["data"]["num_points"], num_functions=config["train"]["data"]["num_functions"], device=config["device"])
        elif config["train"]["data"]["dataset"] == "WhiteSignalDataset":
            return WhiteSignalDataset(num_points=config["train"]["data"]["num_points"], num_functions=config["train"]["data"]["num_functions"], device=config["device"])
        elif config["train"]["data"]["dataset"] == "BrownianMotionDataset":
            return BrownianMotionDataset(num_points=config["train"]["data"]["num_points"], num_functions=config["train"]["data"]["num_functions"], mu=config["train"]["data"]["mu"], sigma=config["train"]["data"]["sigma"], dt=config["train"]["data"]["dt"], device=config["device"])
        else:
            raise ValueError("Unknown dataset")
        

class PolyDataset(Dataset):
    def __init__(self, degree: int, num_points: int, num_functions: int, device: str = "cpu", test: bool = False):
        self.degree = degree # max degree of each sampled polynomial
        self.num_points = num_points # nr of points to sample from each polynomial as the context length
        self.num_functions = num_functions # nr of plolynomials to use, these will be random at each new step
        self.device = device
        self.x = torch.linspace(0, 1, num_points)
        self.test = test

    def __len__(self):
        return self.num_functions

    def __getitem__(self, idx: int ):
        # for beginning since there is a ton of polynomials to sample from we will just sample a random one
        # need to make sure that numerical evals stay within range of of float32
        # sample random polynomial of degree k
        # sample random points from the polynomial
        
        # sample k roots
        roots = torch.rand(self.degree) 

        # numpy get random polynomial
        y = self._polynomial(roots, self.x)
        return torch.tensor(y).reshape(-1).to(self.device)
    
    def _polynomial(self, roots, x):
        coeff = np.poly(roots)
        return np.polyval(coeff, x)


class WhiteSignalDataset(Dataset):
    def __init__(self, num_points: int, num_functions: int, device: str = "cpu", test: bool = False):
        self.TRAINSEED=1
        self.x = torch.linspace(0, 1, num_points)
        self.test = test
        self.device = device
        self.num_points = num_points
        self.num_functions = num_functions
        self.rng = np.random.RandomState(self.TRAINSEED)

    def __len__(self):
        return self.num_functions
    
    def __getitem__(self, idx: int):
        # sample random polynomial of degree k
        # sample random points from the polynomial
        # sample k roots

        high = random.uniform(30, 150)
        y0 = random.uniform(-1, 1)
        process = nengo.processes.WhiteSignal(0.1, high=high, y0=y0)

        y = process.run_steps(self.num_points, dt=0.1/self.num_points,rng=self.rng)
        self.TRAINSEED+=1 # TODO this is a dirty hack

<<<<<<< HEAD
        return torch.tensor(y).reshape(-1).to(self.device)
    

=======
        return torch.tensor(y).reshape(-1).to(torch.float32).to(self.device)
    


>>>>>>> b3269f7904861e4f0d3fb7f222dd9111fc24b222
class BrownianMotionDataset(Dataset):
    def __init__(self, num_points: int, num_functions: int, mu: float, sigma: float, dt: float, initial_state: float=0.0, device: str = "cpu", test: bool = False):
        self.num_points = num_points
        self.num_functions = num_functions
        self.device = device
        self.test = test
        self.mu = mu
        self.dt = dt
        self.sigma = sigma
        self.initial_state = initial_state
        self.x = torch.linspace(0, 1, num_points)

    def __len__(self):
        return self.num_functions
    
    def __getitem__(self, index):
        return self.unroll_brownian_process(index)

    def brownian_update(self, x):
        return np.random.normal(x+self.mu*self.dt, self.dt*self.sigma**2)

    def unroll_brownian_process(self, config):

        data = np.zeros(self.num_points)
        data[0]= self.initial_state

        for i in range(self.num_points-1):
            data[i+1]=self.brownian_update(data[i])

        return torch.Tensor(data)