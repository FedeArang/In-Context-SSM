from torch.utils.data import Dataset
import torch
import numpy as np
import nengo
import random


def get_datasets(config: dict, test: bool):
    if test:
        if config["test"]["dataset"] == "PolyDataset":
            return PolyDataset(degree=config["test"]["data"]["degree"], num_points=config["test"]["data"]["num_points"], num_functions=config["test"]["data"]["num_functions"], device=config["device"], test=True)
        elif config["test"]["dataset"] == "WhiteSignalDataset":
            return WhiteSignalDataset(num_points=config["test"]["data"]["num_points"], num_functions=config["test"]["data"]["num_functions"], device=config["device"], test=True)
        else:
            raise ValueError("Unknown dataset")
    else:
        if config["train"]["dataset"] == "PolyDataset":
            return PolyDataset(degree=config["train"]["data"]["degree"], num_points=config["train"]["data"]["num_points"], num_functions=config["train"]["data"]["num_functions"], device=config["device"])
        elif config["data"]["dataset"] == "WhiteSignalDataset":
            return WhiteSignalDataset(num_points=config["train"]["data"]["num_points"], num_functions=config["train"]["data"]["num_functions"], device=config["device"])
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
        return roots.reshape(-1).to(self.device), torch.tensor(y).reshape(-1).to(self.device)
    
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

        y = process.run_steps(self.num_points, rng=self.rng)
        self.TRAINSEED+=1 ## TODO this is a dirty hack

        return torch.tensor(y).reshape(-1).to(self.device)