from torch.utils.data import Dataset
import torch
import numpy as np

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
