from torch.utils.data import Dataset
import torch

class PolyDataset(Dataset):
    def __init__(self, degree: int, num_points: int, num_functions: int, num_samples: int = 1):
        self.degree = degree # max degree of each sampled polynomial
        self.num_points = num_points # nr of points to sample from each polynomial as the context length
        self.num_functions = num_functions # nr of plolynomials to use, these will be random at each new step
        self.num_samples = num_samples # nr of samples to generate for each function

    def __len__(self):
        return self.num_functions * self.num_points

    def __getitem__(self, idx):
        # for beginning since there is a ton of polynomials to sample from we will just sample a random one
        # need to make sure that numerical evals stay within range of of float32

        # sample random polynomial of degree k
        # sample random points from the polynomial

        # y = x_0^k + x_1^k + ... +  a_0 x_k^k

        coefficients = torch.randn(self.degree + 1, )

        
        