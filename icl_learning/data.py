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
        elif config["test"]["data"]["dataset"] == "SineDataset":
            return SineDataset(num_points=config["test"]["data"]["num_points"], num_functions=config["test"]["data"]["num_functions"], device=config["device"], test=True)
        elif config["test"]["data"]["dataset"] == "MultipleSineDataset":
            return MultipleSineDataset(num_points=config["test"]["data"]["num_points"], num_functions=config["test"]["data"]["num_functions"], max_frequency=config["test"]["data"]["max_frequency"], num_summands=config["test"]["data"]["num_summands"], device=config["device"], test=True)
        elif config["test"]["data"]["dataset"] == "LinearDataset":
            return LinearDataset(num_points=config["test"]["data"]["num_points"], num_functions=config["test"]["data"]["num_functions"], device=config["device"], test=True)
        elif config["test"]["data"]["dataset"] == "LegendreDataset":
            return LegendreDataset(degree=config["test"]["data"]["degree"], num_points=config["test"]["data"]["num_points"], num_functions=config["test"]["data"]["num_functions"], device=config["device"], test=True)
        elif config["test"]["data"]["dataset"] == "MixedDataset":
            return MixedDataset(num_points=config["test"]["data"]["num_points"], num_functions=config["test"]["data"]["num_functions"], config=config, device=config["device"], test=True)
        else:
            raise ValueError("Unknown dataset")
    else:
        if config["train"]["data"]["dataset"] == "PolyDataset":
            return PolyDataset(degree=config["train"]["data"]["degree"], num_points=config["train"]["data"]["num_points"], num_functions=config["train"]["data"]["num_functions"], device=config["device"])
        elif config["train"]["data"]["dataset"] == "WhiteSignalDataset":
            return WhiteSignalDataset(num_points=config["train"]["data"]["num_points"], num_functions=config["train"]["data"]["num_functions"], device=config["device"])
        elif config["train"]["data"]["dataset"] == "BrownianMotionDataset":
            return BrownianMotionDataset(num_points=config["train"]["data"]["num_points"], num_functions=config["train"]["data"]["num_functions"], mu=config["train"]["data"]["mu"], sigma=config["train"]["data"]["sigma"], dt=config["train"]["data"]["dt"], device=config["device"])
        elif config["train"]["data"]["dataset"] == "SineDataset":
            return SineDataset(num_points=config["train"]["data"]["num_points"], num_functions=config["train"]["data"]["num_functions"], device=config["device"])
        elif config["train"]["data"]["dataset"] == "MultipleSineDataset":
            return MultipleSineDataset(num_points=config["train"]["data"]["num_points"], num_functions=config["train"]["data"]["num_functions"], max_frequency=config["train"]["data"]["max_frequency"], num_summands=config["train"]["data"]["num_summands"], device=config["device"])
        elif config["train"]["data"]["dataset"] == "LinearDataset":
            return LinearDataset(num_points=config["train"]["data"]["num_points"], num_functions=config["train"]["data"]["num_functions"], device=config["device"])
        elif config["train"]["data"]["dataset"] == "LegendreDataset":
            return LegendreDataset(degree=config["train"]["data"]["degree"], num_points=config["train"]["data"]["num_points"], num_functions=config["train"]["data"]["num_functions"], device=config["device"])
        elif config["train"]["data"]["dataset"] == "MixedDataset":
            return MixedDataset(num_points=config["train"]["data"]["num_points"], num_functions=config["train"]["data"]["num_functions"], config=config, device=config["device"])
        else:
            raise ValueError("Unknown dataset")
        

class PolyDataset(Dataset):
    def __init__(self, degree: int, num_points: int, num_functions: int, device: str = "cpu", test: bool = False, **kwargs):
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
    def __init__(self, num_points: int, num_functions: int, device: str = "cpu", test: bool = False, **kwargs):
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

        return torch.tensor(y).reshape(-1).to(torch.float32).to(self.device)
    


class BrownianMotionDataset(Dataset):
    def __init__(self, num_points: int, num_functions: int, mu: float, sigma: float, dt: float, initial_state: float=0.0, device: str = "cpu", test: bool = False, **kwargs):
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
    

class SineDataset(Dataset):
    def __init__(self, num_points: int, num_functions: int, device: str = "cpu", test: bool = False, **kwargs):
        self.num_points = num_points
        self.num_functions = num_functions
        self.device = device
        self.test = test
        self.x = torch.linspace(0, 1, num_points)
        self.frequencies = np.random.uniform(0.1, 200, num_functions) 

    def __len__(self):
        return self.num_functions
    
    def __getitem__(self, index):
        sine = np.sin(self.frequencies[index]*self.x) 
        return torch.tensor(sine).to(torch.float32).to(self.device)
    

class MultipleSineDataset(Dataset):
    def __init__(self, num_points: int, num_functions: int, num_summands: int, max_frequency: int, device: str = "cpu", test: bool = False, **kwargs):
        self.num_points = num_points
        self.num_functions = num_functions
        self.num_summands = num_summands #it says how many different sines we are adding up to create one function
        self.device = device
        self.test = test
        self.x = torch.linspace(0, 1, num_points)
        self.max_frequency = max_frequency
        self.frequencies = np.random.uniform(low = 0.1, high = self.max_frequency, size = (self.num_summands, self.num_functions))
        self.coefficients = np.random.uniform(low = -1, high = 1, size = (self.num_summands, self.num_functions))

    def __len__(self):
        return self.num_functions
    
    def __getitem__(self, index):
        frequencies = self.frequencies[:, index]
        coefficients = self.coefficients[:, index]

        def sine(frequency, coefficient):
            return coefficient*np.sin(frequency*self.x)
        
        sines = np.sum(np.array(list(map(sine, frequencies, coefficients))), axis=0)
        sines = sines/max(sines)
        return torch.tensor(sines).to(torch.float32).to(self.device)
        

        


class LinearDataset(Dataset):
    def __init__(self, num_points: int, num_functions: int, device: str = "cpu", test: bool = False, **kwargs):
        self.num_points = num_points
        self.num_functions = num_functions
        self.device = device
        self.test = test
        self.x = torch.linspace(0, 1, num_points)
        

    def __len__(self):
        return self.num_functions
    
    def __getitem__(self, index):
        # sample an offset and a slope
        offset = np.random.uniform(-1,1) * 1000
        slope = np.random.uniform(-1,1) * 1000
        y = offset + slope*self.x
        return torch.tensor(y).to(torch.float32).to(self.device)
        

class LegendreDataset(Dataset):
    def __init__(self, degree: int, num_points: int, num_functions: int, device: str = "cpu", test: bool = False,**kwargs):
        self.degree = degree # max degree of each sampled polynomial
        self.num_points = num_points # nr of points to sample from each polynomial as the context length
        self.num_functions = num_functions # nr of plolynomials to use, these will be random at each new step
        self.device = device
        self.x = torch.linspace(0, 1, num_points)
        self.test = test

    def __len__(self):
        return self.num_functions
    
    def __getitem__(self, index):
        # sample random coefficients for the legendre polynomial
        rands = np.random.uniform(-1,1,self.degree) * 1000
        unnormalized = self._legendre(rands, self.x)
        max = torch.max(np.abs(unnormalized))
        y = unnormalized / max
        return torch.tensor(y).to(torch.float32).to(self.device)

    def _legendre(self, coeff, x):
        return np.polynomial.legendre.legval(x, coeff)
        

class MixedDataset(Dataset):
    def __init__(self, num_points: int, num_functions: int, config: dict, device: str = "cpu", test: bool = False, **kwargs):
        self.num_points = num_points
        self.num_functions = num_functions
        self.device = device
        self.test = test
        self.x = torch.linspace(0, 1, num_points)
        
        if test:
            dataconfig = config["test"]["data"] 
        else:
            dataconfig = config["train"]["data"]

        self.sine_ds = MultipleSineDataset(**dataconfig, device=device, test=test)
        self.lin_ds = LinearDataset(**dataconfig, device=device, test=test)
        self.white_ds = WhiteSignalDataset(**dataconfig, device=device, test=test)
        self.legend_ds = LegendreDataset(**dataconfig, device=device, test=test)

        self.dist = [0.30, 0.05, 0.10, 0.45]

    def __len__(self):
        return self.num_functions
    
    def __getitem__(self, index):
        # sample from dist
        r = random.random()

        if r < self.dist[0]:
            return self.sine_ds[index]
        elif r < self.dist[0] + self.dist[1]:
            return self.lin_ds[index]
        elif r < self.dist[0] + self.dist[1] + self.dist[2]:
            return self.white_ds[index]
        else:
            return self.legend_ds[index]