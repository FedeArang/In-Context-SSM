import torch

class Loss:
    def __init__(self, config):
        self.config = config

        if config['loss'] == 'MSE':
            self.loss_fn = torch.nn.MSELoss()
        elif config['loss'] == 'L1':
            self.loss_fn = torch.nn.L1Loss()
        else:
            raise ValueError(f"Loss {config['loss']} not supported")
        
        if config["regualization"] == "L1":
            self.reg_fn = torch.nn.L1Loss()
        elif config["regualization"] == "L2":
            self.reg_fn = torch.nn.MSELoss()
            
