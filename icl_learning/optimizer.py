import torch

def select_optim(config, model: torch.nn.Module):
    if config["optimizer"]["type"] == "adam":
        return torch.optim.Adam(lr=config["optimizer"]["lr"], params=model.parameters())
    elif config["optimizer"]["type"] == "sgd":
        return torch.optim.SGD(lr=config["optimizer"]["lr"], params=model.parameters())
    elif config["optimizer"]["type"] == "sgd_momentum":
        return torch.optim.SGD(lr=config["optimizer"]["lr"], momentum=0, params=model.parameters())
    elif config["optimizer"]["type"] == "adamw":
        return torch.optim.AdamW(lr=config["optimizer"]["lr"], params=model.parameters())
    else:
        raise ValueError("Optimizer not supported")