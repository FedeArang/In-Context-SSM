import torch
from .data import PolyDataset
from model.hippo import HiPPO_LegT
import yaml
import torch
from torch.utils.data import DataLoader
from .optimizer import select_optim
import wandb
from matplotlib import pyplot as plt


def test(config, dataloader, model):
    model.eval()

    x = dataloader.dataset.x

    with torch.no_grad():
        for i, (roots, y) in enumerate(dataloader):
            y_hat = model(roots)
            loss = torch.nn.MSELoss()(y_hat[:,1:], y[:,:-1])

            # make plots of the predictions and the ground truth and log them to wandb
            plt.figure()
            plt.plot(x, y_hat[0].numpy(), label="prediction")
            plt.plot(x, y[0].numpy(), label="ground truth")
            plt.legend()
            plt.title(f"Function {i}")
            wandb.log({f"function_{i}": plt})
        
    model.train()


def train(config):
    dataset = PolyDataset(degree=config["data"]["degree"], num_points=config["data"]["num_points"], num_functions=config["data"]["num_functions"])
    model = HiPPO_LegT(N=config["data"]["num_points"], dt=1/config["data"]["num_points"], trainable=True)

    dataloader_train = DataLoader(dataset, batch_size=config["train"]["batch_size"], shuffle=True)
    
    opt = select_optim(config, model)
    
    wandb.init(project="HiPPO")
    wandb.watch(model)

    model.to(config["device"])

    for epoch in range(config["train"]["epochs"]):
        epoch_loss = 0
        for i, (roots, y) in enumerate(dataloader_train):
            opt.zero_grad()
            y_hat = model(y)
            loss = torch.nn.MSELoss()(y_hat[:,1:], y[:,:-1])
            loss.backward()
            opt.step()

            epoch_loss += loss.item()

        wandb.log({"loss": epoch_loss})

        if epoch % config["train"]["eval_every"] == 0:
            test(config, dataloader_train, model)



if __name__=="__main__":
    # import config
    with open('icl_learning/config.yaml') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    train(config)
    
            
        

