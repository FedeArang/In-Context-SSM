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
            y_hat = model(y)
            loss = torch.nn.MSELoss()(y_hat[:,1:], y[:,:-1])
            # make plots of the predictions and the ground truth and log them to wandb
            if i==0:
                for j in range(config["test"]["num_plots"]):
                    plt.figure()
                    plt.plot(x, y_hat[j].numpy(), label="prediction")
                    plt.plot(x, y[j].numpy(), label="ground truth")
                    plt.legend()
                    plt.title(f"Function {i}")
                    wandb.log({f"function_{i}": plt})
    model.train()


def train(config):
    dataset = PolyDataset(degree=config["data"]["degree"], num_points=config["data"]["num_points"], num_functions=config["data"]["num_functions"], device=config["device"])
    model = HiPPO_LegT(N=config["model"]["rank"], dt=1/config["data"]["num_points"], trainable=True)
    model.to(config["device"])

    dataloader_train = DataLoader(dataset, batch_size=config["train"]["batch_size"], shuffle=True)
    
    opt = select_optim(config, model)
    
    wandb.init(project="incontextssm",
                config=config,
                name="HiPPO_LegT")
    wandb.watch(model)

    

    for epoch in range(config["train"]["epochs"]):
        epoch_loss = 0
        for i, (roots, y) in enumerate(dataloader_train):
            opt.zero_grad()
            y_hat = model(y)
            loss = torch.nn.MSELoss()(y_hat[:,1:].flatten(), y[:,:-1].flatten())
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
    
            
        

