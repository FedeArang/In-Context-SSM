import torch
from .data import PolyDataset, WhiteSignalDataset, get_datasets
from model.hippo import HiPPO_LegT
import yaml
import torch
from torch.utils.data import DataLoader
from .optimizer import select_optim
import wandb
from matplotlib import pyplot as plt
from .config import load_configs
import random
import numpy as np

def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.deterministic = True  # Ensure deterministic behavior for cuDNN
    torch.backends.cudnn.benchmark = False  # Disable the cuDNN auto-tuner to avoid non-deterministic behavior


def autoregressive(func):
    def wrapper(config, dataloader, model, *args, **kwargs):
        try:
            old_ratio = model.teacher_ratio
            model.teacher_ratio = config["test"]["teacher_ratio"]
            return func(config, dataloader, model, *args, **kwargs)
        finally:
            model.teacher_ratio = old_ratio
    return wrapper


def get_weight_dist(model: HiPPO_LegT):
    with torch.no_grad():
        model_test = HiPPO_LegT(N=model.N, dt=model.dt, trainable=False)
        C, D = model.C_discr, model.D_discr
        C_l, D_l = model_test.C_discr, model_test.D_discr

        # calculate the distance between the learned and the true weights in P=1,2,inf
        dist_1_C = torch.norm(C-C_l, p=1)
        dist_1_D = torch.norm(D-D_l, p=1)

        dist_2_C = torch.norm(C-C_l, p=2)
        dist_2_D = torch.norm(D-D_l, p=2)

        dist_inf_C = torch.norm(C-C_l, p=float('inf'))
        dist_inf_D = torch.norm(D-D_l, p=float('inf'))

        if model.full:
            A, B = model.A, model.B
            A_l, B_l = model_test.A, model_test.B

            dist_1_A = torch.norm(A-A_l, p=1)
            dist_1_B = torch.norm(B-B_l, p=1)

            dist_2_A = torch.norm(A-A_l, p=2)
            dist_2_B = torch.norm(B-B_l, p=2)

            dist_inf_A = torch.norm(A-A_l, p=float('inf'))
            dist_inf_B = torch.norm(B-B_l, p=float('inf'))
        
            return {"dist_1_C": dist_1_C, "dist_1_D": dist_1_D, "dist_2_C": dist_2_C, "dist_2_D": dist_2_D, "dist_inf_C": dist_inf_C, "dist_inf_D": dist_inf_D, "dist_1_A": dist_1_A, "dist_1_B": dist_1_B, "dist_2_A": dist_2_A, "dist_2_B": dist_2_B, "dist_inf_A": dist_inf_A, "dist_inf_B": dist_inf_B}
        else:
            return {"dist_1_C": dist_1_C, "dist_1_D": dist_1_D, "dist_2_C": dist_2_C, "dist_2_D": dist_2_D, "dist_inf_C": dist_inf_C, "dist_inf_D": dist_inf_D}
                                    
            


def save_checkpoint(config, model, epoch, opt, loss):
    # make dir time
    path = f"{config['save_dir']}/checkpoint_{epoch}.pt"
    torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': opt.state_dict(),
            'loss': loss,
            }, path)
    return path


def load_checkpoint(config, model, opt):
    checkpoint = torch.load(config["checkpoint_path"])
    model.load_state_dict(checkpoint['model_state_dict']) 
    opt.load_state_dict(checkpoint['optimizer_state_dict'])
    

@autoregressive
def test(config, dataloader, model, test=True):
    model.eval()
    x = dataloader.dataset.x
    with torch.no_grad():
        total_loss = 0
        for i, y in enumerate(dataloader):
            y_hat = model(y) # Now y is the signal 1,2,3,4,5,N+1, and y is 0,1,2,3,4,5..., N
            loss = torch.nn.MSELoss()(y_hat[:,:-1], y[:,1:])
            total_loss += loss.item()
            # make plots of the predictions and the ground truth and log them to wandb
            if i==0:
                for j in range(config["test"]["num_plots"]):
                    plt.figure()
                    plt.plot(x[1:], y_hat[j][:-1].numpy(), label="prediction")
                    plt.plot(x[1:], y[j][1:].numpy(), label="ground truth")
                    plt.legend()
                    plt.title(f"Function {i}")
                    wandb.log({f"function_{i}": plt})

        # evaluate weight distane of D,C to the learned ones
        weight_dist = get_weight_dist(model)
        wandb.log(weight_dist)
        if test:
            wandb.log({"test_loss": total_loss})
        else:
            wandb.log({"test_loss": total_loss})

    model.train()


def train(config):
    dataset = get_datasets(config=config, test=False)
    dataset_test = get_datasets(config=config, test=True)

    
    model = HiPPO_LegT(N=config["model"]["rank"], dt=1/config["train"]["data"]["num_points"], teacher_ratio=config["train"]["teacher_ratio"], trainable=True, full=config["model"]["full"])

    dataloader_train = DataLoader(dataset, batch_size=config["train"]["batch_size"], shuffle=True)
    dataloader_test = DataLoader(dataset_test, batch_size=config["train"]["batch_size"], shuffle=False)
    
    opt = select_optim(config, model)
    
    wandb.init(
        entity="incontextssm",
        project="incontextssm",
                config=config,
                name="HiPPO_LegT")
    wandb.watch(model)

    if config["load_from_checkpoint"]:
        load_checkpoint(config, model, opt)

    for epoch in range(config["train"]["epochs"]):
        epoch_loss = 0
        for i, y in enumerate(dataloader_train):
            opt.zero_grad()
            y_hat = model(y) # signal y is 0,1,2,3,4,5..., N / y_hat is 1,2,3,4,5,6..., N+1
            loss = torch.nn.MSELoss()(y_hat[:,:-1].flatten(), y[:,1:].flatten())
            loss.backward()
            opt.step()

            epoch_loss += loss.item()

        wandb.log({"loss": epoch_loss})

        if epoch % config["train"]["eval_every"] == 0:
            test(config, dataloader_test, model)
            test(config, dataloader_train, model, test=False)

        if epoch % config["train"]["save_every"] == 0:
            save_checkpoint(config, model, epoch, opt, epoch_loss)


if __name__=="__main__":
    for i in range(5):
        seed = 44 + i
        set_seeds(42)
        config = load_configs()
        config["seed"] = seed
        train(config)
    
            
        

