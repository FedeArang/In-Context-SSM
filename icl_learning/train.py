import torch
from .data import PolyDataset, WhiteSignalDataset, get_datasets
from model.hippo import HiPPO_LegT
from model.hippo import HiPPO_FouT
from model.hippo import HiPPO_LegS
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
    

        C, D = model.C_discr.clone().detach(), model.D_discr.clone().detach()
        C_l, D_l = model_test.C_discr, model_test.D_discr

        # calculate the distance between the learned and the true weights in P=1,2,inf
        dist_1_C = torch.norm(C-C_l, p=1)
        dist_1_D = torch.norm(D-D_l, p=1)

        dist_2_C = torch.norm(C-C_l, p=2)
        dist_2_D = torch.norm(D-D_l, p=2)

        dist_inf_C = torch.norm(C-C_l, p=float('inf'))
        dist_inf_D = torch.norm(D-D_l, p=float('inf'))
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
    model_test = HiPPO_LegT(N=model.N, dt=model.dt, trainable=False)

    with torch.no_grad():
        total_loss = 0
        for i, y in enumerate(dataloader):
            y_hat = model(y) # Now y is the signal 1,2,3,4,5,N+1, and y is 0,1,2,3,4,5..., N
            y_hat_exp = model_test(y)
            loss = torch.nn.L1Loss()(y_hat[:,5000:-1], y[:,5000+1:])
            total_loss += loss.item()
            # make plots of the predictions and the ground truth and log them to wandb
            if i==0:
                for j in range(config["test"]["num_plots"]):
                    plt.figure()
                    plt.plot(x[1:], y_hat[j][:-1].numpy(), label="prediction")
                    plt.plot(x[1:], y[j][1:].numpy(), label="ground truth")
                    plt.plot(x[1:], y_hat_exp[j][:-1].numpy(), label="explicit")
                    plt.legend()
                    plt.title(f"Function {i}")
                    wandb.log({f"function_{i}_{str(type(dataloader.dataset))}": plt}, commit=False)

            if not test:
                break

        # evaluate weight distane of D,C to the learned ones
        weight_dist = get_weight_dist(model)
        wandb.log(weight_dist, commit=False)
        if test:
            wandb.log({f"loss_{str(type(dataloader.dataset))}": total_loss}, commit=False)

    model.train()

def log_model(model,test=False):
    C_discr = model.C_discr
    D_discr = model.D_discr

    # now log this to wandb as a plot
    cax_c=plt.matshow(C_discr[None,:].detach().numpy())
    cax_c.get_figure().colorbar(cax_c)

    if test:
        wandb.log({"C_discr_test": wandb.Image(cax_c.get_figure())}, commit=False)
    else:
        wandb.log({"C_discr": wandb.Image(cax_c.get_figure())}, commit=False)

    cax_d=plt.matshow(D_discr[None,None,:].detach().numpy())
    cax_d.get_figure().colorbar(cax_d)

    if test:
        wandb.log({"D_discr_test": wandb.Image(cax_d.get_figure())}, commit=False)
    else:
        wandb.log({"D_discr": wandb.Image(cax_d.get_figure())}, commit=False)

    plt.close("all")



def train(config):
    dataset = get_datasets(config=config, test=False)
    dataset_tests = get_datasets(config=config, test=True)
    
    model = HiPPO_LegT(N=config["model"]["rank"], dt=1/config["train"]["data"]["num_points"], teacher_ratio=config["train"]["teacher_ratio"], trainable=True, init_opt=config["train"]["init_opt"], basis_learnable=config["train"]["basis_learnable"], init_opt_AB=config["train"]["init_opt_AB"])
    model_test = HiPPO_LegT(N=model.N, dt=model.dt, trainable=False)
    
    dataloaders_test = [DataLoader(dataset_test, batch_size=config["train"]["batch_size"], shuffle=False, num_workers=1) for dataset_test in dataset_tests]
    dataloader_train = DataLoader(dataset, batch_size=config["train"]["batch_size"], shuffle=True)
    
    opt = select_optim(config, model)
    
    wandb.init(
        entity="incontextssm",
        project="incontextssm",
                config=config,
                name=f"HiPPO_LegT_{config['seed']}")
    
    wandb.watch(model, log_freq=1, log="all")

    log_model(model_test, test=True)
    log_model(model)

    if config["load_from_checkpoint"]:
        load_checkpoint(config, model, opt)
    for epoch in range(config["train"]["epochs"]):
        epoch_loss = 0

        if epoch % config["train"]["eval_every"] == 0:
            log_model(model)
            for dataloader_test in dataloaders_test:
                test(config, dataloader_test, model)

        if epoch % config["train"]["save_every"] == 0:
            save_checkpoint(config, model, epoch, opt, epoch_loss)

        for i, y in enumerate(dataloader_train):
            opt.zero_grad()
            y_hat = model(y) # signal y is 0,1,2,3,4,5..., N / y_hat is 1,2,3,4,5,6..., N+1
            loss = torch.nn.L1Loss()(y_hat[:,5000:-1].flatten(), y[:,5000+1:].flatten())

            if epoch!=0:
                loss.backward()
                opt.step()

            epoch_loss += loss.item()

        wandb.log({"loss": epoch_loss},commit=True)




if __name__=="__main__":
    for i in range(5):
        seed = 42 + i
        set_seeds(seed)
        config = load_configs()
        config["seed"] = seed
        train(config)
    
            
        

