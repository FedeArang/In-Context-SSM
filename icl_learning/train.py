import torch
from .data import PolyDataset
from model.hippo import HiPPO_LegS
import yaml


if __name__=="__main__":
    # import config
    with open('icl_learning/config.yaml') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    dataset = PolyDataset(degree=["data"]["degree"], num_points=["data"]["num_points"], num_functions=["data"]["num_functions"])
    model = HiPPO_LegS(N=3, dt=1/config["data"]["num_points"], )
