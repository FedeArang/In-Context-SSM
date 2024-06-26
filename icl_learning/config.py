import yaml
import os
import time

def load_configs():
    with open("icl_learning/config.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    config["save_dir"] = config["save_dir"] + time.strftime("/%Y_%m_%d_%H_%M_%S")
    # make this directory
    os.makedirs(config["save_dir"])
    return config