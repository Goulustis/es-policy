import argparse
from train_utils import OPTIM_METHOD
from models import make_policy
from proj_env import make_env
from stat_writer import Renderer

import os
import os.path as osp
from tqdm import tqdm
import json

EXP_DIR = "experiments"

def build_config(args=None):
    config = {
        "exp_dir": osp.join(EXP_DIR,"dev"),
        "n_iters": 128,
        "fnc_type": "narrowing_peaks", # consult fit_fncs.FITNESS_FNC for more
        "optim_method": "es", # consult train_utils.OPTIM_METHOD for more
        "learning_rate": 0.008,#1/512, # 0.01,
        "noise_std": 0.1,
        "lr_grad" : 0.008,
        "noise_std_grad": 0.0001,  # step eval for finite differences                     
        "noise_decay": 0.99, # optional
        "lr_decay": 1.0, # optional
        "decay_step": 20, # optional
        "population_size": 32,
        "env_steps":3
    }
    os.makedirs(config["exp_dir"], exist_ok=True)
    with open(osp.join(config["exp_dir"], "config.json"), "w") as f:
        json.dump(config, f, indent=4)

    return config

def train(config):
    policy = make_policy(config)
    os.makedirs(config["exp_dir"], exist_ok=True)
    env = make_env(config["fnc_type"])
    optim_method = OPTIM_METHOD[config["optim_method"]]

    vid_path = osp.join(config["exp_dir"], "train_traj.mp4")
    recorder = Renderer(env, policy, vid_path, config)

    for i in tqdm(range(config["n_iters"])):
        log = optim_method(policy, env, config)
        recorder.update_log(log)
        recorder.capture_frame(log)
    
    recorder.close()


if __name__ == "__main__":
    config = build_config()
    train(config)