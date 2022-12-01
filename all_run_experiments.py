import argparse
from train_utils import OPTIM_METHOD
from fit_fncs import FITNESS_FNS
from models import make_policy
from proj_env import make_env
from stat_writer import Renderer

import os
import os.path as osp
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import json
from moviepy.video.io.ImageSequenceClip import ImageSequenceClip

EXP_DIR = "experiments"

def build_config(fnc_type, optim_method):
    config = {
        "exp_dir": None,
        "n_iters": 128,
        "fnc_type": fnc_type, # consult fit_fncs.FITNESS_FNC for more
        "optim_method": optim_method, # consult train_utils.OPTIM_METHOD for more
        "learning_rate": 0.008,#1/512, # 0.01,
        "noise_std": 0.1,
        "lr_grad" : 0.01,
        "noise_std_grad": 0.0001,  # step eval for finite differences                     
        "noise_decay": 0.99, # optional
        "lr_decay": 1.0, # optional
        "decay_step": 20, # optional
        "population_size": 32,
        "env_steps":3
    }

    return config

def train(config):
    policy = make_policy(config)
    env = make_env(config["fnc_type"])
    optim_method = OPTIM_METHOD[config["optim_method"]]

    vid_path = None
    recorder = Renderer(env, policy, vid_path, config)

    for i in tqdm(range(config["n_iters"])):
        log = optim_method(policy, env, config)
        recorder.update_log(log)
        recorder.capture_frame(log)
    
    return recorder



def init_exp_log():
    exp_log = {}
    for fnc_type in FITNESS_FNS.keys():
        fnc_dict = {}
        for optim_method in ["es", "fd", "es_grad"]:
            data_dict = {}
            for log_data in ["video", "rewards"]:
                data_dict[log_data] = None
            
            fnc_dict[optim_method] = data_dict
        exp_log[fnc_type] = fnc_dict

    return exp_log


def update_exp_log(exp_log, recorder, fnc_type, optim_method):
    exp_log[fnc_type][optim_method]["video"] = recorder.recorded_frames
    exp_log[fnc_type][optim_method]["rewards"] = recorder.rewards


def process_video(fnc_log, exp_dir):
    n_frames = len(fnc_log["fd"]["video"])
    full_vid = []

    for i_frame in range(n_frames):
        curr_new_frame = None
        for opt_methd in fnc_log.keys():
            if curr_new_frame is None:
                curr_new_frame = fnc_log[opt_methd]["video"][i_frame]
            else:
                curr_new_frame = np.concatenate([curr_new_frame, 
                                                 fnc_log[opt_methd]["video"][i_frame]], axis=1)
        full_vid.append(curr_new_frame)
    
    vid_path = osp.join(exp_dir, "train_traj.gif")
    clip = ImageSequenceClip(full_vid, fps = 10)
    clip.write_gif(vid_path)


def process_rewards(fnc_log, exp_dir):
    plt.clf()
    for opt_method in ["es", "es_grad", "fd"]:
        plt.plot(fnc_log[opt_method]["rewards"], label=opt_method)
    plt.xlabel("time step")
    plt.ylabel("fitness")
    plt.title("fitness achieved")
    plt.legend()
    plt.savefig(osp.join(exp_dir, "rewards.png"))


def process_fnc_type_exps(exp_log, fnc_type):
    save_dir = "presentation_exps"
    exp_dir = osp.join(save_dir, fnc_type)
    os.makedirs(exp_dir, exist_ok=True)

    process_video(exp_log[fnc_type], exp_dir)
    process_rewards(exp_log[fnc_type], exp_dir)
    

def train_all():  
    exp_log = init_exp_log()
    for fnc_type in ["narrowing_peaks_large"]: #tqdm(FITNESS_FNS.keys(), desc="full exp progress"):
        for optim_method in ["es", "fd", "es_grad"]:
            config = build_config(fnc_type, optim_method)
            recorder = train(config)
            update_exp_log(exp_log, recorder, fnc_type, optim_method)
        process_fnc_type_exps(exp_log, fnc_type)


if __name__ == "__main__":
    train_all()
    # config = build_config()
    # train(config)