import pickle
import uuid

import numpy as np
from numpy.linalg import norm
import os
import os.path as osp

from tqdm import tqdm
from joblib import Parallel
from collections import defaultdict

from es import OpenAiES
from evaluation import eval_policy_delayed, eval_policy
import torch
from fd import FiniteDiff

# from https://github.com/Howuhh/evolution_strategies_openai/tree/8e9c369b5df94a4afeb6773f686fca1298a69285


def train_step_es(policy, env, config, n_jobs=1, verbose=True):
    es = OpenAiES(
        model=policy, 
        learning_rate=config["learning_rate"], 
        noise_std=config["noise_std"],
        noise_decay=config.get("noise_decay", 1.0),
        lr_decay=config.get("lr_decay", 1.0),
        decay_step=config.get("decay_step", 50)
    )
    
    log = {}
    population = es.generate_population(config["population_size"])

    rewards_jobs = (eval_policy_delayed(new_policy, env) for new_policy in population)
    rewards = np.array(Parallel(n_jobs=n_jobs)(rewards_jobs))   

    # for debug 
    # rewards = eval_policy(population[0], env, config["env_steps"])

    es.update_population(rewards)

    # populations stats
    log["pop_mean_rewards"] = np.mean(rewards)
    log["pop_std_rewards"] = np.std(rewards)
    log["rewards"] = eval_policy(policy, env)#rewards.max()
    log["pop"] = [pop.W[0] for pop in population]
        

    return log

def train_step_grad(policy, env, config, n_jobs=1, verbose=True):
    log = {}

    lr = config["learning_rate"]
    act = policy.predict()
    act = torch.from_numpy(act)
    act.requires_grad = True
    _, rew, _, _, _ = env.step(act, ret_numpy = False)

    rew.backward()
    grads = act.grad.numpy()

    new_w = policy.get_weights() + lr*grads
    policy.set_weights(new_w)

    log["rewards"] = rew.item()

    return log


def train_step_finite_diff(policy, env, config, n_jobs=1, verbose = True):
    fd = FiniteDiff(policy, 
                    config["lr_grad"], 
                    config["noise_std_grad"])

    log = {}
    population = fd.generate_population()

    rewards_jobs = (eval_policy_delayed(new_policy, env) for new_policy in population)
    rewards = np.array(Parallel(n_jobs=n_jobs)(rewards_jobs))   

    fd.update_model(rewards)
    log["rewards"] = eval_policy(policy, env)

    return log

def train_step_es_grad(policy, env, config, n_jobs=1, verbose=True):
    es = OpenAiES(
        model=policy, 
        learning_rate=config["learning_rate"], 
        noise_std=config["noise_std"],
        noise_decay=config.get("noise_decay", 1.0),
        lr_decay=config.get("lr_decay", 1.0),
        decay_step=config.get("decay_step", 50)
    )

    fd = FiniteDiff(policy, 
                    config["lr_grad"], 
                    config["noise_std_grad"])
    
    log = {}
    pop_es = es.generate_population(config["population_size"])
    rewards_jobs_es = (eval_policy_delayed(new_policy, env) for new_policy in pop_es)
    rewards_es = np.array(Parallel(n_jobs=n_jobs)(rewards_jobs_es))

    pop_fd = fd.generate_population()
    rewards_jobs_fd = (eval_policy_delayed(new_policy, env) for new_policy in pop_fd)
    rewards_fd = np.array(Parallel(n_jobs=n_jobs)(rewards_jobs_fd))

    es_grad = es.calculate_grad(rewards_es)
    fd_grad = fd.calculate_grad(rewards_fd)

    # sigmoid = lambda x : 1/(1+np.exp(-1000*(norm(x)-0.008)))
    sigmoid = lambda x : 1/(1+np.exp(-20*(norm(x)-0.2)))
    msk = sigmoid(fd_grad)
    # fd_norm = norm(fd_grad)
    # msk = 1/(1+np.exp(-fd_norm/(fd_norm + norm(es_grad))))

    grad_step = (1 - msk)*config["learning_rate"]*es_grad + msk*config["lr_grad"]*fd_grad
    new_weights = np.clip(policy.get_weights() + grad_step, -1,1)
    policy.set_weights(new_weights)

    log["msk_val"] = msk
    log["pop"] = [pop.W[0] for pop in pop_es]
    log["es_norm"] = norm(config["learning_rate"]*es_grad)
    log["fd_norm"] = norm(config["lr_grad"]*fd_grad)
    log["rewards"] = eval_policy(policy, env)

    return log
    


OPTIM_METHOD = {"es":train_step_es,
                "grad":train_step_grad,
                "fd":train_step_finite_diff,
                "es_grad": train_step_es_grad}