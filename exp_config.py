import os.path as osp
from single_run_experiments import EXP_DIR


## grad narrow path
config = {
        "exp_dir": osp.join(EXP_DIR,"dev"),
        "n_iters": 128,
        "fnc_type": "narrowing_path", # consult fit_fncs.FITNESS_FNC for more
        "optim_method": "grad", # consult train_utils.OPTIM_METHOD for more
        "learning_rate": 0.0001, #0.01,
        "noise_std": 0.1,            
        "noise_decay": 0.99, # optional
        "lr_decay": 1.0, # optional
        "decay_step": 20, # optional
        "population_size": 32,
        "env_steps":3
    }

## es narrow path
config = {
        "exp_dir": osp.join(EXP_DIR,"dev"),
        "n_iters": 128,
        "fnc_type": "narrowing_path", # consult fit_fncs.FITNESS_FNC for more
        "optim_method": "es", # consult train_utils.OPTIM_METHOD for more
        "learning_rate": 0.0001, #0.01,
        "noise_std": 0.05,            
        "noise_decay": 0.99, # optional
        "lr_decay": 1.0, # optional
        "decay_step": 20, # optional
        "population_size": 32,
        "env_steps":3
    }