import numpy as np

from joblib import delayed

# from https://github.com/Howuhh/evolution_strategies_openai/tree/8e9c369b5df94a4afeb6773f686fca1298a69285


def eval_policy(policy, env, n_steps=1):
    
    total_reward = 0
    
    obs, _ = env.reset()
    for i in range(n_steps):
        action = policy.predict(obs)

        new_obs, reward, done, _,  _ = env.step(action)
        
        total_reward = total_reward + reward
        obs = new_obs

        if done:
            break

    return total_reward


# for parallel
eval_policy_delayed = delayed(eval_policy)