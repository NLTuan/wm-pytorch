import os
import numpy as np
from env import make_env

DATA_DIR = 'data'
NUM_EPS = 1000
ENV_NAME = "VizdoomTakeCover-v1"

def collect_random_rollouts():
    print(f"Starting data collection for {NUM_EPS} episodes in '{ENV_NAME}'...")
    os.makedirs(DATA_DIR, exist_ok=True)
    env = make_env(ENV_NAME)
    
    total_steps = 0
    for ep in range(NUM_EPS):
        obs, info = env.reset()
        obs_list = [(obs * 255.0).astype(np.uint8)]
        action_list = []
        reward_list = []
        done_list = []
        
        done = False
        while not done:
            action = env.action_space.sample()
            obs_next, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            obs_list.append((obs_next * 255.0).astype(np.uint8))
            action_list.append(action)
            reward_list.append(reward)
            done_list.append(done)
            total_steps += 1
            
        np.savez_compressed(
            os.path.join(DATA_DIR, f'rollout_{ep}.npz'),
            obs=np.array(obs_list),
            action=np.array(action_list),
            reward=np.array(reward_list),
            done=np.array(done_list)
        )
        
        if (ep + 1) % 100 == 0:
            print(f"Collected {ep + 1}/{NUM_EPS} episodes (Total steps: {total_steps})...")
            
    env.close()
    print("Data collection completed successfully!")

if __name__ == "__main__":
    collect_random_rollouts()