import numpy as np

import gymnasium as gym
from gymnasium import ObservationWrapper
from gymnasium.spaces import Box

import cv2
import vizdoom.gymnasium_wrapper  # Register Vizdoom environments

class WMWrapper(ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = Box(low=0.0, high=1.0, shape=(3, 64, 64), dtype=np.float32)
        
    def observation(self, obs):
        if isinstance(obs, dict):
            img = obs["screen"] # For doom
        else:
            img = obs
    
        resized = cv2.resize(img, (64, 64), interpolation=cv2.INTER_AREA)
        
        transpose = np.transpose(resized, (2, 0, 1))
        final_obs = transpose.astype(np.float32) / 255.0
        
        return final_obs
        
def make_env(env_name="VizdoomTakeCover-v1"):
    env = gym.make(env_name)
    env = WMWrapper(env)
    return env
