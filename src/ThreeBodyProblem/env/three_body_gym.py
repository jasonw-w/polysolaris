import gymnasium as gym
from env.fast_sim import fast_system_simulation
import numpy as np
import torch.nn.functional as F
class ThreeBodyEnv(gym.Env):
    def __init__(self, masses, initial_pos, initial_v):
        self.simulation = fast_system_simulation(masses, initial_pos, initial_v)
        self.action_space = gym.spaces.Box(low=-100, high=100, shape=(9,), dtype=np.float64)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(21,), dtype=np.float64)
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        pos, v = self.simulation.reset()
        flat_pos = pos.flatten().cpu().numpy()
        flat_v = pos.flatten().cpu().numpy()
        flat_m = self.simulation.masses.flatten().cpu().numpy()
        observation = np.concatenate([flat_pos, flat_v, flat_m])
        return observation, {}
    def step(self, action):
        import torch
        new_velocity = torch.tensor(action, dtype=torch.float32).view(3,3)
        _, _ = self.simulation.reset(new_v=new_velocity)
        T = int(1e4)
        dt = 1e-2
        reward = 0
        for _ in range(T):
            pos, v = self.simulation.step(dt)
            dists = F.pdist(pos) 
            if dists.max() > 20:
                reward -= 100
                break
            else:
                reward += 0.1
        flat_pos = self.simulation.pos.flatten().cpu().numpy()
        flat_v = self.simulation.v.flatten().cpu().numpy()
        flat_m = self.simulation.masses.flatten().cpu().numpy()
        observation = np.concatenate([flat_pos, flat_v, flat_m])
        terminated = True
        truncated = False
        info = {}
        return observation, reward, terminated, truncated, info