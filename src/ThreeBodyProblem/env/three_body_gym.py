import gymnasium as gym
from env.fast_sim import fast_system_simulation
import numpy as np
import torch.nn.functional as F
import math
import torch
class ThreeBodyEnv(gym.Env):
    def __init__(self, masses, initial_pos, initial_v):
        self.simulation = fast_system_simulation(masses, initial_pos, initial_v)
        self.action_space = gym.spaces.Box(low=-100, high=100, shape=(9,), dtype=np.float64)
        self.observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(21,), dtype=np.float64)
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        import torch
        new_pos = torch.randn(3,3)
        self.simulation.pos = new_pos
        self.simulation.v = torch.randn(3,3) # Placeholder, will be overwritten by step
        
        pos = self.simulation.pos
        flat_pos = pos.flatten().cpu().numpy()
        flat_v = self.simulation.v.flatten().cpu().numpy()
        flat_m = self.simulation.masses.flatten().cpu().numpy()
        observation = np.concatenate([flat_pos, flat_v, flat_m])
        return observation, {}
    def step(self, action): 
        import torch
        new_velocity = torch.tensor(action,     dtype=torch.float32).view(3,3)
        self.simulation.v = new_velocity
        # Do not call reset here, it wipes the position!
        T = 10000
        dt = 1e-2
        reward = 0
        for t in range(T):
            pos, v = self.simulation.step(dt)
            dists = F.pdist(pos)
            min_dist = dists.min()
            max_dist = dists.max()
            # Overlap penalty
            # if dists.min() < 0.1:
            #      reward -= 5 * dt # Scaled by dt for consistency
            
            # if dists.max() > 7 and 15 >= dists.max():
            #     reward -= (torch.exp(0.01*(dists.max()-7)**2)-1)*0.01*dt
            # elif dists.max() > 15:
            #     reward -= (torch.exp(0.01*(dists.max()-7)**2)-1)*0.01*dt
            #     reward -= 1000
            #     break
            # else:
            #     reward += (math.exp(t/T) + 0.01*t)*dt
            #     if t % 1000 == 0:
            #         reward += 10
            if min_dist < 0.1:
                reward -= 1/T
            elif max_dist > 15:
                reward -= 5
                terminated = True
                break
            elif max_dist > 7.5:
                reward -= 30/T
            elif 1 < max_dist < 5:
                target = 3.0
                reward = 2*(torch.exp(torch.tensor(-0.5))*(max_dist-target)**2)/T
        flat_pos = self.simulation.pos.flatten().cpu().numpy()
        flat_v = self.simulation.v.flatten().cpu().numpy()
        flat_m = self.simulation.masses.flatten().cpu().numpy()
        observation = np.concatenate([flat_pos, flat_v, flat_m])
        terminated = True
        truncated = False
        info = {}
        return observation, reward, terminated, truncated, info