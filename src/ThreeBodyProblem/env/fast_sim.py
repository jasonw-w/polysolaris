import torch
class fast_system_simulation:
    def __init__(self, masses, initial_position: torch.tensor, initial_velocity, device='cpu'):
        self.masses=masses.view(1, -1, 1) #(1, N, 1)
        self.pos=initial_position #(N, N)
        self.v=initial_velocity#(N, N)
        self.initial_position = initial_position.clone() 
        self.initial_velocity = initial_velocity.clone()
        self.pos = self.initial_position.clone()
        self.v = self.initial_velocity.clone()
        self.G = 1
    def step(self, dt):
        diff = self.pos.unsqueeze(0) - self.pos.unsqueeze(1) #(N, N, N)
        sq_diff = diff.pow(2)
        dist_sq = sq_diff.sum(dim=-1, keepdim=True) #(N, N, 1)
        dist = torch.sqrt(dist_sq + 1e-9)
        acceleration =  (self.G*self.masses*diff)/dist.pow(3)
        net_acc = acceleration.sum(dim=1)
        self.v += net_acc*dt
        self.pos += self.v*dt
        return self.pos, self.v
    def reset(self, new_pos=None, new_v=None):
        if new_pos is not None:
            self.pos = new_pos
            self.v = new_v
        else:
            self.pos = self.initial_position.clone()
            self.v = self.initial_velocity.clone()
        return self.pos, self.v


