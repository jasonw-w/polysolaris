import torch
import torch.optim as optim
import numpy as np
from ThreeBodyProblem.env.three_body_vectorized import VectorizedThreeBodyEnv
from ThreeBodyProblem.RL_agent.model import Actor_Critic
lr = 1e-5
gamma = 0
tau = 0.9
epochs = 10
batch_size = 256
clip_eps = 0.2
entropy_coef = 0.05
max_grad_norm = 0.5
num_envs = 2048
# steps_per_update = 128
steps_per_update = 4096
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"using device: {device}")
env = VectorizedThreeBodyEnv(
    num_envs=num_envs,
    device=device
)
num_input = 21
num_output = 9
agent = Actor_Critic(num_input, num_output).to(device)
optimizer = optim.AdamW(agent.parameters(), lr=lr)
for param_group in optimizer.param_groups:
    param_group['initial_lr'] = lr
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=75, gamma=0.9, last_epoch=150)
print("start training")
iterations_per_update = max(1, steps_per_update // num_envs)
score = 0
best_score = -float('inf')
import matplotlib.pyplot as plt

plt.ion() # Interactive mode
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True)
ax1.set_title("Training Loss")
line1, = ax1.plot([], [], color='red')
loss_plt = []
ax2.set_title("Training Score")
line2, = ax2.plot([], [], color='blue')
score_plt = []
ax3.set_title("Avg Stability Steps")
line3, = ax3.plot([], [], color='green')
steps_plt = []
epoch_idx = 0
state = env.reset()
while True:
    batch_log_probs = []
    batch_values = []
    batch_states = []
    batch_actions = []
    batch_rewards = []
    batch_dones = []
    batch_steps_list = []
    for step in range(iterations_per_update):
        with torch.no_grad():
            action_mean, value, log_std = agent(state)
            dist = torch.distributions.Normal(action_mean, log_std.exp())
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(axis=-1)

        next_state, step_rewards, dones, info = env.step(action)

        batch_log_probs.append(log_prob)
        batch_values.append(value.squeeze(-1))
        batch_states.append(state)
        batch_actions.append(action)
        batch_rewards.append(step_rewards)
        batch_dones.append(dones)
        if 'avg_steps' in info:
             batch_steps_list.append(info['avg_steps'])

        state = env.reset()
    with torch.no_grad():
        _, next_value, _ = agent(state) 

    batch_advantages = []
    gae = 0
    
    for step in reversed(range(iterations_per_update)):
        rewards = batch_rewards[step]
        dones = batch_dones[step] 
        values = batch_values[step]
        
        if step == iterations_per_update - 1:
            next_vals = next_value.squeeze(-1) 
        else:
            next_vals = batch_values[step + 1]
        
        mask = 1.0 - dones.float().squeeze() if len(dones.shape) > 1 else 1.0 - dones.float()
        
        delta = rewards + gamma * next_vals * mask - values
        gae = delta + gamma * tau * mask * gae
        batch_advantages.insert(0, gae)

    states_t = torch.cat(batch_states)
    actions_t = torch.cat(batch_actions)
    log_probs_t = torch.cat(batch_log_probs)
    values_t = torch.cat(batch_values)
    returns_t = torch.cat(batch_rewards)
    advantages_t = torch.cat(batch_advantages)
    returns_t = advantages_t + values_t
    advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)
    
    total_loss_val = 0
    total_actor_loss = 0
    total_critic_loss = 0
    count = 0
    count = 0
    dataset_size = states_t.size(0)
    indices = np.arange(dataset_size)
    batch_size = 256

    for _ in range(epochs):
        np.random.shuffle(indices)
        for start in range(0, dataset_size, batch_size):
            end = start + batch_size
            idx = indices[start:end]
            mb_states = states_t[idx]
            mb_actions = actions_t[idx]
            mb_old_log_probs = log_probs_t[idx]
            mb_advantages = advantages_t[idx]
            mb_returns = returns_t[idx]
            
            new_action_mean, new_value, new_log_std = agent(mb_states)
            new_value = new_value.squeeze(1)
            new_dist = torch.distributions.Normal(new_action_mean, new_log_std.exp())
            new_log_prob = new_dist.log_prob(mb_actions).sum(dim=-1)
            
            ratio = (new_log_prob - mb_old_log_probs).exp()
            
            surr1 = ratio * mb_advantages
            surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * mb_advantages
            actor_loss = -torch.min(surr1, surr2).mean()
            
            critic_loss = (mb_returns - new_value).pow(2).mean()
            entropy_loss = new_dist.entropy().mean()
            
            loss = actor_loss + critic_loss - entropy_coef * entropy_loss
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(agent.parameters(), max_grad_norm)
            optimizer.step()
            scheduler.step()
            
            total_loss_val += loss.item()
            total_actor_loss += actor_loss.item()
            total_critic_loss += critic_loss.item()
            count += 1
    rewards_t = torch.cat(batch_rewards)
    avg_score = rewards_t.mean().item()
    current_score = avg_score
    avg_loss = total_loss_val/count
    
    avg_steps = np.mean(batch_steps_list) if batch_steps_list else 0
    
    print(f"Update: {epoch_idx} | Score: {avg_score:.2f} | Loss: {total_loss_val/count:.4f} "
          f"(A: {total_actor_loss/count:.4f}, C: {total_critic_loss/count:.4f}) | Steps: {avg_steps:.1f}") 
    if current_score > best_score:
        best_score = current_score
        torch.save(agent.state_dict(), "best_agent.pth")
        print("saved best model")
    epoch_idx += 1

    loss_plt.append(total_loss_val/count)
    score_plt.append(current_score)
    steps_plt.append(avg_steps)
    
    line1.set_data(range(len(loss_plt)), loss_plt)
    ax1.relim()
    ax1.autoscale_view()
    line2.set_data(range(len(score_plt)), score_plt)
    ax2.relim()
    ax2.autoscale_view()
    line3.set_data(range(len(steps_plt)), steps_plt)
    ax3.relim()
    ax3.autoscale_view()
    fig.canvas.draw() 
    fig.canvas.flush_events()