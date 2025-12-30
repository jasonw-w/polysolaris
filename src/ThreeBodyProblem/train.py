import torch
import torch.optim as optim
import numpy as np
from env.three_body_gym import ThreeBodyEnv
from RL_agent.model import Actor_Critic
lr = 1e-4
gamma = 0.99
tau = 0.95
epochs = 10
batch_size = 64
clip_eps = 0.1
steps_per_update = 128
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"using device: {device}")
env = ThreeBodyEnv(
    masses=torch.tensor([1, 1, 1], dtype=torch.float32),
    initial_pos=torch.randn(3,3),
    initial_v=torch.randn(3,3)
)
num_input = env.observation_space.shape[0]
num_output = env.action_space.shape[0]
agent = Actor_Critic(num_input, num_output).to(device)
optimizer = optim.AdamW(agent.parameters(), lr=lr)
print("start training")

def compute_gae(next_value, rewards, masks, values, gamma=0.99, tau=0.95):
    values = values + [next_value]
    gae = 0
    returns = []
    for step in reversed(range(len(rewards))):
        delta = rewards[step] + gamma * values[step + 1] * masks[step] - values[step]
        gae = delta + gamma * tau * masks[step] * gae
        returns.insert(0, gae + values[step])
    return returns
state, _ = env.reset()
score = 0
epoch_idx = 0
best_score = -float('inf')
import matplotlib.pyplot as plt

plt.ion() # Interactive mode
fig, ax = plt.subplots()
line, = ax.plot([], [])
ax.set_title("Training Loss")
loss_plt = []
while True:
    log_probs = []
    values = []
    states = []
    actions = []
    rewards = []
    masks = []
    for step in range(steps_per_update):
        state_t = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            action_mean, value, log_std = agent(state_t)
            dist = torch.distributions.Normal(action_mean, log_std.exp())
            action = dist.sample()
            log_prob = dist.log_prob(action).sum(axis=-1)
        next_state, reward, terminated, truncated, _ = env.step(action.cpu().numpy()[0])
        done = terminated or truncated
        log_probs.append(log_prob)
        values.append(value.item())
        states.append(state_t)
        actions.append(action)
        rewards.append(reward)
        masks.append(1-done)
        state = next_state
        if done:
            state, _ = env.reset()
        if step % 32 == 0:
            print(f"step: {step}")

    next_state_t = torch.FloatTensor(next_state).unsqueeze(0).to(device)
    with torch.no_grad():
        _, next_value, _ = agent(next_state_t)
        next_value = next_value.item()
        returns = compute_gae(next_value, rewards, masks, values, gamma, tau)
        returns = torch.tensor(returns).float().to(device)
        values = torch.tensor(values).float().to(device)
        log_probs = torch.cat(log_probs).detach()
        states = torch.cat(states)
        actions = torch.cat(actions)
        advantages = returns - values
    # Prepare for logging
    total_actor_loss = 0
    total_critic_loss = 0
    total_loss_val = 0
    count = 0

    for _ in range(epochs):
        sampler = np.random.permutation(len(states))
        for i in range(0, len(states), batch_size):
            indices = sampler[i:i+batch_size]
            batch_states = states[indices]
            batch_actions = actions[indices]
            batch_log_probs = log_probs[indices]
            batch_returns = returns[indices]
            batch_advantages = advantages[indices] # Fix: Define batch_advantages
            
            new_action_mean, new_value, new_log_std = agent(batch_states)
            new_value = new_value.squeeze(1)
            new_dist = torch.distributions.Normal(new_action_mean, new_log_std.exp())
            new_log_prob = new_dist.log_prob(batch_actions).sum(axis=-1)

            ratio = (new_log_prob-batch_log_probs).exp()

            surr1 = ratio * batch_advantages
            surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * batch_advantages
            actor_loss = -torch.min(surr1, surr2).mean()

            critic_loss = (batch_returns-new_value).pow(2).mean()

            loss = 0.5*critic_loss + actor_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Accumulate logs
            total_actor_loss += actor_loss.item()
            total_critic_loss += critic_loss.item()
            total_loss_val += loss.item()
            count += 1
    current_score = np.mean(rewards)
    if current_score > best_score:
        torch.save(agent.state_dict(), "best_agent.pth")
        print("saved best model")
    print(f"Epoch: {epoch_idx} | Score: {np.mean(rewards):.2f} | Loss: {total_loss_val/count:.4f} (A: {total_actor_loss/count:.4f}, C: {total_critic_loss/count:.4f})")
    epoch_idx += 1
    loss_plt.append(total_loss_val/count)
    
    # Update plot
    line.set_data(range(len(loss_plt)), loss_plt)
    ax.relim()
    ax.autoscale_view()
    plt.draw()
    plt.pause(0.1)