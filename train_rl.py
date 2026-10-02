import os
import numpy as np
import torch
from tqdm import tqdm
from sim.environment import SchedulingEnv
from schedulers.rl_dqn import DQNAgent

def evaluate(env, agent, episodes=2):
    total = 0.0
    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        ep_r = 0.0
        while not done:
            with torch.no_grad():
                action = agent.act(obs)  # epsilon greedy ama eps_end'e yakınsa neredeyse greedy
            obs, r, term, trunc, _ = env.step(action)
            done = term or trunc
            ep_r += r
        total += ep_r
    return total / episodes

def train_dqn(episodes=400, users_n=6, slots=400, total_rb=12, lam=0.35, seed=7, save_path="models/dqn.pt"):
    env = SchedulingEnv(users_n=users_n, slots=slots, total_rb=total_rb, lam=lam, seed=seed)
    obs, _ = env.reset()
    state_dim = obs.shape[0]
    action_dim = env.action_space.n

    agent = DQNAgent(state_dim, action_dim,
                     lr=5e-4, buffer_size=50000, batch_size=256,
                     eps_start=1.0, eps_end=0.05, eps_decay=0.996)

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    WARMUP = 3000  # replay’e yeter veri dolmadan optimize etme
    best_avg_reward = -1e9

    for ep in tqdm(range(episodes), desc="Training"):
        obs, _ = env.reset()
        done = False
        ep_reward = 0.0

        while not done:
            action = agent.act(obs)
            nobs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            agent.push(obs, action, nobs, reward, float(done))
            if len(agent.replay) > WARMUP:
                agent.train_step()

            obs = nobs
            ep_reward += reward

        # periyodik değerlendirme + en iyiyi kaydet
        if ep % 10 == 0 and len(agent.replay) > WARMUP:
            old_eps = agent.epsilon
            agent.epsilon = agent.eps_end
            eval_rew = evaluate(env, agent, episodes=2)
            agent.epsilon = old_eps
            if eval_rew > best_avg_reward:
                best_avg_reward = eval_rew
                torch.save(agent.q.state_dict(), save_path)

    # son kaydetme
    torch.save(agent.q.state_dict(), save_path)
    print(f"Model saved to {save_path}")

if __name__ == "__main__":
    train_dqn(
        episodes=400,      # istersen 300-600 arası kullanabilirsin
        users_n=6,
        slots=400,
        total_rb=12,
        lam=0.35,
        seed=7,
        save_path="models/dqn.pt"
    )
