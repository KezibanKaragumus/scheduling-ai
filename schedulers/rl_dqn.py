import random
from collections import deque, namedtuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

Transition = namedtuple('Transition', ('state','action','next_state','reward','done'))

class QNet(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, action_dim)
        )
    def forward(self, x):
        return self.net(x)

class DQNAgent:
    def __init__(self, state_dim, action_dim,
                 gamma=0.99, lr=5e-4, buffer_size=50000, batch_size=256,
                 eps_start=1.0, eps_end=0.05, eps_decay=0.996, tau=0.005, device=None):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.batch_size = batch_size
        self.epsilon = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.tau = tau
        self.replay = deque(maxlen=buffer_size)
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.q = QNet(state_dim, action_dim).to(self.device)
        self.target = QNet(state_dim, action_dim).to(self.device)
        self.target.load_state_dict(self.q.state_dict())
        self.opt = optim.Adam(self.q.parameters(), lr=lr)

    def act(self, state):
        if random.random() < self.epsilon:
            return random.randrange(self.action_dim)
        with torch.no_grad():
            s = torch.as_tensor(np.array(state), dtype=torch.float32, device=self.device).unsqueeze(0)
            qv = self.q(s)
            return int(torch.argmax(qv, dim=1).item())

    def push(self, *args):
        self.replay.append(Transition(*args))

    def soft_update(self):
        with torch.no_grad():
            for tp, sp in zip(self.target.parameters(), self.q.parameters()):
                tp.data.mul_(1 - self.tau).add_(self.tau * sp.data)

    def train_step(self):
        if len(self.replay) < self.batch_size:
            return 0.0
        batch = random.sample(self.replay, self.batch_size)
        batch = Transition(*zip(*batch))

        s  = torch.as_tensor(np.array(batch.state), dtype=torch.float32, device=self.device)
        a  = torch.as_tensor(np.array(batch.action).reshape(-1,1), dtype=torch.int64, device=self.device)
        r  = torch.as_tensor(np.array(batch.reward).reshape(-1,1), dtype=torch.float32, device=self.device)
        ns = torch.as_tensor(np.array(batch.next_state), dtype=torch.float32, device=self.device)
        d  = torch.as_tensor(np.array(batch.done).reshape(-1,1), dtype=torch.float32, device=self.device)

        qsa = self.q(s).gather(1, a)
        with torch.no_grad():
            max_q_ns = self.target(ns).max(1, keepdim=True)[0]
            target = r + (1 - d) * self.gamma * max_q_ns

        loss = nn.functional.mse_loss(qsa, target)
        self.opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q.parameters(), max_norm=1.0)
        self.opt.step()

        # epsilon decay + target soft update
        self.epsilon = max(self.eps_end, self.epsilon * self.eps_decay)
        self.soft_update()
        return float(loss.item())
