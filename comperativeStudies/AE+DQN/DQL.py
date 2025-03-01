import random
from collections import deque
import numpy as np
import torch
from torch.nn import Linear, ReLU, Module, Sequential, MSELoss
from torch.optim import SGD

class DQN(Module):
    def __init__(self, state_dim, action_dim):
        super(DQN, self).__init__()
        self.seq = Sequential(

            Linear(state_dim, 22),
            ReLU(),
            Linear(22, 11),
            ReLU(),
            Linear(11, 22),
            ReLU(),
            Linear(22, 44),
            ReLU(),
            Linear(44, action_dim),
            ReLU(),

        )

    def forward(self, x):
        x = self.seq(x)
        return x


class DQNAgent:
    def __init__(self, state_dim, action_dim, buffer_size):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = max(0.9, 0.99 - 0.01 * 0.9)
        self.memory = deque(maxlen=buffer_size)
        self.DQL1 = DQN(state_dim, action_dim)
        self.optimizer = torch.optim.Adam(self.DQL1.parameters(), lr=3e-4)
        self.DQL2 = DQN(state_dim, action_dim)
        self.mse = MSELoss()

    def select_action(self, state, epsilon_range=(0.01, 15.0), n_noise_samples=100):
        state = torch.FloatTensor(state)

        # Get base Q-values
        with torch.no_grad():
            base_q_values = self.DQL1(state)

        # Apply Laplacian noise mechanism n times
        noisy_decisions = []
        for _ in range(n_noise_samples):
            # Generate Laplacian noise
            sensitivity = torch.max(torch.abs(base_q_values))  # L1 sensitivity (Manhattan distance)
            epsilon = random.uniform(epsilon_range[0], epsilon_range[1])
            scale = sensitivity / epsilon
            noise = torch.tensor(np.random.laplace(0, scale, base_q_values.shape))

            # Add noise to Q-values
            noisy_q_values = base_q_values + noise

            # Apply threshold (as mentioned in paper: set to 1 if >= 0.5, else 0)
            decision = (noisy_q_values >= 0.5).long()
            noisy_decisions.append(decision)

        # Majority voting as mentioned in the paper
        noisy_decisions = torch.stack(noisy_decisions)
        final_action = torch.mode(noisy_decisions, dim=0).values

        return final_action

    def agents(self):
        return self.DQL1, self.DQL2

    def update(self, replay_buffer):
        states, actions, next_states, rewards, dones = replay_buffer.sample()

        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)

        q_values = self.DQL1(states)

        with torch.no_grad():
            next_actions = self.DQL1(next_states).argmax(dim=1, keepdim=True)
            next_q_values = self.DQL2(next_states).gather(1, next_actions)
            target_q_values = rewards + self.gamma * next_q_values * (1 - dones)

        loss = self.mse(q_values, target_q_values)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        for target_param, local_param in zip(self.DQL2.parameters(), self.DQL1.parameters()):
            target_param.data.copy_(local_param.data)

        return loss.item()


class ReplayBuffer(object):
    def __init__(self):
        self.storage = []
        self._saved = []
        self._sample_ind = None
        self._ind_to_save = 0

    def add(self, data):
        self.storage.append(data)
        self._saved.append(False)

    def sample(self):
        ind = np.random.randint(len(self.storage))
        self._sample_ind = ind
        return self[ind]

    def __len__(self):
        return len(self.storage)

    def __getitem__(self, items):
        if hasattr(items, '__iter__'):
            items_iter = items
        else:
            items_iter = [items]

        x, y, u, r, d = [], [], [], [], []
        for i in items_iter:
            X, Y, U, R, D = self.storage[i]
            x.append(np.array(X, copy=False))
            y.append(np.array(Y, copy=False))
            u.append(np.array(U, copy=False))
            r.append(np.asarray(R))
            d.append(np.asarray(D))

        return (np.array(x).squeeze(0), np.array(y).squeeze(0), np.array(u).squeeze(0), np.array(r).squeeze(0),
                np.array(d).squeeze(0).reshape(-1, 1))


