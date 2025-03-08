import random

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
import os

import utils


# Implementation of Twin Delayed Deep Deterministic Policy Gradients (TD3)
# Paper: https://arxiv.org/abs/1802.09477


class Actor(nn.Module):
    """
    actor model
    """
    def __init__(self, state_dim, action_dim, max_action):
        """
        initialize actor model

        Parameters
        ----------
        state_dim : int
            dimension of state
        action_dim : int
            dimension of action
        max_action : int, float
            maximum of absolute value possible for action
        """
        super(Actor, self).__init__()
        self.l1 = nn.Linear(state_dim + 1, 128)
        self.l2 = nn.Linear(128, 128)
        self.l3 = nn.Linear(128, 30)
        self.continuous_head = nn.Sequential(
            nn.Linear(30, action_dim),
            nn.Tanh()
        )

        self.max_action = max_action

    def forward(self, state, target):
        """
        forward pass of actor model

        Parameters
        ----------
        state : torch.Tensor
            state tensor

        Returns
        -------
        torch.Tensor
            action tensor in range (-max_action, max_action)
        """
        sa = torch.cat([state, target.unsqueeze(1)], 1)

        a = F.relu(self.l1(sa))  # B x state_dim ---> B x 256
        a = F.relu(self.l2(a))  # B x 256 --->  B x 256
        a = F.relu(self.l3(a))  # B x 256 --->  B x 256
        a = self.max_action * self.continuous_head(a)
        return a  # (values in range [-max_action, max_action])


class Critic(nn.Module):
    """
    critic model
    """
    def __init__(self, state_dim, action_dim, discrete_ranges):
        """
        initialize critic model

        Parameters
        ----------
        state_dim :  int
            dimension of state
        action_dim :  int
            dimension of action
        """
        super(Critic, self).__init__()
        self.discrete_ranges = discrete_ranges
        # Q1 architecture
        self.l1 = nn.Linear(state_dim + action_dim + 1, 256)
        self.l2 = nn.Linear(256, 256)
        self.l3 = nn.Linear(256, 1)

        # Q2 architecture
        self.l4 = nn.Linear(state_dim + action_dim + 1, 256)
        self.l5 = nn.Linear(256, 256)
        self.l6 = nn.Linear(256, 1)

        self.discrete_q = nn.ModuleDict({
            name: nn.Sequential(
                nn.Linear(state_dim + action_dim + 1, 25),
                nn.ReLU(),
                nn.Linear(25, 25),
                nn.ReLU(),
                nn.Linear(25, num_actions)  # Q-values for each discrete action
            ) for name, num_actions in discrete_ranges.items()
        })

    def forward(self, state, action, target):
        sa = torch.cat([state, action, target.unsqueeze(1)], 1)  # (B x state_dim, B x action_dim) ---> B x (state_dim + action_dim)

        q1 = F.relu(self.l1(sa))  # B x (state_dim + action_dim) ---> B x 256
        q1 = F.relu(self.l2(q1))  # B x 256 ---> B x 256
        q1 = self.l3(q1)  # B x 256 ---> B x 1

        q2 = F.relu(self.l4(sa))  # B x (state_dim + action_dim) ---> B x 256
        q2 = F.relu(self.l5(q2))  # B x 256 ---> B x 256
        q2 = self.l6(q2)  # B x 256 ---> B x 1

        discrete_q_values = {
            name: self.discrete_q[name](sa)
            for name in self.discrete_ranges.keys()
        }
        return q1, q2, discrete_q_values  # B x 1, B x 1

    def Q1(self, state, action, target):
        sa = torch.cat([state, action, target.unsqueeze(1)], 1)  # (B x state_dim, B x action_dim) ---> B x (state_dim + action_dim)

        q1 = F.relu(self.l1(sa))  # B x (state_dim + action_dim) ---> B x 256
        q1 = F.relu(self.l2(q1))  # B x 256 ---> B x 256
        q1 = self.l3(q1)  # B x 256 ---> B x 1
        return q1


class TD3(object):
    """
    Twin Delayed Deep Deterministic Policy Gradients model
    """
    def __init__(self, state_dim, action_dim, discrete_features, max_action):
        self.discrete_ranges = discrete_features
        self.max_action = max_action
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995

        self.actor = Actor(state_dim, action_dim, max_action)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=0.00001)

        self.critic = Critic(state_dim, action_dim, discrete_features)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=0.00001)
        self.replay_buffer = utils.ReplayBuffer()

        self.policy_noise = 0.2 * max_action
        self.noise_clip = 0.5 * max_action
        self.policy_freq = 2
        self.total_it = 0

    def select_action(self, state, target):
        """
        select action from actor model

        Parameters
        ----------
        state : torch.Tensor
            tensor of state

        Returns
        -------
        numpy.ndarray
            numpy array of action chosen by actor
        """
        action = self.actor(state, target).cpu().data.numpy()
        action = torch.tensor(action).float()
        sa = torch.cat([state, action, target.unsqueeze(1)], 1)
        discrete_actions = {}

        # Epsilon-greedy selection for discrete actions
        for name, num_actions in self.discrete_ranges.items():
            if random.random() < self.epsilon:
                discrete_actions[name] = random.randrange(num_actions)
            else:
                q_values = self.critic.discrete_q[name](sa)
                discrete_actions[name] = q_values.argmax().item()
        return action, discrete_actions

    def train(self):
        self.total_it += 1

        state, continuous_action, discrete_action, next_state, reward, done, target = self.replay_buffer.sample()

        state = torch.FloatTensor(state)
        continuous_action = torch.FloatTensor(continuous_action)
        next_state = torch.FloatTensor(next_state)
        reward = torch.FloatTensor(reward)
        done = torch.FloatTensor(done).reshape(-1, 1)
        target = torch.FloatTensor(target)

        with torch.no_grad():
            # Select next continuous actions with noise
            noise = torch.randn_like(continuous_action) * self.policy_noise
            noise = noise.clamp(-self.noise_clip, self.noise_clip)

            next_continuous_action = (
                    self.actor_target(next_state, target) + noise
            ).clamp(-self.max_action, self.max_action)

            # Compute target Q values
            target_Q1, target_Q2, target_discrete_Q = self.critic_target(next_state, next_continuous_action, target)
            target_Q = torch.min(target_Q1, target_Q2)

            # Final targets for continuous Q values
            target_Q = reward + (1 - done) * 0.99 * target_Q

            # Compute target Q values for discrete actions
            discrete_targets = {}
            for name in self.discrete_ranges.keys():
                next_q_values = target_discrete_Q[name]
                next_q_value = next_q_values.max(dim=1, keepdim=True)[0]
                discrete_targets[name] = reward + (1 - done) * 0.99 * next_q_value

        # Get current Q estimates
        current_Q1, current_Q2, current_discrete_Q = self.critic(state, continuous_action, target)

        # Compute critic losses
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Delayed policy updates
        if self.total_it % self.policy_freq == 0:
            # Compute actor loss (only for continuous actions)
            continuous_actions = self.actor(state, target)
            actor_loss = -self.critic.Q1(state, continuous_actions, target).mean()

            # Optimize the actor
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            # Update target networks
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(0.005 * param.data + (1 - 0.005) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(0.005 * param.data + (1 - 0.005) * target_param.data)

        # Update epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def store_transition(self, state, continuous_action, discrete_actions, next_state, reward, done, target):
        self.replay_buffer.add((state, continuous_action, discrete_actions, next_state, reward, done, target))
    def save(self, actor_path, critic_path):
        """
        save models and optimizers

        Parameters
        ----------
        filename : str
            file name
        directory : str
            directory names

        """
        torch.save(self.critic.state_dict(), f'{critic_path}')
        torch.save(self.actor.state_dict(), f'{actor_path}')

    def load(self, actor_path, critic_path):
        torch.save(self.critic.state_dict(), f'{critic_path}')
        torch.save(self.actor.state_dict(), f'{actor_path}')
