# libraries
import copy
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import utils
# check cude
cuda = True if torch.cuda.is_available() else False

"""
Actor (TD3):
------------
state dim: 30 -> action dim (continuous): 7
action dim * max action: 7*1 = 7
"""
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor, self).__init__()
        # set action range for continuous action
        self.max_action = max_action

        self.l1 = nn.Linear(state_dim, 25)
        self.l2 = nn.Linear(25, 25)
        self.l3 = nn.Linear(25, action_dim)
        # self.l4 = nn.Linear(50, 50)
        # self.l5 = nn.Linear(50, 50)
        # self.l6 = nn.Linear(50, 50)
        # self.l7 = nn.Linear(50, 50)
        # self.l8 = nn.Linear(50, action_dim)


    def forward(self, state):
        a = F.relu(self.l1(state))
        a = F.relu(self.l2(a))
        # a = F.relu(self.l3(a))
        # a = F.relu(self.l4(a))
        # a = F.relu(self.l5(a))
        # a = F.relu(self.l6(a))
        # a = F.relu(self.l7(a))
        continuous_actions = self.max_action * F.relu(self.l3(a))
        return continuous_actions

"""
Critic (TD3):
------------
state dim: 30 + action dim (continuous): 7 -> max action: 1
+
DDQN block => state dim: 30 + action dim (discrete) : num_classes_in_discrete_features
"""

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, discrete_ranges):
        super(Critic, self).__init__()
        self.discrete_ranges = discrete_ranges

        self.l1 = nn.Linear(state_dim+action_dim, 25)
        self.l2 = nn.Linear(25, 25)
        self.l3 = nn.Linear(25, 1)

        self.l4 = nn.Linear(state_dim+action_dim, 25)
        self.l5 = nn.Linear(25, 25)
        self.l6 = nn.Linear(25, 1)

        self.discrete_q = nn.ModuleDict({
            name: nn.Sequential(
                nn.Linear(state_dim + action_dim, 25),
                nn.ReLU(),
                nn.Linear(25, 25),
                nn.ReLU(),
                nn.Linear(25, num_actions),
            ) for name, num_actions in discrete_ranges.items()
        })

    def forward(self, state, continuous_action):
        # join state and continuous action dim
        sa = torch.cat([state, continuous_action], 1)

        # Q1
        q1_cont = F.relu(self.l1(sa))
        q1_cont = F.relu(self.l2(q1_cont))
        q1_cont = self.l3(q1_cont)

        # Q2
        q2_cont = F.relu(self.l4(sa))
        q2_cont = F.relu(self.l5(q2_cont))
        q2_cont = self.l6(q2_cont)

        # Q discrete
        discrete_q_values = {
            name: self.discrete_q[name](sa)
            for name in self.discrete_ranges.keys()
        }

        return q1_cont, q2_cont, discrete_q_values

    def Q1(self, state, action):
        sa = torch.cat([state, action], 1)
        q1 = F.relu(self.l1(sa))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)
        return q1




class TD3DDQN(object):
    def __init__(self, state_dim, action_dim, discrete_features, max_action, discount=0.8, tau=0.005, policy_noise=0.2,
                 noise_clip=0.5, policy_freq=1):

        self.discrete_features = discrete_features

        self.actor = Actor(state_dim, action_dim, max_action).cuda() if cuda else (
            Actor(state_dim, action_dim, max_action))
        # copy actor to define actor's target
        self.actor_target = copy.deepcopy(self.actor)
        # define actor's optimizer
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=0.00001)

        self.critic = Critic(state_dim, action_dim, discrete_features).cuda() if cuda else (
            Critic(state_dim, action_dim, discrete_features))
        # copy critic to define critic's target
        self.critic_target = copy.deepcopy(self.critic)
        # define critic's optimizer
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=0.00001)

        # DDQN parameters: epsilon
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995

        # TD3 parameters
        self.max_action = max_action # continuous max action
        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq
                     
        self.total_it = 0
        # define replay buffer for training
        self.replay_buffer = utils.ReplayBuffer()


    def select_action(self, state):
        # convert state np array to tensor
        state_tensor = torch.FloatTensor(state).cuda() if cuda else torch.FloatTensor(state)
        # fit state into actor to define continuous actions
        continuous_actions = self.actor(state_tensor)
        # define critic input to calculate discrete actions
        sa = torch.cat([state_tensor, (continuous_actions.cuda() if cuda else continuous_actions)], 1)
        discrete_actions = {}

        for name, num_actions in self.discrete_features.items():
            if random.random() < self.epsilon:
                # select random action => epsilon greedy => EXPLORATION
                discrete_actions[name] = random.randrange(num_actions)
            else:
                # select action according to DDQN block (in critic network) => EXPLOITATION
                q_values = self.critic.discrete_q[name](sa)
                discrete_actions[name] = q_values.argmax().item()
        return continuous_actions, discrete_actions

    def train(self):
        self.total_it += 1

        # define items in replay buffer and convert them from np arrays to tensors
        state, continuous_action, discrete_action, next_state, reward, done, target = self.replay_buffer.sample()
        state = torch.FloatTensor(state).cuda() if cuda else torch.FloatTensor(state)
        continuous_action = torch.FloatTensor(continuous_action).cuda() if cuda else torch.FloatTensor(continuous_action)
        next_state = torch.FloatTensor(next_state).cuda() if cuda else torch.FloatTensor(next_state)
        reward = torch.FloatTensor(reward).cuda() if cuda else torch.FloatTensor(reward)
        done = torch.FloatTensor(done).reshape(-1, 1).cuda() if cuda else torch.FloatTensor(done).reshape(-1, 1)
        target = torch.FloatTensor(target).cuda() if cuda else torch.FloatTensor(target)

        with torch.no_grad():
            # TD3 noise
            noise = (torch.randn_like(continuous_action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            # clip next action (TD3)
            next_continuous_action = (self.actor_target(next_state) + noise).clamp(0, self.max_action) # max action range set to [0, 1]
            # define critic targets : Q1, Q2 and Q discrete
            target_Q1, target_Q2, target_discrete_Q = self.critic_target(next_state, next_continuous_action)
            # minimum Q estimates in TD3 
            target_Q = torch.min(target_Q1, target_Q2)
            # calculate TD3 target
            target_Q = reward + (1 - done) * 0.99 * target_Q

            # define DDQN target
            discrete_targets = {}
            for name in self.discrete_features.keys():
                next_q_values = target_discrete_Q[name]
                next_q_value = next_q_values.max(dim=1, keepdim=True)[0]
                # calculate DDQN target
                discrete_targets[name] = reward + (1 - done) * 0.99 * next_q_value

        # define current Qs
        current_Q1, current_Q2, current_discrete_Q = self.critic(state, continuous_action)
        # calculate TD3 critic loss
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)
        # calculate DDQN loss and join with TD3 critic loss
        for name in self.discrete_features.keys():
            critic_loss += F.mse_loss(current_discrete_Q[name], discrete_targets[name])

        print("critic loss", critic_loss)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        if self.total_it % self.policy_freq == 0:
            # continuous_actions = self.actor(state)
            # actor_loss = -self.critic.q1(torch.cat([state, continuous_actions], dim=1)).mean()
            # calculate actor loss
            actor_loss = -self.critic.Q1(state, self.actor(state)).mean()
            print("actor_loss", actor_loss)

            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            
            # save critic parameters
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(0.005 * param.data + (1 - 0.005) * target_param.data)
            # save actor parameters
            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(0.005 * param.data + (1 - 0.005) * target_param.data)
        # calculate epsilon (DDQN)
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def store_transition(self, state, continuous_action, discrete_actions, next_state, reward, done, target):
        # store in replay buffer
        self.replay_buffer.add((state, continuous_action, discrete_actions, next_state, reward, done, target))

    def save_model(self, actor_path, critic_path):
        # save state dictionaries after validation
        torch.save(self.actor.state_dict(), f"{actor_path}")
        torch.save(self.critic.state_dict(), f"{critic_path}")

    def load_model(self, actor_path, critic_path):
        # load state dictionaries before testing
        self.actor.load_state_dict(torch.load(f"{actor_path}"))
        self.critic.load_state_dict(torch.load(f"{critic_path}"))
