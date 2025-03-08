import os
import sys

from comperativeStudies.AEDQN import DQL
import numpy as np
import torch
from comperativeStudies.AEDQN import autoencoder
from comperativeStudies.AEDQN import classifiers
from comperativeStudies.AEDQN import custom_env
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import utils



def evaluate_policy(policy, dataloader, env, episode_num=10):
    avg_reward = 0.
    env.reset()

    for i in range(0, episode_num):
        input, episode_target = dataloader.next_data()
        obs = env.set_state(input)
        done = False
        # episode_target = (label + torch.randint(4, label.shape)) % 4

        while not done:
            action = policy.select_action(obs)
            new_state, reward, done = env(action, episode_target)
            avg_reward += reward

    avg_reward /= episode_num

    return avg_reward



class Train(object):
    def __init__(self, train_loader, valid_loader, model_decoder, model_classifierA, model_classifierB, max_timestep,
                 batch_size, eval_freq, start_timestep, max_ep_steps, DQL2_state):
        np.random.seed(5)
        torch.manual_seed(5)


        self.train_loader = utils.RL_dataloader(train_loader)
        self.valid_loader = utils.RL_dataloader(valid_loader)

        self.epoch_size = len(self.valid_loader)
        self.max_timesteps = max_timestep

        self.batch_size = batch_size
        self.start_timesteps = start_timestep
        self.max_episodes_steps = max_ep_steps
        self.eval_freq = eval_freq

        self.D = model_decoder
        self.A = model_classifierA
        self.B = model_classifierB

        self.env = custom_env.Env(self.D, self.A, self.B)
        self.replay_buffer = DQL.ReplayBuffer()

        self.state_dim = 30
        self.action_dim = 2
        self.policy = DQL.DQNAgent(self.state_dim, self.action_dim, buffer_size=100000)
        self.DQL1, self.DQL2 = self.policy.agents()
        self.d_decoded = {feature: [] for feature in self.D.discrete_features}
        self.c_decoded = {feature: [] for feature in self.D.continuous_features}
        self.b_decoded = {feature: [] for feature in self.D.binary_features}

        self.continue_timesteps = 0
        self.DQL2_state = DQL2_state
        self.evaluations = []

    def train(self):
        sum_return = 0
        episode_reward = 0
        episode_timesteps = 0
        episode_num = 0
        d_list = []
        c_list = []
        b_list = []

        state_t, episode_target = self.train_loader.next_data()
        state = self.env.set_state(state_t)
        # episode_target = (torch.randint(4, label.shape) + label) % 4

        done = False
        self.env.reset()

        print('start/continue model from t: {}'.format(0))
        print('start buffer length: {}'.format(len(self.replay_buffer)))
        for t in range((self.continue_timesteps), int(self.max_timesteps)):
            episode_timesteps += 1
            if t < 10:
                action_t = torch.randn(self.batch_size, 2)
                action = action_t.detach().cpu().numpy()
            else:
                action = self.policy.select_action(state)
                action = np.float32(action)
                action_t = torch.tensor(action)
            next_state, reward, done = self.env(action_t, episode_target)

            self.replay_buffer.add((state, action, next_state, reward, done))

            state = next_state
            episode_reward += reward


            if t >= 10:
                self.DQL2.load_state_dict(self.DQL1.state_dict())
                self.policy.update(self.replay_buffer)

            if done:
                state_t, episode_target = self.train_loader.next_data()
                # episode_target = (torch.randint(4, label.shape) + label) % 4
                state = self.env.set_state(state_t)

                done = False
                self.env.reset()

                print('\rstep: {}, episode: {}, reward: {}'.format(t + 1, episode_num + 1, episode_reward), end='')
                sum_return += episode_reward
                episode_reward = 0
                episode_timesteps = 0
                episode_num += 1

            if (t + 1) % self.eval_freq == 0:
                episode_result = "episode: {} average reward: {}".format(episode_num,
                                                                         sum_return / episode_num)
                print('\r' + episode_result)

                valid_episode_num = 6
                self.evaluations.append(evaluate_policy(self.policy, self.valid_loader, self.env,
                                                        episode_num=valid_episode_num))
                eval_result = "episodes: {}".format(self.evaluations[-1])
                print(eval_result)

            self.policy.save_model(self.DQL2_state)

            nd, nc, nb = self.D(torch.tensor(action).float())
            # labels = one_hot(episode_target, num_classes=4)
            d_list.append(nd)
            c_list.append(nc)
            b_list.append(nb)
        d_cat = {key: torch.cat([d[key] for d in d_list], dim=0) for key in d_list[0]}
        c_cat = {key: torch.cat([d[key] for d in c_list], dim=0) for key in c_list[0]}
        b_cat = {key: torch.cat([d[key] for d in b_list], dim=0) for key in b_list[0]}
        return d_cat, c_cat, b_cat



encoder = autoencoder.Encoder()
encoder.eval()

decoder = autoencoder.Decoder(utils.discrete, utils.continuous, utils.binary)
decoder.load_state_dict(torch.load("ae_2nd.pth", map_location="cpu")["dec"])
decoder.eval()

classifierA = classifiers.Classifier(30, 4)
classifierA.load_state_dict(torch.load("classifer2_model.pth", map_location="cpu"))
classifierA.eval()

classifierB = classifiers.Classifier(30, 4)
classifierB.load_state_dict(torch.load("classifer2_model.pth", map_location="cpu"))
classifierB.eval()


