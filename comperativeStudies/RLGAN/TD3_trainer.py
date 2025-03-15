import csv
import os
import sys

import numpy as np
import torch
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from comperativeStudies.RLGAN.TD3Env import Env
from comperativeStudies.RLGAN.RL_model import TD3
from comperativeStudies.RLGAN import AE, gan, MNISTClassifier

import utils




def evaluate_policy(policy, dataloader, env, episode_num=10, t=None):
    avg_reward = 0.
    env.reset()

    for i in range(0, episode_num):
        input, label = dataloader.next_data()
        obs = env.set_state(input)
        done = False
        episodeTarget = (label + torch.randint(4, label.shape)) % 4
        obs = torch.tensor(obs).float()
        while not done:
            continuous_act, discrete_act = policy.select_action(obs, episodeTarget)
            new_state, reward, done, _ = env(continuous_act, discrete_act, episodeTarget, t)
            avg_reward += reward

    avg_reward /= episode_num

    return avg_reward


class Trainer(object):
    def __init__(self, train_loader, valid_loader, model_encoder, model_decoder, model_g, model_d,
                 model_classifier, discrete, max_timesteps, eval_freq, start_timestep, max_ep_steps, actor_path, critic_path):

        self.train_loader = utils.RL_dataloader(train_loader)
        self.valid_loader = utils.RL_dataloader(valid_loader)

        self.epoch_size = len(self.valid_loader)
        self.max_timesteps = max_timesteps

        self.batch_size = 10
        self.eval_freq = eval_freq
        self.start_timesteps = start_timestep
        self.max_episodes_steps = max_ep_steps

        self.expl_noise = 0.3

        self.encoder = model_encoder
        self.De = model_decoder
        self.G = model_g
        self.D = model_d
        self.classifier = classifier

        self.env = Env(self.G, self.D, model_classifier, model_decoder)

        torch.manual_seed(0)
        np.random.seed(0)

        self.state_dim = 30
        self.action_dim = 7
        self.discrete_features = discrete
        self.max_action = 1
        self.policy = TD3(self.state_dim, self.action_dim, self.discrete_features, self.max_action)
        self.d_decoded = {feature: [] for feature in self.De.discrete_features}
        self.c_decoded = {feature: [] for feature in self.De.continuous_features}
        self.b_decoded = {feature: [] for feature in self.De.binary_features}


        self.continue_timesteps = 0
        self.actor_path = actor_path
        self.critic_path = critic_path
        self.evaluations = []

    def train(self):
        sum_return = 0
        episode_reward = 0
        episode_timesteps = 0
        episode_num = 0
        d_list = []
        c_list = []
        b_list = []

        state_t, label = self.train_loader.next_data()
        state = self.env.set_state(state_t)
        episode_target = (torch.randint(4, label.shape) + label) % 4


        done = False
        self.env.reset()

        for t in range(int(self.continue_timesteps), int(self.max_timesteps)):
            episode_timesteps += 1

            state = torch.tensor(state).float()
            continuous_act, discrete_act = self.policy.select_action(state, episode_target)

            next_state, reward, done, _ = self.env(continuous_act, discrete_act, episode_target)

            self.policy.store_transition(state, continuous_act, discrete_act,
                          next_state, reward, done, episode_target)

            state = next_state
            episode_reward += reward
            if t >= self.start_timesteps:
                self.policy.train()

            if done:
                state_t, label = self.train_loader.next_data()
                episode_target = (torch.randint(4, label.shape) + label) % 4

                state = self.env.set_state(state_t)

                done = False
                self.env.reset()

                print('\repisode: {}, reward: {}'.format(episode_num + 1, episode_reward), end='')
                sum_return += episode_reward
                episode_reward = 0
                episode_timesteps = 0
                episode_num += 1

            # Evaluate episode
            if (t + 1) % self.eval_freq == 0:
                episode_result = "episode: {} average reward: {}".format(episode_num,
                                                                         sum_return / episode_num)
                print('\r' + episode_result)

                valid_episode_num = 6
                self.evaluations.append(evaluate_policy(self.policy, self.valid_loader, self.env,
                                                        episode_num=valid_episode_num, t=t))
                eval_result = "episodes: {}".format(self.evaluations[-1])
                print(eval_result)

            self.policy.save(self.actor_path, self.critic_path)
            nd, nc, nb = self.De(torch.tensor(state).transpose(0,1).float())

            d_list.append(nd)
            c_list.append(nc)
            b_list.append(nb)

        d_cat = {key: torch.cat([d[key] for d in d_list], dim=0) for key in d_list[0]}
        c_cat = {key: torch.cat([d[key] for d in c_list], dim=0) for key in c_list[0]}
        b_cat = {key: torch.cat([d[key] for d in b_list], dim=0) for key in b_list[0]}
        return d_cat, c_cat, b_cat




encoder = AE.Encoder()
encoder.eval()
decoder = AE.Decoder(10, 30, 64, utils.discrete, utils.continuous, utils.binary)
decoder.load_state_dict(torch.load("ae1.pth", map_location="cpu")["dec"])
decoder.eval()
generator = gan.Generator()
discriminator = gan.Discriminator()
discriminator.load_state_dict(torch.load("gan.pth", map_location="cpu")["disc"])
discriminator.eval()
classifier = MNISTClassifier.Classifier()
#not generated according to threshold
# classifier.load_state_dict(torch.load("best_model1.pth", map_location="cpu"))
classifier.eval()


