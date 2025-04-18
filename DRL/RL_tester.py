"""credits: https://github.com/sfujim/TD3"""

# libraries
import csv
import os
import numpy as np
import pandas as pd
import torch
import torch.utils.data
import utils
from DRL.EnvClass import Env
from AAE import AAE_archi_opt
from clfs import classifier
from utils import RL_dataloader
from DRL.RL import TD3DDQN
# check for cuda and disable DSA
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ['TORCH_USE_CUDA_DSA'] = "1"

cuda = True if torch.cuda.is_available() else False
torch.cuda.empty_cache()
torch.manual_seed(0)

class Tester(object):
    def __init__(self, test_loader, model_encoder, model_decoder, model_disc, classifier, in_out, discrete, actor_path,
                 critic_path):
        # define test loader
        self.test_loader = RL_dataloader(test_loader)

        # self.expl_noise = 0.3

        self.encoder = model_encoder
        self.discriminator = model_disc
        self.decoder = model_decoder
        self.classifier = classifier
        # call enviroment in DRL/EnvClass
        self.env = Env(self.encoder, self.discriminator, self.classifier, self.decoder)
        # call replay buffer in ./utils
        self.replay_buffer = utils.ReplayBuffer()
        # set seed to 0
        torch.manual_seed(0)
        np.random.seed(0)

        self.state_dim = in_out # 30 features
        self.action_dim = 7 # 7 continuous action dim
        self.discrete_features = discrete # 3 discrete action dim: tuple of unique values in each discrete feature => max action
        self.max_action = 1 # max continuous action 
        # define TD3+DDQN network
        self.policy = TD3DDQN(self.state_dim, self.action_dim, self.discrete_features, self.max_action)

        self.continue_timesteps = 0
        # actor state dictionary
        self.actor_path = actor_path
        # critic state dictionary
        self.critic_path = critic_path
        self.evaluations = []

    def evaluate(self):
        episode_num = 0
        self.policy.load_model(self.actor_path, self.critic_path)

        while True:
            try:
                state_t, label = self.test_loader.next_data()
                state = self.env.set_state(state_t)
                done = False
                episode_return = 0
            except:
                break

            while not done:
                with torch.no_grad():
                    continuous_act, discrete_act = self.policy.select_action(state)
                    next_state, reward, done = self.env(continuous_act, discrete_act, label)

                state = next_state
                episode_return += reward.mean()
            print('\repisode: {}, reward: {}'.format(episode_num + 1, episode_return))
            episode_num += 1

            self.env.reset()

            with torch.no_grad():
                old_state = self.encoder(torch.tensor(state_t).cuda().float() if cuda else torch.tensor(state).float())
                new_state = self.encoder(torch.tensor(state).cuda() if cuda else torch.tensor(state))

            yield old_state, new_state, episode_return


