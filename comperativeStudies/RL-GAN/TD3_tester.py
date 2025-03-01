import csv
import os
import sys

import numpy as np
import torch
from comperativeStudies.RLGAN.TD3Env import Env
from comperativeStudies.RLGAN.RL_model import TD3
from comperativeStudies.RLGAN import AE, gan, MNISTClassifier
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import utils


class Tester(object):
    def __init__(self, test_loader, model_encoder, model_decoder, model_g, model_d, model_classifier, discrete, actor_path,
                 critic_path):


        self.test_loader = utils.RL_dataloader(test_loader)

        self.batch_size = 10
        self.max_episodes_steps = 4000

        self.z_dim = 7
        self.max_action = 1

        self.encoder = model_encoder
        self.decoder = model_decoder
        self.G = model_g
        self.D = model_d
        self.classifier = classifier

        self.env = Env(self.G, self.D, model_classifier, model_decoder)

        self.state_dim = 30
        self.action_dim = 7
        self.discrete_features = discrete
        self.max_action = 1
        self.policy = TD3(self.state_dim, self.action_dim, self.discrete_features, self.max_action)
        self.actor_path = actor_path
        self.critic_path = critic_path

    def evaluate(self):
        episode_num = 0
        number_correct = 0
        self.policy.load(self.actor_path, self.critic_path)
        while True:
            print('input loader')
            try:
                state_t, label = self.test_loader.next_data()
                episode_target = (torch.randint(4, label.shape) + label) % 4
                state = self.env.set_state(state_t)
                state = torch.tensor(state).float()
                done = False
                episode_return = 0
            except:
                break

            while not done:
                with torch.no_grad():
                    continuous_act, discrete_act = self.policy.select_action(state, episode_target)
                    next_state, reward, done, episode_target = self.env(continuous_act, discrete_act, episode_target)

                state = next_state
                episode_return += reward.mean()
            print('\repisode: {}, reward: {}'.format(episode_num + 1, episode_return))
            episode_num += 1

            self.env.reset()

            with torch.no_grad():
                od, oc, ob = self.decoder(state_t.float().transpose(0,1))
                nd, nc, nb = self.decoder(torch.tensor(state).transpose(0, 1).float())

            yield od, oc, ob, nd, nc, nb, episode_return



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
# classifier.load_state_dict(torch.load("best_model1.pth", map_location="cpu"))
classifier.eval()
