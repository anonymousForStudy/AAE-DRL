# libraries
import torch
import torch.nn as nn
import os
import numpy as np
import torch.nn.functional as F

import utils


class Env(nn.Module):
    def __init__(self, model_G, model_D, model_classifier, model_decoder):
        """
        environment model of RL

        Parameters
        ----------
        args : dict
            args dictionary look at :func: '~gan_main.parse_args'
        model_G : torch.nn.Module
            generator model
        model_D : torch.nn.Module
            discriminator model
        model_classifier : torch.nn.Module
            classifier model
        model_decoder : torch.nn.Module
            decoder model
        """
        super(Env, self).__init__()

        self._state = None

        # generator model
        self.generator = model_G

        # disciminator model
        self.disciminator = model_D

        # classifier model for caluclating the reward
        self.classifier = model_classifier
        # decoder model
        self.decoder = model_decoder

        self.d_reward_coeff = 1
        self.cl_reward_coeff = 0.5

        # for calculating the discriminator reward
        self.bin = torch.nn.BCELoss()
        self.ce = torch.nn.CrossEntropyLoss()

        self.count = 0

    def reset(self):
        """
        reset env by setting count to zero
        """
        self.count = 0

    def set_state(self, state):
        """
        detach input from tensor to numpy (get numpy state)
        Parameters
        ----------
        state : torch.Tensor
            tensor of input state

        Returns
        -------
        numpy.ndarray
            numpy array of state
        """
        self._state = state
        return state.detach().cpu().numpy().squeeze()

    def forward(self, action, disc, episode_target, t=None):
        d_decoded = {feature: [] for feature in self.decoder.discrete_features}
        c_decoded = {feature: [] for feature in self.decoder.continuous_features}
        b_decoded = {feature: [] for feature in self.decoder.binary_features}

        with torch.no_grad():
            with (torch.no_grad()):
                z_cont = torch.tensor(action)
                z_disc = torch.tensor(list(disc.values()))
                z_disc = z_disc.expand(z_cont.size(0), -1)
                z = torch.concat([z_cont, z_disc], 1)
                gen_out, _ = self.generator(z)

                dis_judge, _ = self.disciminator(gen_out)
                d, c, b = self.decoder(gen_out.transpose(0,1))
                d_decoded, c_decoded, b_decoded = utils.types_append(self.decoder, d, c, b, d_decoded, c_decoded, b_decoded)
                d_decoded, c_decoded, b_decoded = utils.type_concat(self.decoder, d_decoded, c_decoded, b_decoded)
                decoded = utils.all_samples(d_decoded, c_decoded, b_decoded)
                classification = self.classifier(decoded)

        episode_target = episode_target.to(torch.int64)
        probs = F.softmax(classification, dim=1)
        correct_class_probs = probs.gather(1, episode_target.unsqueeze(1)).squeeze()
        cross_entropy_loss = -torch.log(correct_class_probs)

        reward_cl = self.cl_reward_coeff * cross_entropy_loss.cpu().data.numpy().squeeze()
        reward_d = self.d_reward_coeff * self.bin(dis_judge, torch.ones_like(dis_judge)).cpu().data.numpy()

        reward = reward_cl + reward_d

        done = True

        self.count += 1

        # the nextState
        next_state = gen_out.detach().squeeze(1)
        self._state = gen_out
        return next_state, reward, done, cross_entropy_loss.cpu().data.numpy().squeeze()
