import numpy as np
import torch
import torch.nn as nn
import os

import utils


class Env(nn.Module):
    def __init__(self, model_decoder, classifierA, classifierB):
        super(Env, self).__init__()

        self._state = None
        self.decoder = model_decoder
        self.classifierA = classifierA
        self.classifierB = classifierB

        self.ce = torch.nn.CrossEntropyLoss()

        self.count = 0

    def reset(self):
        self.count = 0

    def set_state(self, state):
        self._state = state
        return state.detach().cpu().numpy()

    def forward(self, action, episode_target):
        d_decoded = {feature: [] for feature in self.decoder.discrete_features}
        c_decoded = {feature: [] for feature in self.decoder.continuous_features}
        b_decoded = {feature: [] for feature in self.decoder.binary_features}
        with (torch.no_grad()):
            d, c, b = self.decoder(action.float())
            d_decoded, c_decoded, b_decoded = utils.types_append(self.decoder, d, c, b, d_decoded, c_decoded, b_decoded)
            d_decoded, c_decoded, b_decoded = utils.type_concat(self.decoder, d_decoded, c_decoded, b_decoded)
            decoded = utils.all_samples(d_decoded, c_decoded, b_decoded)
            decoded_normalized = (decoded - decoded.min(dim=0)[0]) / (decoded.max(dim=0)[0] - decoded.min(dim=0)[0] + 1e-7)
            classifierA_output = self.classifierA(decoded_normalized)
            classifierB_output = self.classifierB(decoded_normalized)
            sftA = torch.nn.functional.softmax(classifierA_output, dim=1)
            predA = np.argmax(sftA, axis=1)
            sftB = torch.nn.functional.softmax(classifierB_output, dim=1)
            predB = np.argmax(sftB, axis=1)


        episode_target = episode_target.to(torch.int64)

        reward_A = 0.4 * self.ce(classifierA_output, predA).cpu().data.numpy().squeeze()
        reward_B = 0.4 * self.ce(classifierB_output, predB).cpu().data.numpy().squeeze()

        reward = reward_A + reward_B
        done = True


        self.count += 1

        # the nextState
        next_states = decoded.detach().cpu().data.numpy()
        self._state = decoded
        return next_states, reward, done