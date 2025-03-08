from comperativeStudies.AEDQN import DQL
import numpy as np
import torch
from comperativeStudies.AEDQN import autoencoder
from comperativeStudies.AEDQN import custom_env
import utils

class Test(object):
    def __init__(self, test_loader, model_encoder, model_decoder, model_classifierA, model_classifierB,
                 DQL2_state):
        np.random.seed(5)
        torch.manual_seed(5)


        self.test_loader = utils.RL_dataloader(test_loader)

        self.E = model_encoder
        self.D = model_decoder
        self.A = model_classifierA
        self.B = model_classifierB

        self.env = custom_env.Env(self.D, self.A, self.B)
        self.replay_buffer = DQL.ReplayBuffer()

        self.state_dim = 30
        self.action_dim = 2
        self.policy = DQL.DQNAgent(self.state_dim, self.action_dim, buffer_size=100000)
        self.DQL1, self.DQL2 = self.policy.agents()

        self.continue_timesteps = 0
        self.DQL2_state = DQL2_state

        self.evaluations = []

    def evaluate(self):
        episode_num = 0
        self.policy.load_model(self.DQL2_state)

        while True:
            print('input loader')
            try:
                state_t, episode_target = self.test_loader.next_data()
                state = self.env.set_state(state_t)
                done = False
                episode_return = 0
            except:
                break

            while not done:
                with torch.no_grad():
                    action = self.policy.select_action(state)
                    action = np.float32(action)
                    action_t = torch.tensor(action)
                    next_state, reward, done = self.env(action_t, episode_target)

                state = next_state
                episode_return += reward.mean()
            print('\repisode: {}, reward: {}'.format(episode_num + 1, episode_return))
            episode_num += 1

            self.env.reset()
            with torch.no_grad():
                new_state = self.E(torch.tensor(state))
                nd, nc, nb = self.D(new_state)
            nd = {key: tensor.detach().cpu().numpy() for key, tensor in nd.items()}
            nc = {key: tensor.detach().cpu().numpy() for key, tensor in nc.items()}
            nb = {key: tensor.detach().cpu().numpy() for key, tensor in nb.items()}

            yield nd, nc, nb, episode_return



encoder = autoencoder.Encoder()
encoder.eval()


