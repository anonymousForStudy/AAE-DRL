# libraries
import torch
from torch.nn import Module, BCELoss, CrossEntropyLoss
import os

from torch.nn.functional import one_hot

import utils

# Check cuda
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ['TORCH_USE_CUDA_DSA'] = "1"
cuda = True if torch.cuda.is_available() else False

"""
Build custom environment:
-------------------------
selected action -> latent dim (z)
latent dim -> decoder {continuous and discrete (combine multi-class and binary)
decoder -> encoder/generator
encoder/generator -> discriminator
decoder -> classifier
"""
class Env(Module):
    def __init__(self, model_G, model_D, model_c, model_de):
        super(Env, self).__init__()
        # set state to None
        self._state = None
        self.decoder = model_de
        self.generator = model_G
        self.disciminator = model_D
        self.classifier = model_c
        # discriminator reward coefficient
        self.d_reward_coeff = 0.4
        # classifier reward coefficient
        self.cl_reward_coeff = 0.2

        self.count = 0

    def reset(self):
        # set state to None when you reset environment
        self._state = None
        self.count = 0
        return self._state


    def set_state(self, state):
        # set state to current state
        self._state = state
        return state.detach().cpu().numpy()

    def forward(self, action, disc, episode_target):
        # define decoded output
        d_decoded = {feature: [] for feature in self.decoder.discrete_features}
        c_decoded = {feature: [] for feature in self.decoder.continuous_features}
        b_decoded = {feature: [] for feature in self.decoder.binary_features}

        with (torch.no_grad()):
            # one hot encode labels
            episode_target = episode_target.to(torch.long).cuda() if cuda else episode_target.to(torch.long)
            labels = one_hot(episode_target, num_classes=4)
            episode_target_indices = labels.argmax(dim=1)

            # fit action in continuous latent vector 
            z_cont = torch.tensor(action).cuda() if cuda else torch.tensor(action)
            # fit action in discrete latent vector
            z_disc = torch.tensor(list(disc.values())).cuda() if cuda else torch.tensor(list(disc.values()))
            z_disc = z_disc.expand(z_cont.size(0), -1).cuda() if cuda else z_disc.expand(z_cont.size(0), -1)
            
            for i in range(z_cont.shape[0]):
                # combine continuous latent vector and discrete latent vector
                z = torch.concat([z_cont[i].unsqueeze(0), z_disc[i].unsqueeze(0), labels[i].unsqueeze(0)], 1)
                # fit latent vector (combined) into decoder
                d, c, b = self.decoder(z)
                d_decoded, c_decoded, b_decoded = utils.types_append(self.decoder, d, c, b, d_decoded, c_decoded, b_decoded)
            d_decoded, c_decoded, b_decoded = utils.type_concat(self.decoder, d_decoded, c_decoded, b_decoded)
            # combine all decoded types
            decoded = utils.all_samples(d_decoded, c_decoded, b_decoded)

            #fit decoded output into the encoder/generator
            gen_out = self.generator(decoded)

            # fit encoded/generated output (new latent vector) into the discriminator
            dis_judge = self.disciminator(gen_out)

            # normalize decoded output
            decoded_normalized = (decoded - decoded.min(dim=0)[0]) / (decoded.max(dim=0)[0] - decoded.min(dim=0)[0] + 1e-7)
            # feed decoded output into classifier
            classifier_output, entropy = self.classifier.encoder(decoded_normalized)
            # predict labels with classifier
            classifier_logits, predictions = self.classifier.classify(classifier_output)
            # generate pseudo labels
            max_probs, pseudo_labels = torch.max(predictions, dim=1)

        # set class weights for minority sampling
        class_weights = torch.tensor([4.0, 2.0, 1.0, 1.0]).cuda() if cuda else torch.tensor(
            [4.0, 2.0, 1.0, 1.0])
        # calculate classifier reward
        reward_cl = self.cl_reward_coeff * torch.nn.functional.cross_entropy(classifier_logits,
                                                                             pseudo_labels, weight=class_weights).cpu().data.numpy()
        # calculate discriminator reward
        reward_d = self.d_reward_coeff * torch.nn.functional.binary_cross_entropy(dis_judge,
                    torch.ones((dis_judge.shape[0], 1), requires_grad=False).cuda() if cuda else torch.ones(
                                                    (dis_judge.shape[0], 1), requires_grad=False)).cpu().data.numpy()

        print("d", torch.nn.functional.binary_cross_entropy(dis_judge, torch.ones((dis_judge.shape[0], 1),
                requires_grad=False).cuda() if cuda else torch.ones((dis_judge.shape[0], 1), requires_grad=False)).cpu().data.numpy())
        print("cl", torch.nn.functional.cross_entropy(classifier_logits, pseudo_labels, weight=class_weights).cpu().data.numpy())

        # combine rewards
        reward = reward_cl + reward_d

        done = True
        self.count += 1
        
        # set next state to the decoded output
        next_state = decoded.detach().cpu().data.numpy()
        self._state = decoded
        return next_state, reward, done
