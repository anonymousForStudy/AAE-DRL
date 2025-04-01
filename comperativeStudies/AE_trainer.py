# libraries
import os
import sys
import warnings

import pandas as pd
import torch
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import utils

warnings.filterwarnings("ignore")

def save_features_to_csv(discrete_samples, continuous_samples, binary_samples, file_name):
    def dict_to_df(tensor_dict):
        all_data = []
        for sample_idx in range(next(iter(tensor_dict.values())).shape[0]):
            row_data = {}
            for feature_name, tensor in tensor_dict.items():
                if len(tensor.shape) > 2:
                    tensor = tensor.reshape(tensor.shape[0], -1)

                values = tensor[sample_idx].detach().cpu().numpy()
                if len(values.shape) == 0:
                    row_data[f"{feature_name}"] = values.item()
                else:
                    for _, value in enumerate(values):
                        row_data[f"{feature_name}"] = value
            all_data.append(row_data)
        return pd.DataFrame(all_data)

    discrete_df = dict_to_df(discrete_samples)
    continuous_df = dict_to_df(continuous_samples)
    binary_df = dict_to_df(binary_samples)

    combined_df = pd.concat([discrete_df, continuous_df, binary_df], axis=1)
    combined_df.to_csv(f'{file_name}.csv', index=False)

    return combined_df




def train_model(train_loader, encoder, decoder, encoder_opt, decoder_opt):
    encoder.train()
    decoder.train()
    discrete_samples = {feature: [] for feature in decoder.discrete_features}
    continuous_samples = {feature: [] for feature in decoder.continuous_features}
    binary_samples = {feature: [] for feature in decoder.binary_features}
    losses = 0.0
    for i, (X, y) in enumerate(train_loader):
        X = X.type(torch.FloatTensor)
        continuous_targets = {}
        binary_targets = {}
        discrete_targets = {}

        for feature in decoder.discrete_features:
            continuous_targets[feature] = X[:, :4]

        for feature in decoder.continuous_features:
            continuous_targets[feature] = X[:, 6:]

        for feature in decoder.binary_features:
            binary_targets[feature] = X[:, 4:6]


        encoder_opt.zero_grad()
        decoder_opt.zero_grad()
        encoding = encoder(X)
        discrete_outputs, continuous_outputs, binary_outputs = decoder(encoding)

        loss = decoder.compute_loss((discrete_outputs, continuous_outputs, binary_outputs),
                                             (discrete_targets, continuous_targets, binary_targets))

        loss.backward()
        encoder_opt.step()
        decoder_opt.step()
        losses += loss.item()

        discrete_samples, continuous_samples, binary_samples = utils.types_append(decoder,
                                                                                  discrete_outputs, continuous_outputs,
                                                                                  binary_outputs, discrete_samples,
                                                                                  continuous_samples,
                                                                                  binary_samples)

    discrete_samples, continuous_samples, binary_samples = utils.type_concat(decoder, discrete_samples,
                                                                             continuous_samples, binary_samples)

    total_loss = losses / len(train_loader)

    return total_loss, discrete_samples, continuous_samples, binary_samples


def evaluate_model(val_loader, encoder, decoder):
    total_loss = 0
    with torch.no_grad():
        for X, y in val_loader:
            X = X.type(torch.FloatTensor)
            continuous_targets = {}
            binary_targets = {}
            discrete_targets = {}

            for feature in decoder.discrete_features:
                continuous_targets[feature] = X[:, :4]

            for feature in decoder.continuous_features:
                continuous_targets[feature] = X[:, 6:]

            for feature in decoder.binary_features:
                binary_targets[feature] = X[:, 4:6]

            encoding = encoder(X)
            discrete_outputs, continuous_outputs, binary_outputs = decoder(encoding)

            loss = decoder.compute_loss((discrete_outputs, continuous_outputs, binary_outputs),
                                        (discrete_targets, continuous_targets, binary_targets))
            total_loss += loss.item()

        avg_loss = total_loss / len(val_loader)

    return avg_loss

def test_model(test_loader, state_path, encoder, decoder):
    encoder.load_state_dict(torch.load(f"{state_path}")["enc"])
    decoder.load_state_dict(torch.load(f"{state_path}")["dec"])
    total_loss = 0
    with torch.no_grad():
        for X, y in test_loader:
            X = X.type(torch.FloatTensor)
            continuous_targets = {}
            binary_targets = {}
            discrete_targets = {}

            for feature in decoder.discrete_features:
                continuous_targets[feature] = X[:, :4]

            for feature in decoder.continuous_features:
                continuous_targets[feature] = X[:, 6:]

            for feature in decoder.binary_features:
                binary_targets[feature] = X[:, 4:6]

            encoding = encoder(X)
            discrete_outputs, continuous_outputs, binary_outputs = decoder(encoding)

            loss = decoder.compute_loss((discrete_outputs, continuous_outputs, binary_outputs),
                                        (discrete_targets, continuous_targets, binary_targets))
            total_loss += loss.item()

        avg_loss = total_loss / len(test_loader)
    return avg_loss




