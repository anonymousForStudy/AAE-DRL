import argparse
import csv
import os
import sys

import numpy as np
import torch
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import utils

from AEDQN import trainDQN, testDQN, custom_env

# check cuda and block DSA if error
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
os.environ['TORCH_USE_CUDA_DSA'] = "1"

cuda = True if torch.cuda.is_available() else False


def parse_args(args):
    parser = argparse.ArgumentParser()

    # Trainer args
    parser.add_argument('--batch_size_train', default=32, type=int)
    parser.add_argument("--max_timestep", default=4000, type=int)
    parser.add_argument("--eval_freq", default=400, type=int)
    parser.add_argument("--start_timestep", default=50, type=int)
    parser.add_argument("--max_ep_steps", default=100, type=int)

    # Tester args
    parser.add_argument('--batch_size_test', default=64, type=int)
    parser.add_argument('--numEpochs', default=100, type=int)

    # if test: ---train False
    parser.add_argument('--train', action='store_true')
    # unaug = unaugmented dataset = original dataset : if False then augmented dataset
    parser.add_argument("--unaug_dataset", default=True)

    # save samples and state dictionaries
    parser.add_argument("--rl_dataset", default="rl_ds2.csv")  # newly generated dataset
    parser.add_argument("--DQL2_state", default="DQL2.pth")  # state dictionary

    return parser.parse_args(args)


if __name__ == "__main__":
    args = sys.argv[1:]
    args = parse_args(args)
    dataset = utils.dataset(original=args.unaug_dataset, train=args.train)

    if args.train:
        # Train DRL
        train_loader, val_loader = utils.dataset_function(dataset, batch_size_t=args.batch_size_train,
                                                          batch_size_o=args.batch_size_test, train=True)
        d, c, b = trainDQN.Train(train_loader, val_loader, trainDQN.decoder, trainDQN.classifierA,
                                     trainDQN.classifierB,
                                     args.max_timestep, args.batch_size_train, args.eval_freq,
                                     args.start_timestep, args.max_ep_steps, args.DQL2_state
                                     ).train()

        # Convert from torch tensor to numpy array
        d_dict = {key: tensor.detach().cpu().numpy() for key, tensor in d.items()}
        c_dict = {key: tensor.detach().cpu().numpy() for key, tensor in c.items()}
        b_dict = {key: tensor.detach().cpu().numpy() for key, tensor in b.items()}

        # Round discrete and binary feature values
        d_max = {key: np.argmax(value, axis=1) for key, value in d_dict.items()}
        b_max = {key: np.argmax(value, axis=1) for key, value in b_dict.items()}

        all_dict = {**d_max, **c_dict, **b_max}

        # Save samples
        with open(f'{args.rl_dataset}', 'w', newline='') as file_d:
            writer = csv.writer(file_d)
            keys = list(all_dict.keys())
            writer.writerow(keys)
            max_len = max(len(all_dict[key]) for key in keys)
            for i in range(max_len):
                row = [all_dict[key][i] if i < len(all_dict[key]) else '' for key in keys]
                writer.writerow(row)


    else:
        # Test DRL
        test_loader = utils.dataset_function(dataset, args.batch_size_train, args.batch_size_test, train=False)

        tester = testDQN.Test(test_loader, trainDQN.encoder, trainDQN.decoder,
                                  trainDQN.classifierA, trainDQN.classifierB,
                                  args.DQL2_state)
        evaluater = tester.evaluate()
        for i in range(args.numEpochs):
            next(evaluater)


