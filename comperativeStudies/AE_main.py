import argparse
import os
import sys

import torch
import AE_trainer
from comperativeStudies.RLGAN import gan

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import utils
from AEDQN import autoencoder
from RLGAN import AE, gan_trainer, gan_tester


def parse_args(args):
    parser = argparse.ArgumentParser(description='Select which model to run: RL-GAN or AE+DQN [REQUIRED]')

    parser.add_argument('--model', choices=['RL-GAN', 'AE+DQN'], required=True)
    parser.add_argument('--batch_size_train', default=32, type=int)
    parser.add_argument('--batch_size_test', default=64, type=int)
    parser.add_argument('--numEpochs', default=101, type=int)
    # when the discriminator loss reaches a threshold, we save the AAE state dictionary
    parser.add_argument("--loss_threshold", default=0.6, type=float)


    # if test: ---train False
    parser.add_argument('--train', action='store_true')
    # unaug = unaugmented dataset = original dataset : if False then augmented dataset
    parser.add_argument("--unaug_dataset", action="store_true")
    parser.add_argument("--dataset_file", default="ds2.csv")
    # PLEASE USE THE ABSOLUTE PATH IF YOU GET A NO FILE IS FOUND!!!
    # Save AAE state dictionary
    parser.add_argument("--save_state_dict", default="ae2.pth")
    parser.add_argument("--gan_state_dict", default="gan.pth")

    # Path to augmented dataset
    parser.add_argument('--X_ds', default="rl1.csv")
    # Path to augmented dataset's labels
    parser.add_argument('--y_ds', default="labels1.csv")

    return parser.parse_args(args)

if __name__ == "__main__":
    args = sys.argv[1:]
    args = parse_args(args)
    print("using", args.unaug_dataset)
    dataset = utils.dataset(original=args.unaug_dataset, train=args.train)

    if args.model == "RL-GAN":
        encoder = AE.Encoder()
        decoder = AE.Decoder(10, 30, 64, utils.discrete, utils.continuous, utils.binary)
    else:
        encoder = autoencoder.Encoder()
        decoder = autoencoder.Decoder(utils.discrete, utils.continuous, utils.binary)
    encoder_opt = torch.optim.SGD(encoder.parameters(), lr=0.001)
    decoder_opt = torch.optim.SGD(decoder.parameters(), lr=0.001)

    generator = gan.Generator()
    discriminator = gan.Discriminator()
    optimizer_G = torch.optim.SGD(generator.parameters(), lr=0.001, momentum=0.9)
    optimizer_D = torch.optim.SGD(discriminator.parameters(), lr=0.001, momentum=0.9)

    if args.train:
        train_loader, val_loader = utils.dataset_function(dataset, batch_size_t=args.batch_size_train,
                                                          batch_size_o=args.batch_size_test, train=True)

        best_d_val_loss = args.loss_threshold
        for epoch in range(args.numEpochs):
            # Train AAE
            total_loss, discrete_samples, continuous_samples, binary_samples = AE_trainer.train_model(train_loader, encoder, decoder, encoder_opt, decoder_opt)
            print(f"Epoch {epoch + 1}/{args.numEpochs}, loss: {total_loss}")
            # Evaluate AAE
            if epoch % 10 == 0:
                loss_val = AE_trainer.evaluate_model(val_loader, encoder, decoder)
                print(f"eval loss: {loss_val}")
                # Save state dictionary
                torch.save({'epoch': epoch,
                            'enc': encoder.state_dict(),
                            'dec': decoder.state_dict(),
                            }, f"{args.save_state_dict}")

        # Generate samples and save
        AE_trainer.save_features_to_csv(discrete_samples, continuous_samples, binary_samples)

        if args.model == "RL-GAN":
            for epoch in range(101):
                g_loss, d_loss = gan_trainer.train_model(train_loader, generator, discriminator, optimizer_D, optimizer_G)
                print(f"Epoch {epoch + 1}/{101}, g loss: {g_loss}, d loss: {d_loss}")
                if epoch % 10 == 0:
                    g_val, d_val = gan_trainer.evaluate_model(val_loader, generator, discriminator)
                    print(f"g val loss: {g_val}, d val loss: {d_val}")
                    if d_loss < best_d_val_loss:
                        best_d_val_loss = d_val
                        torch.save({'epoch': epoch,
                                    'gen': generator.state_dict(),
                                    'disc': discriminator.state_dict(),
                                    'val_loss': d_val}, f"{args.gan_state_dict}")

    else:
        # Test AAE
        test_loader = utils.dataset_function(dataset, batch_size_t=args.batch_size_train,
                                             batch_size_o=args.batch_size_test, train=False)
        test_loss = AE_trainer.test_model(test_loader, args.save_state_dict, encoder, decoder)
        print(f"g_loss: {test_loss}")
        if args.model == "RL-GAN":
            test_loss=gan_tester.test_model(test_loader, args.gan_state_dict, generator, discriminator)




# X_synth = pd.DataFrame(pd.read_csv("/first_approach/rl_1st.csv"))
# X_synth = X_synth.apply(lambda col: col.str.strip("[]").astype(float) if col.dtype == "object" else col)
#
# labels_synth = pd.DataFrame(pd.read_csv("/first_approach/labels_1st.csv"))
# X_all = pd.concat([X_synth, X_train_sc, X_test_sc], axis=0)
# y_all = pd.concat([labels_synth, y_train, y_test], axis=0)
# X_train_all, X_test_all, y_train_all, y_test_all = train_test_split(X_all, y_all, test_size=0.2, random_state=45)
# Encoded_data_all = CustomDataset(X_train_all.to_numpy(), y_train_all.to_numpy())
