import argparse
import subprocess
import sys

import torch

import utils
from AAE import AAE_training, AAE_testing
import mlflow
from mlflow.models import infer_signature



def parse_args(args):
    parser = argparse.ArgumentParser()

    parser.add_argument('--batch_size_train', default=32, type=int)
    parser.add_argument('--batch_size_test', default=64, type=int)
    parser.add_argument('--numEpochs', default=101, type=int)
    # when the discriminator loss reaches a threshold, we save the AAE state dictionary
    parser.add_argument("--loss_threshold", default=0.5, type=float)
    # define number of interpolations and sample size per interpolation
    # !!!DESIRED DATASIZE = number of interpolations * sample size per interpolation!!!
    parser.add_argument("--n_inter", default=5, type=int) # we set it to 4 when --unaug_dataset = False
    parser.add_argument("--n_samples_per_inter", default=27321, type=int) # we set it to 43313 when --unaug_dataset = False


    # if test: ---train False
    parser.add_argument('--train', action='store_true')
    # unaug = unaugmented dataset = original dataset : if False then augmented dataset
    parser.add_argument("--unaug_dataset", action="store_true")
    parser.add_argument("--dataset_file", default="ds.csv")
    # PLEASE USE THE ABSOLUTE PATH IF YOU GET A NO FILE IS FOUND!!!
    parser.add_argument("--save_state_dict", default="aae3.pth")
    parser.add_argument('--X_ds', default="rl_ds.csv")
    parser.add_argument('--y_ds', default="labels.csv")

    return parser.parse_args(args)

if __name__ == "__main__":
    args = sys.argv[1:]
    args = parse_args(args)
    process = subprocess.Popen(
        ["mlflow", "server", "--host", "127.0.0.1", "--port", "8080"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")
    print("using", args.unaug_dataset)
    dataset = utils.dataset(original=args.unaug_dataset, train=args.train)

    if args.train:
        train_loader, val_loader = utils.dataset_function(dataset, batch_size_t=args.batch_size_train,
                                                          batch_size_o=args.batch_size_test, train=True)
        best_d_val_loss = args.loss_threshold
        mlflow.set_experiment("AAE")
        with mlflow.start_run():
            for epoch in range(args.numEpochs):
                g_loss, d_loss = AAE_training.train_model(train_loader)
                print(f"Epoch {epoch + 1}/{args.numEpochs}, g loss: {g_loss}, d loss: {d_loss}")
                if epoch % 10 == 0:
                    g_val, d_val = AAE_training.evaluate_model(val_loader)
                    mlflow.log_metric("g val", g_val, step=epoch)
                    mlflow.log_metric("d val", d_val, step=epoch)
                    print(f"g loss: {g_val}, d loss: {d_val}")
                    if d_val < best_d_val_loss:
                        best_d_val_loss = d_val
                        torch.save({'epoch': epoch,
                                    'enc_gen': AAE_training.encoder_generator.state_dict(),
                                    'dec': AAE_training.decoder.state_dict(),
                                    "disc": AAE_training.discriminator.state_dict(),
                                    'val_loss': d_loss}, f"{args.save_state_dict}")

            model_info_gen = mlflow.pytorch.log_model(
                pytorch_model = AAE_training.encoder_generator,
                artifact_path="mlflow/gen",
                input_example=30,
                registered_model_name="G_tracking",
            )
            model_info_disc = mlflow.pytorch.log_model(
                pytorch_model=AAE_training.discriminator,
                artifact_path="mlflow/discriminator",
                input_example=10,
                registered_model_name="D_tracking",
            )

        d, c, b = AAE_training.sample_runs(args.n_inter, args.n_samples_per_inter)
        AAE_training.save_features_to_csv(d, c, b, args.dataset_file)

    else:
        test_loader = utils.dataset_function(dataset, batch_size_t=args.batch_size_train,
                                             batch_size_o=args.batch_size_test, train=False)
        g_loss, d_loss = AAE_testing.test_model(test_loader, args.save_state_dict)
        print(f"g_loss: {g_loss}, d_loss: {d_loss}")
